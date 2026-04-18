// =================================================================================================
// Photoreal HDR Color Grader (V5.9.7 - The Ultimate Director's Cut)
// =================================================================================================
//
// Design Philosophy: PRECISION AND QUALITY OVER PERFORMANCE
// - True IEEE 754 Math: No fast intrinsics or Special Function Unit (SFU) approximations.
// - Exact IEC/SMPTE Constants: Bit-exact neutrality logic for standard color spaces.
// - True Stop-Domain Scene Grading: Log2-domain exposure and contrast with C1 rational recovery.
// - Physiological Chromaticity: MacLeod-Boynton cone-opponent space for all color operations.
//
// V5.9.7 Changes from V5.9.6:
// - Added: Analytical MacLeod-Boynton Gamut Guard (Stage 6.5).
//          Mathematically exact hue-preserving soft-knee compression. Zero loops.
//          Analytically solves the ray-gamut boundary intersection in MB chromaticity space.
//          Preserves luminance exactly. Prevents out-of-gamut clipping from saturation/contrast.
// - Added: "Gamut Guard Knee" slider for user control over boundary compression threshold.
// - Fix: Fast-bypass guard updated to include new Gamut Guard parameter.
//
// V5.9.6 Changes from V5.9.5:
// - Fix: Dehaze / Black Point no longer crushes shadow noise floor to pure black.
// - Added: "Dehaze Shadow Floor" slider for user control over the crush floor.
// - Optimize: Eliminated double LMS matrix multiply in ApplyMBPurity (1x instead of 2x).
//
// V5.9.5 Changes from V5.9.4:
// - Added: Debug Visualization System (9 modes).
// - Fix: SRGB_THRESHOLD_OETF unified to derived value (0.04045/12.92).
// =================================================================================================

#include "ReShade.fxh"

// =================================================================================================
// 1. Constants & Definitions
// =================================================================================================

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

static const float FLT_MIN              = 1.175494351e-38;
static const float SCRGB_WHITE_NITS     = 80.0;
static const float NEUTRAL_EPS          = 1e-6;
static const float PI                   = 3.14159265358979323846;

// -------------------------------------------------------------------------------------------------
// sRGB Constants (IEC 61966-2-1:1999)
// -------------------------------------------------------------------------------------------------
static const float SRGB_THRESHOLD_EOTF  = 0.04045;

// Derived from EOTF threshold (0.04045 / 12.92) for perfect round-trip with
// companion shader (Bilateral Contrast v8.4.5+).
static const float SRGB_THRESHOLD_OETF  = 0.04045 / 12.92;

static const float SRGB_GAMMA           = 2.4;
static const float SRGB_INV_GAMMA       = 0.41666666666666667; // 1/2.4 = 5/12

// -------------------------------------------------------------------------------------------------
// ST.2084 (PQ) EOTF Constants (SMPTE ST 2084:2014)
// -------------------------------------------------------------------------------------------------
static const float PQ_M1                = 0.1593017578125;
static const float PQ_M2                = 78.84375;
static const float PQ_C1                = 0.8359375;
static const float PQ_C2                = 18.8515625;
static const float PQ_C3                = 18.6875;
static const float PQ_PEAK_LUMINANCE    = 10000.0;
static const float PQ_INV_M1            = 6.2773946360153257;
static const float PQ_INV_M2            = 0.012683313515655966;

// -------------------------------------------------------------------------------------------------
// Color Science Constants
// -------------------------------------------------------------------------------------------------
static const float CHROMA_STABILITY_THRESH      = 1e-4;
static const float CHROMA_RELIABILITY_START     = 5e-5;
static const float INV_CHROMA_RELIABILITY_SPAN  = 1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);

static const float3 Luma709             = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020            = float3(0.2627, 0.6780, 0.0593);

static const float MB_PURITY_PROTECTION_CEILING = 0.35;

// -------------------------------------------------------------------------------------------------
// Biological Bleaching Constants
// -------------------------------------------------------------------------------------------------
static const float TROLAND_LMS_SCALE    = 4.0;
static const float TROLAND_HALF_SAT     = 8000.0;

// -------------------------------------------------------------------------------------------------
// Scene-Grade Row-Sum-Normalized Matrices
// -------------------------------------------------------------------------------------------------
static const float3x3 RGB709_to_LMS = float3x3(
    0.4122214708,  0.5363325363,  0.0514459929,
    0.2119034982,  0.6806995451,  0.1073969566,
    0.0883024619,  0.2817188376,  0.6299787005
);

static const float3x3 LMS_to_RGB709 = float3x3(
     4.0767416621, -3.3077115913,  0.2309699292,
    -1.2684380046,  2.6097574011, -0.3413193965,
    -0.0041960863, -0.7034186147,  1.7076147010
);

static const float3x3 RGB2020_to_LMS = float3x3(
    0.6167596970,  0.3601880240,  0.0230522790,
    0.2651316740,  0.6358515800,  0.0990167460,
    0.1001279150,  0.2038783840,  0.6959937010
);

static const float3x3 LMS_to_RGB2020 = float3x3(
     2.1398540771, -1.2462788877,  0.1064290765,
    -0.8846737634,  2.1631158093, -0.2784377818,
    -0.0486976682, -0.4543507342,  1.5030526721
);

// -------------------------------------------------------------------------------------------------
// Zone System: Mathematically Exact Powers of 2
// Aligned with Bilateral Contrast v8.4.5+ zone definitions.
// Zone V (Middle Grey) anchored at exactly 2^-2.5.
// -------------------------------------------------------------------------------------------------
static const float ZONE_I    = 0.04419417382; // pow(2, -4.5)
static const float ZONE_II   = 0.06250000000; // pow(2, -4.0)
static const float ZONE_III  = 0.08838834764; // pow(2, -3.5)
static const float ZONE_IV   = 0.12500000000; // pow(2, -3.0)
static const float ZONE_V    = 0.17677669529; // pow(2, -2.5)
static const float ZONE_VI   = 0.25000000000; // pow(2, -2.0)
static const float ZONE_VII  = 0.35355339059; // pow(2, -1.5)
static const float ZONE_VIII = 0.50000000000; // pow(2, -1.0)
static const float ZONE_IX   = 0.70710678118; // pow(2, -0.5)
static const float ZONE_X    = 1.00000000000; // pow(2, 0.0)
static const float ZONE_XI   = 2.00000000000; // pow(2, 1.0)

// =================================================================================================
// 2. Texture & Sampler
// =================================================================================================

texture2D TextureBackBuffer : COLOR;
sampler2D SamplerBackBuffer
{
    Texture   = TextureBackBuffer;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = NONE;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

// =================================================================================================
// 3. UI Parameters
// =================================================================================================

// -------------------------------------------------------------------------------------------------
// UI: Part 1 - Scene Grade
// -------------------------------------------------------------------------------------------------
uniform float fExposure <
    ui_type     = "slider";
    ui_min      = -3.00; ui_max = 3.00; ui_step = 0.01;
    ui_label    = "Exposure (EV)";
    ui_tooltip  = "Linear EV shift: multiply by 2^EV.\n"
                  "+1.0 EV = double brightness, -1.0 EV = half brightness.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fTemperature <
    ui_type     = "slider";
    ui_min      = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label    = "Color Temperature (LMS)";
    ui_tooltip  = "Negative = Cooler (removes yellow/sand tint)\n"
                  "Positive = Warmer\n"
                  "Uses exponential gain (always positive, no channel collapse).\n"
                  "Luminance-preserving for neutral tones.\n"
                  "Saturated colors may shift ~1-3%% in luminance.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fTint <
    ui_type     = "slider";
    ui_min      = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label    = "Color Tint (LMS)";
    ui_tooltip  = "Negative = Greener\nPositive = More Magenta";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fBlackPoint <
    ui_type     = "slider";
    ui_min      = 0.000; ui_max = 0.050; ui_step = 0.001;
    ui_label    = "Dehaze / Black Point";
    ui_tooltip  = "Subtracts a percentage of reference white from the entire luminance range.\n"
                  "Removes atmospheric haze / dusty lifted blacks.\n"
                  "Uses a smooth C1 curve with a shadow floor to avoid hard contours\n"
                  "and prevent total black crush.\n"
                  "0.003 = 0.3%% of white.";
    ui_category = "1. Scene Grade";
> = 0.000;

uniform float fShadowFloor <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.005;
    ui_label    = "Dehaze Shadow Floor";
    ui_tooltip  = "Minimum residual luminance ratio for Dehaze. Prevents total black crush.\n"
                  "0.03 = default (invisible but preserves dither and shadow detail).\n"
                  "0.00 = maximum crush (can clip to pure black, may cause banding).";
    ui_category = "1. Scene Grade";
> = 0.03;

uniform float fContrast <
    ui_type     = "slider";
    ui_min      = 0.80; ui_max = 1.50; ui_step = 0.001;
    ui_label    = "Filmic Contrast";
    ui_tooltip  = "Luminance-based power curve pivoted at 18%% grey.\n"
                  "Preserves chromaticity by applying a scalar ratio to RGB.\n"
                  "Handles negative-luminance scRGB via absolute value.";
    ui_category = "1. Scene Grade";
> = 1.00;

uniform float fContrastPivot <
    ui_type     = "slider";
    ui_min      = 0.01; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Contrast Pivot";
    ui_tooltip  = "The luminance value that remains unchanged when contrast is adjusted.\n"
                  "0.18 = Photographic middle gray (default, filmic look).\n"
                  "Lower values: more shadow modification, less highlight modification.\n"
                  "Higher values: less shadow modification, more highlight modification.\n"
                  "Does not prevent highlight expansion — use the Highlights slider for that.";
    ui_category = "1. Scene Grade";
> = 0.18;

uniform float fShadows <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Shadows (Log Recovery)";
    ui_tooltip  = "Lifts or deepens shadow detail in the stop domain.\n"
                  "+1.0 = Lift up to 3 stops (recover shadow detail).\n"
                  "-1.0 = Deepen up to 3 stops (crush shadows for mood).\n"
                  "Operates below the Contrast Pivot. Zero effect at the pivot.\n"
                  "C1 continuous with the contrast curve (seamless transition).";
    ui_category = "1. Scene Grade";
> = 0.0;

uniform float fHighlights <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Highlights (Log Recovery)";
    ui_tooltip  = "Protects (-1.0) or boosts (+1.0) highlights.\n"
                  "Use negative values to recover detail blown out by high contrast.\n"
                  "Operates as a mathematically smooth shoulder curve.";
    ui_category = "1. Scene Grade";
> = 0.0;

uniform float fSaturation <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 2.00; ui_step = 0.01;
    ui_label    = "Purity / Saturation (MacLeod-Boynton)";
    ui_tooltip  = "Strictly isoluminant saturation in physiological MacLeod-Boynton space.\n"
                  "Alters chromaticity distance from D65 white without changing luminance (L+M).\n"
                  "Above 1.0: vibrance-style boost (protects already vivid colors from clipping).\n"
                  "Below 1.0: uniform purity reduction toward neutral.\n"
                  "Near-black pixels fade toward neutral to prevent math instability.\n"
                  "Rec.2020 receives gentler boost to avoid gamut boundary clipping.";
    ui_category = "1. Scene Grade";
> = 1.08;

uniform float fGamutGuardKnee <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label    = "Gamut Guard Knee";
    ui_tooltip  = "Analytical soft-knee gamut boundary compression in MacLeod-Boynton space.\n"
                  "Prevents out-of-gamut colors from saturation/contrast/exposure boosts.\n"
                  "Guaranteed hue-preserving (radial from D65) and luminance-preserving.\n\n"
                  "0.00 = off (no protection).\n"
                  "0.10 = subtle (start compressing at 90%% of max purity).\n"
                  "0.30 = aggressive (start compressing at 70%% of max purity).\n\n"
                  "Uses exact analytical ray-boundary intersection (zero loops).";
    ui_category = "1. Scene Grade";
> = 0.10;

// -------------------------------------------------------------------------------------------------
// UI: Part 2 - Tone Mapping
// -------------------------------------------------------------------------------------------------
uniform float fBleaching <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Highlight Bleaching (Trolands)";
    ui_tooltip  = "Physiological highlight burnout toward a white-hot core.\n"
                  "Based on absolute scene intensity (nits), before tonemapping.\n\n"
                  "Mimics photopigment depletion: as retinal illuminance increases,\n"
                  "available photopigment decreases, reducing color purity.\n\n"
                  "Approximate purity reduction at full strength (1.00):\n"
                  "  203 nits  (BT.2408 white):  ~9%%  — subtle, midtones untouched\n"
                  "  1000 nits (HDR specular):    ~33%% — clearly visible burnout\n"
                  "  4000 nits (bright highlight): ~67%% — strong white-hot core\n"
                  "  10000 nits (PQ peak):        ~83%% — near-complete bleaching\n\n"
                  "Hue is preserved exactly: radial scaling from D65 in MB space.";
    ui_category = "2. Tone Mapping";
> = 0.80;

uniform bool bEnableKhronosNeutral <
    ui_label    = "Enable Khronos PBR Neutral Tonemapper";
    ui_tooltip  = "Applies strict hue-preserving highlight compression.\n"
                  "Prevents hard-clipping and color shifts in extreme highlights.\n\n"
                  "Hue preservation is mathematically guaranteed: Khronos applies\n"
                  "a uniform per-pixel affine transform to all RGB channels, which\n"
                  "preserves the MacLeod-Boynton chromaticity direction from D65\n"
                  "through row-sum-normalized LMS matrices.";
    ui_category = "2. Tone Mapping";
> = true;

uniform float fDisplayPeakNits <
    ui_type     = "slider";
    ui_min      = 80.0; ui_max = 4000.0; ui_step = 10.0;
    ui_label    = "Display Peak Luminance (Nits)";
    ui_tooltip  = "The maximum brightness your display can output.\n"
                  "For SDR, this is ignored (locked to 1.0x Reference White).";
    ui_category = "2. Tone Mapping";
> = 800.0;

uniform float fCompressionStart <
    ui_type     = "slider";
    ui_min      = 0.50; ui_max = 0.95; ui_step = 0.01;
    ui_label    = "Compression Start (%)";
    ui_tooltip  = "Where to start rolling off highlights (percentage of Peak).\n"
                  "0.80 = 1:1 color mapping up to 80%% of peak display brightness.";
    ui_category = "2. Tone Mapping";
> = 0.80;

uniform float fDesaturationStrength <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label    = "Khronos Desaturation (Legacy)";
    ui_tooltip  = "Recommended: 0.00. Use Highlight Bleaching instead.\n"
                  "This performs legacy math-based desaturation near display peak.\n\n"
                  "Note: Desaturation reduces purity (distance from white) but\n"
                  "cannot alter hue — the lerp toward a uniform scalar preserves\n"
                  "the MacLeod-Boynton chromaticity direction exactly.";
    ui_category = "2. Tone Mapping";
> = 0.00;

// -------------------------------------------------------------------------------------------------
// UI: System
// -------------------------------------------------------------------------------------------------
uniform int iColorSpaceOverride <
    ui_type     = "combo";
    ui_label    = "Color Space Override";
    ui_items    = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0";
    ui_tooltip  = "Must match Bilateral Contrast v8.4.4+.";
    ui_category = "System";
> = 0;

uniform float fWhitePoint <
    ui_type     = "slider";
    ui_min      = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_label    = "Reference White (Nits)";
    ui_tooltip  = "Should match Zone White Point in Bilateral Contrast.\n"
                  "SDR stays fixed at 80 nits.\n"
                  "203 = ITU-R BT.2408 reference diffuse white.";
    ui_category = "System";
> = 203.0;

uniform int iDebugMode <
    ui_type     = "combo";
    ui_label    = "Debug Visualization";
    ui_items    = "Off\0"
                  "Luminance (False Color Stops)\0"
                  "Zone Map\0"
                  "Bleaching Factor\0"
                  "MB Purity\0"
                  "MB Hue Wheel\0"
                  "LMS Cone Response\0"
                  "Negative / WCG\0"
                  "Compression Map\0";
    ui_tooltip  = "Debug visualizations operate on the fully graded output.\n"
                  "All outputs are display-linear [0,1] encoded per color space.\n\n"
                  "1. Luminance: False-color heatmap of stops from white (-8 to +8).\n"
                  "2. Zone Map: Ansel Adams zone coloring (aligned with Bilateral Contrast).\n"
                  "3. Bleaching: k factor per pixel (blue=1.0 untouched, red=0.0 fully bleached).\n"
                  "4. MB Purity: Distance from D65 white in MB chromaticity (0=neutral, 0.35+=boundary).\n"
                  "5. MB Hue: Hue angle as color wheel. Dark = achromatic, bright = saturated.\n"
                  "6. LMS: Cone responses as RGB (L=red, M=green, S=blue). Normalized to peak.\n"
                  "7. Negative: Highlights out-of-gamut channels (R=magenta, G=cyan, B=yellow).\n"
                  "8. Compression: Khronos compression ratio (1.0=none, 0.0=max).";
    ui_category = "Debug";
> = 0;

// =================================================================================================
// 4. True Math Utilities (IEEE 754 Compliant)
// =================================================================================================

float PowNonNegPreserveZero(float x, float e)
{
    if (x <= 0.0) return 0.0;
    return pow(x, e);
}

float3 PowNonNegPreserveZero3(float3 x, float e)
{
    return float3(
        PowNonNegPreserveZero(x.r, e),
        PowNonNegPreserveZero(x.g, e),
        PowNonNegPreserveZero(x.b, e)
    );
}

float SqrtIEEE(float x)
{
    return PowNonNegPreserveZero(x, 0.5);
}

bool IsNanVal(float x)   { return (asuint(x) & 0x7FFFFFFF) > 0x7F800000; }
bool IsInfVal(float x)   { return (asuint(x) & 0x7FFFFFFF) == 0x7F800000; }
bool3 IsNan3(float3 v)   { return bool3(IsNanVal(v.x), IsNanVal(v.y), IsNanVal(v.z)); }
bool3 IsInf3(float3 v)   { return bool3(IsInfVal(v.x), IsInfVal(v.y), IsInfVal(v.z)); }

// =================================================================================================
// 5. Color Science & EOTF Utilities
// =================================================================================================

float3 sRGB_EOTF(float3 V)
{
    float3 abs_V  = abs(V);
    float3 lin_lo = abs_V / 12.92;
    float3 lin_hi = PowNonNegPreserveZero3((abs_V + 0.055) / 1.055, SRGB_GAMMA);

    float3 out_lin;
    out_lin.r = (abs_V.r <= SRGB_THRESHOLD_EOTF) ? lin_lo.r : lin_hi.r;
    out_lin.g = (abs_V.g <= SRGB_THRESHOLD_EOTF) ? lin_lo.g : lin_hi.g;
    out_lin.b = (abs_V.b <= SRGB_THRESHOLD_EOTF) ? lin_lo.b : lin_hi.b;
    return sign(V) * out_lin;
}

float3 sRGB_OETF(float3 L)
{
    float3 abs_L  = abs(L);
    float3 enc_lo = abs_L * 12.92;
    float3 enc_hi = 1.055 * PowNonNegPreserveZero3(abs_L, SRGB_INV_GAMMA) - 0.055;

    float3 out_enc;
    out_enc.r = (abs_L.r <= SRGB_THRESHOLD_OETF) ? enc_lo.r : enc_hi.r;
    out_enc.g = (abs_L.g <= SRGB_THRESHOLD_OETF) ? enc_lo.g : enc_hi.g;
    out_enc.b = (abs_L.b <= SRGB_THRESHOLD_OETF) ? enc_lo.b : enc_hi.b;
    return sign(L) * out_enc;
}

float3 PQ_EOTF(float3 N)
{
    N = saturate(N);
    float3 Np  = PowNonNegPreserveZero3(N, PQ_INV_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    return PowNonNegPreserveZero3(num / den, PQ_INV_M1) * PQ_PEAK_LUMINANCE;
}

float3 PQ_InverseEOTF(float3 L)
{
    L = clamp(L, 0.0, PQ_PEAK_LUMINANCE);
    float3 Lp  = PowNonNegPreserveZero3(L / PQ_PEAK_LUMINANCE, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    return saturate(PowNonNegPreserveZero3(num / den, PQ_M2));
}

float3 DecodeToLinear(float3 encoded, int space)
{
    [branch] if (space == 3) return PQ_EOTF(encoded);
    [branch] if (space == 2) return encoded * SCRGB_WHITE_NITS;
    return sRGB_EOTF(encoded) * SCRGB_WHITE_NITS;
}

float3 EncodeFromLinear(float3 lin, int space)
{
    [branch] if (space == 3) return PQ_InverseEOTF(lin);
    [branch] if (space == 2) return lin / SCRGB_WHITE_NITS;
    return sRGB_OETF(lin / SCRGB_WHITE_NITS);
}

// =================================================================================================
// 6. Mathematical Scene-Grade Functions
// =================================================================================================

float ComputeBlackPointRatio(float luma, float bpNits, float shadowFloor)
{
    if (bpNits <= FLT_MIN || luma <= FLT_MIN) return 1.0;

    // Subtractive ratio: asymptotically → 1.0 for bright pixels
    float raw = max((luma - bpNits) / luma, 0.0);

    // Smooth toe over [0, 4×bpNits] blends floor → raw.
    // C1 at both endpoints: smoothstep derivative = 0 at t=0 and t=1.
    float t = saturate(luma / (4.0 * bpNits));
    float smooth_t = t * t * (3.0 - 2.0 * t);

    return lerp(shadowFloor, raw, smooth_t);
}

float3 ApplyLMSWhiteBalance(float3 color, float temp, float tint, float3 lumaCoeffs, float3x3 to_LMS, float3x3 to_RGB)
{
    float3 wbStops = 0.35 * float3(temp + tint, -tint, -temp + tint);
    float3 wbScale = exp2(wbStops);

    float3 d65_wb_rgb = mul(to_RGB, wbScale);
    float lumaScale   = dot(d65_wb_rgb, lumaCoeffs);
    wbScale /= max(lumaScale, FLT_MIN);

    float3 lms = mul(to_LMS, color);
    lms *= wbScale;

    return mul(to_RGB, lms);
}

// =================================================================================================
// 7. MacLeod-Boynton Physiological Space Functions
// =================================================================================================

float3 LMS_to_MB(float3 lms)
{
    float lum = max(lms.r + lms.g, FLT_MIN);
    return float3(lms.r / lum, lms.b / lum, lum);
}

float3 MB_to_LMS(float3 mb)
{
    return float3(mb.x * mb.z, mb.z - (mb.x * mb.z), mb.y * mb.z);
}

float2 GetMBWhite(float3x3 to_LMS)
{
    float3 white_lms = mul(to_LMS, float3(1.0, 1.0, 1.0));
    return LMS_to_MB(white_lms).xy;
}

float3 ApplyMBPurity(float3 color, float purity_scale, int space, float3 lumaCoeffs, float3x3 to_LMS, float3x3 to_RGB, float2 mb_white)
{
    if (abs(purity_scale - 1.0) < NEUTRAL_EPS) return color;

    float luma = dot(color, lumaCoeffs);
    float ct   = saturate((luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);

    if (chroma_reliability <= 0.0) return color;

    // Single LMS conversion (optimized from 2x to 1x in V5.9.6)
    float3 lms = mul(to_LMS, color);
    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0) return color;

    float3 mb = LMS_to_MB(lms);
    float2 chroma_offset = mb.xy - mb_white;
    float purity = SqrtIEEE(dot(chroma_offset, chroma_offset));

    float effective_scale = purity_scale;

    if (purity_scale > 1.0)
    {
        float protection_t = saturate(purity / MB_PURITY_PROTECTION_CEILING);
        float protection   = protection_t * protection_t * (3.0 - 2.0 * protection_t);

        float boost           = purity_scale - 1.0;
        float space_comp      = (space >= 3) ? 0.90 : 1.0;
        float min_boost_share = (space >= 3) ? 0.20 : 0.25;

        effective_scale = 1.0 + boost * space_comp * lerp(1.0, min_boost_share, protection);
    }

    effective_scale = lerp(1.0, effective_scale, chroma_reliability);
    if (abs(effective_scale - 1.0) < NEUTRAL_EPS) return color;

    mb.xy = lerp(mb_white, mb.xy, effective_scale);

    return mul(to_RGB, MB_to_LMS(mb));
}

// =================================================================================================
// 8. Tonemapping & Gamut Functions
// =================================================================================================

float3 ApplyTrolandBleaching(float3 color, float strength, float3x3 to_LMS, float3x3 to_RGB, float2 mb_white)
{
    if (strength <= NEUTRAL_EPS) return color;

    float3 lms = mul(to_LMS, color);
    float lm_sum = lms.r + lms.g;

    if (lm_sum <= 0.0) return color;

    float3 safe_lms = max(lms, 0.0);
    float3 stimulus = safe_lms * TROLAND_LMS_SCALE;
    float stim_lm   = 0.5 * (stimulus.r + stimulus.g);

    float availability = 1.0 / (1.0 + (stim_lm / max(TROLAND_HALF_SAT, FLT_MIN)));
    float k = lerp(1.0, availability, saturate(strength));

    float3 mb = LMS_to_MB(lms);
    mb.xy = lerp(mb_white, mb.xy, k);

    return mul(to_RGB, MB_to_LMS(mb));
}

float3 ApplyKhronosPBRNeutral(float3 color, float targetPeak, float compressionStart, float desatStrength)
{
    float3 safeColor = max(color, 0.0);
    float x = min(safeColor.r, min(safeColor.g, safeColor.b));

    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    float peak = max(safeColor.r, max(safeColor.g, safeColor.b)) - offset;
    float startComp = (targetPeak * compressionStart) - 0.04;

    [branch]
    if (peak >= startComp && startComp > 0.0)
    {
        float d = targetPeak - startComp;
        float newPeak = targetPeak - (d * d) / (peak + d - startComp);

        float3 working = color - offset;
        float ratio = newPeak / max(peak, FLT_MIN);
        working *= ratio;

        float t = saturate((newPeak - startComp) / max(d, FLT_MIN));
        float g = desatStrength * t * t;
        working = lerp(working, newPeak.xxx, g);

        return working + offset;
    }

    return color;
}

// -------------------------------------------------------------------------------------------------
// 8.1 Analytical MB Gamut Guard (V5.9.7)
// -------------------------------------------------------------------------------------------------
// Mathematically exact hue-preserving soft-knee gamut compression.
// Solves the ray-gamut boundary intersection analytically (zero loops).
//
// Derivation:
//   In MB space, chromaticity is scaled radially from white:
//     mb.xy = mb_white + t * (mb.xy - mb_white)
//   In LMS space:
//     L = (wx + t*dx) * lum,  M = (1 - wx - t*dx) * lum,  S = (wy + t*dy) * lum
//   In RGB space:
//     rgb_i = lum * [ (to_RGB[i][0] - to_RGB[i][1])*(wx + t*dx) + to_RGB[i][1] + to_RGB[i][2]*(wy + t*dy) ]
//   The constraint rgb_i >= 0 reduces to:
//     A_i * t + B_i >= 0
//   where:
//     A_i = dx*(to_RGB[i][0] - to_RGB[i][1]) + dy*to_RGB[i][2]
//     B_i = wx*(to_RGB[i][0] - to_RGB[i][1]) + to_RGB[i][1] + wy*to_RGB[i][2]
//   Since B_i = 1/lum for white (which is > 0), the boundary is at t_max = min(-B_i / A_i) for A_i < 0.
// -------------------------------------------------------------------------------------------------

float3 ApplyGamutGuard(float3 color, float knee, float3 lumaCoeffs,
                       float3x3 to_LMS, float3x3 to_RGB, float2 mb_white)
{
    if (knee <= FLT_MIN) return color;

    float luma = dot(color, lumaCoeffs);
    float ct = saturate((luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float reliability = ct * ct * (3.0 - 2.0 * ct);
    if (reliability <= 0.0) return color;

    float3 lms = mul(to_LMS, color);
    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0) return color;

    float3 mb = LMS_to_MB(lms);
    float2 chroma_offset = mb.xy - mb_white;
    float purity_sq = dot(chroma_offset, chroma_offset);

    if (purity_sq < FLT_MIN) return color; // Achromatic

    float purity = SqrtIEEE(purity_sq);

    // Analytical ray-boundary intersection: find max t where all RGB >= 0
    float dx = chroma_offset.x;
    float dy = chroma_offset.y;
    float wx = mb_white.x;
    float wy = mb_white.y;

    float t_max = 1e10; // Initialize to a very large scale

    // Row 0 (R channel constraint)
    float A0 = dx * (to_RGB[0][0] - to_RGB[0][1]) + dy * to_RGB[0][2];
    float B0 = wx * (to_RGB[0][0] - to_RGB[0][1]) + to_RGB[0][1] + wy * to_RGB[0][2];
    if (A0 < -FLT_MIN) t_max = min(t_max, -B0 / A0);

    // Row 1 (G channel constraint)
    float A1 = dx * (to_RGB[1][0] - to_RGB[1][1]) + dy * to_RGB[1][2];
    float B1 = wx * (to_RGB[1][0] - to_RGB[1][1]) + to_RGB[1][1] + wy * to_RGB[1][2];
    if (A1 < -FLT_MIN) t_max = min(t_max, -B1 / A1);

    // Row 2 (B channel constraint)
    float A2 = dx * (to_RGB[2][0] - to_RGB[2][1]) + dy * to_RGB[2][2];
    float B2 = wx * (to_RGB[2][0] - to_RGB[2][1]) + to_RGB[2][1] + wy * to_RGB[2][2];
    if (A2 < -FLT_MIN) t_max = min(t_max, -B2 / A2);

    // Max purity at this hue angle
    float max_purity = t_max * purity;

    // Soft knee: start compressing at (1.0 - knee) * max_purity
    float threshold = max_purity * (1.0 - knee);

    float3 out_color = color;

    if (purity > threshold && threshold > FLT_MIN)
    {
        float excess = purity - threshold;
        float headroom = max_purity - threshold;

        // Soft asymptote: compressed purity approaches max_purity as excess → ∞
        // f(x) = threshold + headroom * (1 - exp(-x / headroom))
        // Uses true IEEE 754 exp(). Luminance (mb.z) is completely untouched.
        float compressed = threshold + headroom * (1.0 - exp(-excess / max(headroom, FLT_MIN)));
        float scale = compressed / max(purity, FLT_MIN);

        mb.xy = mb_white + chroma_offset * scale;
        out_color = mul(to_RGB, MB_to_LMS(mb));
    }

    // Gentle negative channel clamp (luminance-preserving fallback for extreme edge cases)
    float min_ch = min(min(out_color.r, out_color.g), out_color.b);
    if (min_ch < 0.0)
    {
        float luma_fix = dot(out_color, lumaCoeffs);
        float3 chroma_fix = out_color - luma_fix;
        float ratio = luma_fix / (luma_fix - min_ch + FLT_MIN);
        ratio = min(ratio, 1.0);
        out_color = luma_fix + chroma_fix * ratio;
    }

    return out_color;
}

// =================================================================================================
// 9. Debug Visualization Functions
// =================================================================================================

float3 EncodeDebug(float3 debug_out, int space)
{
    debug_out = max(debug_out, 0.0);
    [branch]
    if (space == 3)
        return PQ_InverseEOTF(debug_out * 1000.0);
    else if (space == 2)
        return debug_out;
    else
        return sRGB_OETF(saturate(debug_out));
}

int GetZone(float nl)
{
    if (nl < 0.0)       return 0;
    if (nl < ZONE_I)    return 1;
    if (nl < ZONE_II)   return 2;
    if (nl < ZONE_III)  return 3;
    if (nl < ZONE_IV)   return 4;
    if (nl < ZONE_V)    return 5;
    if (nl < ZONE_VI)   return 6;
    if (nl < ZONE_VII)  return 7;
    if (nl < ZONE_VIII) return 8;
    if (nl < ZONE_IX)   return 9;
    if (nl < ZONE_X)    return 10;
    if (nl < ZONE_XI)   return 11;
    return 12;
}

float3 GetZoneColor(int index)
{
    [flatten]
    switch (clamp(index, 0, 12))
    {
        case 0:  return float3(0.5,  0.0,  0.5);
        case 1:  return float3(0.02, 0.02, 0.05);
        case 2:  return float3(0.1,  0.0,  0.1);
        case 3:  return float3(0.2,  0.0,  0.3);
        case 4:  return float3(0.3,  0.0,  0.5);
        case 5:  return float3(0.2,  0.2,  0.8);
        case 6:  return float3(0.5,  0.5,  0.5);
        case 7:  return float3(0.8,  0.8,  0.2);
        case 8:  return float3(1.0,  0.8,  0.3);
        case 9:  return float3(1.0,  0.6,  0.4);
        case 10: return float3(1.0,  0.9,  0.8);
        case 11: return float3(1.0,  1.0,  1.0);
        case 12: return float3(1.0,  1.0,  0.5);
        default: return float3(0.0,  0.0,  0.0);
    }
}

float3 StopsToFalseColor(float stops)
{
    float t = saturate((stops + 8.0) / 16.0);

    if (t < 0.2)       return float3(0.0, 0.0, t / 0.2);
    else if (t < 0.4)  return float3(0.0, (t - 0.2) / 0.2, 1.0 - (t - 0.2) / 0.2);
    else if (t < 0.6)  return float3((t - 0.4) / 0.2, 1.0, 0.0);               // green → yellow
    else if (t < 0.8)  return float3(1.0, 1.0 - (t - 0.6) / 0.2, 0.0);
    else               return float3(1.0, (t - 0.8) / 0.2, (t - 0.8) / 0.2);
}

float3 HueToRGB(float hue)
{
    float h = hue * 6.0;
    float i = floor(h);
    float f = h - i;
    float p = 0.0;
    float q = 1.0 - f;
    float t_val = f;

    [flatten]
    switch (int(i) % 6)
    {
        case 0:  return float3(1.0, t_val, p);
        case 1:  return float3(q,   1.0, p);
        case 2:  return float3(p,   1.0, t_val);
        case 3:  return float3(p,   q,   1.0);
        case 4:  return float3(t_val, p,   1.0);
        case 5:  return float3(1.0, p,   q);
        default: return float3(1.0, 1.0, 1.0);
    }
}

float ComputeBleachingK(float3 color, float strength, float3x3 to_LMS)
{
    if (strength <= NEUTRAL_EPS) return 1.0;

    float3 lms = mul(to_LMS, color);
    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0) return 1.0;

    float3 safe_lms = max(lms, 0.0);
    float3 stimulus = safe_lms * TROLAND_LMS_SCALE;
    float stim_lm   = 0.5 * (stimulus.r + stimulus.g);

    float availability = 1.0 / (1.0 + (stim_lm / max(TROLAND_HALF_SAT, FLT_MIN)));
    return lerp(1.0, availability, saturate(strength));
}

float ComputeCompressionRatio(float3 color, float targetPeak, float compressionStart)
{
    float3 safeColor = max(color, 0.0);
    float x = min(safeColor.r, min(safeColor.g, safeColor.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    float peak = max(safeColor.r, max(safeColor.g, safeColor.b)) - offset;
    float startComp = (targetPeak * compressionStart) - 0.04;

    if (peak >= startComp && startComp > 0.0)
    {
        float d = targetPeak - startComp;
        float newPeak = targetPeak - (d * d) / (peak + d - startComp);
        return newPeak / max(peak, FLT_MIN);
    }

    return 1.0;
}

// =================================================================================================
// 10. Main Pipeline Shader
// =================================================================================================

void PS_PhotorealHDR(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 fragColor : SV_Target)
{
    int2 pos   = int2(vpos.xy);
    float4 src = tex2Dfetch(SamplerBackBuffer, pos);

    int space         = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float whitePt     = (space <= 1) ? SCRGB_WHITE_NITS : fWhitePoint;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;

    // Fast-Bypass Guard (skipped when debug is active)
    // ⚠ MAINTENANCE: When adding new pipeline stages, update this bypass check!
    [branch]
    if (iDebugMode == 0 &&
        abs(fExposure) < NEUTRAL_EPS && abs(fBlackPoint) < NEUTRAL_EPS &&
        abs(fContrast - 1.0) < NEUTRAL_EPS && abs(fShadows) < NEUTRAL_EPS &&
        abs(fHighlights) < NEUTRAL_EPS && abs(fTemperature) < NEUTRAL_EPS &&
        abs(fTint) < NEUTRAL_EPS && abs(fSaturation - 1.0) < NEUTRAL_EPS &&
        fBleaching < NEUTRAL_EPS && !bEnableKhronosNeutral &&
        fGamutGuardKnee < NEUTRAL_EPS)
    {
        fragColor = src;
        return;
    }

    // Decode & Sanitize
    float3 original_lin = DecodeToLinear(src.rgb, space);
    if (any(IsNan3(original_lin)) || any(IsInf3(original_lin))) original_lin = 0.0;

    float3 color = original_lin;

    // Matrix resolution
    float3x3 to_LMS, to_RGB;
    [branch]
    if (space >= 3)
    {
        to_LMS = RGB2020_to_LMS;
        to_RGB = LMS_to_RGB2020;
    }
    else
    {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
    }

    float2 mb_white = GetMBWhite(to_LMS);

    // ---------------------------------------------------------------------------------------------
    // STAGE 1: EXPOSURE & WHITE BALANCE
    // ---------------------------------------------------------------------------------------------
    if (abs(fExposure) > NEUTRAL_EPS) color *= exp2(fExposure);

    if (abs(fTemperature) > NEUTRAL_EPS || abs(fTint) > NEUTRAL_EPS)
        color = ApplyLMSWhiteBalance(color, fTemperature, fTint, lumaCoeffs, to_LMS, to_RGB);

    // ---------------------------------------------------------------------------------------------
    // STAGE 2: DEHAZE & BLACK POINT (C1 smooth floor, crush-free)
    // ---------------------------------------------------------------------------------------------
    if (fBlackPoint > NEUTRAL_EPS)
    {
        float luma = dot(color, lumaCoeffs);
        if (luma > FLT_MIN)
        {
            float bpNits = fBlackPoint * whitePt;
            color *= ComputeBlackPointRatio(luma, bpNits, fShadowFloor);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 3: FILMIC CONTRAST & TONAL EQ
    // ---------------------------------------------------------------------------------------------
    if (abs(fContrast - 1.0) > NEUTRAL_EPS || abs(fShadows) > NEUTRAL_EPS || abs(fHighlights) > NEUTRAL_EPS)
    {
        float luma = dot(color, lumaCoeffs);
        float absLuma = abs(luma);

        if (absLuma > FLT_MIN)
        {
            float pivot = fContrastPivot * whitePt;
            float logRatio = log2(absLuma / pivot);

            float x = logRatio * fContrast;
            float a2 = 6.0;

            if (x < 0.0 && abs(fShadows) > NEUTRAL_EPS)
            {
                float S = fShadows * 3.0;
                x = x + S * ((x * x) / (x * x + a2));
            }
            else if (x > 0.0 && abs(fHighlights) > NEUTRAL_EPS)
            {
                float H = fHighlights * 3.0;
                x = x + H * ((x * x) / (x * x + a2));
            }

            float contrastLuma = pivot * exp2(x);
            color *= min(contrastLuma / absLuma, 100.0);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 4: BIOLOGICAL HIGHLIGHT BLEACHING
    // ---------------------------------------------------------------------------------------------
    float3 pre_bleach_color = color;

    color = ApplyTrolandBleaching(color, fBleaching, to_LMS, to_RGB, mb_white);

    // ---------------------------------------------------------------------------------------------
    // STAGE 5: KHRONOS HIGHLIGHT COMPRESSION
    // ---------------------------------------------------------------------------------------------
    float3 pre_khronos_color = color;

    [branch]
    if (bEnableKhronosNeutral)
    {
        color /= max(whitePt, FLT_MIN);

        float targetPeak = (space <= 1) ? 1.0 : max(1.0, fDisplayPeakNits / whitePt);
        color = ApplyKhronosPBRNeutral(color, targetPeak, fCompressionStart, fDesaturationStrength);

        color *= whitePt;
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 6: MACLEOD-BOYNTON ISOLUMINANT PURITY
    // ---------------------------------------------------------------------------------------------
    color = ApplyMBPurity(color, fSaturation, space, lumaCoeffs, to_LMS, to_RGB, mb_white);

    // ---------------------------------------------------------------------------------------------
    // STAGE 6.5: ANALYTICAL MB GAMUT GUARD (V5.9.7)
    // ---------------------------------------------------------------------------------------------
    if (fGamutGuardKnee > NEUTRAL_EPS)
        color = ApplyGamutGuard(color, fGamutGuardKnee, lumaCoeffs, to_LMS, to_RGB, mb_white);

    // ---------------------------------------------------------------------------------------------
    // Safety: Pipeline NaN/Inf Catch
    // ---------------------------------------------------------------------------------------------
    if (any(IsNan3(color)) || any(IsInf3(color))) color = original_lin;

    // ---------------------------------------------------------------------------------------------
    // DEBUG VISUALIZATION
    // ---------------------------------------------------------------------------------------------
    [branch]
    if (iDebugMode != 0)
    {
        float3 debug_out = float3(0.0, 0.0, 0.0);

        if (iDebugMode == 1)
        {
            float luma = dot(color, lumaCoeffs);
            float stops = log2(max(abs(luma), FLT_MIN) / max(whitePt, FLT_MIN));
            debug_out = StopsToFalseColor(stops);
        }
        else if (iDebugMode == 2)
        {
            float luma = dot(color, lumaCoeffs);
            float nl = luma / max(whitePt, FLT_MIN);
            debug_out = GetZoneColor(GetZone(nl));
        }
        else if (iDebugMode == 3)
        {
            float k = ComputeBleachingK(pre_bleach_color, fBleaching, to_LMS);
            debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(k));
        }
        else if (iDebugMode == 4)
        {
            float3 lms = mul(to_LMS, color);
            float lm_sum = lms.r + lms.g;

            if (lm_sum > 0.0)
            {
                float3 mb = LMS_to_MB(lms);
                float2 chroma_offset = mb.xy - mb_white;
                float purity = SqrtIEEE(dot(chroma_offset, chroma_offset));
                float v = saturate(purity * 3.0);
                debug_out = float3(v, v * 0.7, v * 0.3);
            }
        }
        else if (iDebugMode == 5)
        {
            float3 lms = mul(to_LMS, color);
            float lm_sum = lms.r + lms.g;

            if (lm_sum > 0.0)
            {
                float3 mb = LMS_to_MB(lms);
                float2 chroma_offset = mb.xy - mb_white;
                float purity_sq = dot(chroma_offset, chroma_offset);

                if (purity_sq > 1e-12)
                {
                    float hue = atan2(chroma_offset.y, chroma_offset.x) / (2.0 * PI) + 0.5;
                    float brightness = saturate(SqrtIEEE(purity_sq) * 5.0);
                    debug_out = HueToRGB(saturate(hue)) * brightness;
                }
            }
        }
        else if (iDebugMode == 6)
        {
            float3 lms = mul(to_LMS, color);
            float max_lms = max(max(abs(lms.r), abs(lms.g)), abs(lms.b));
            if (max_lms > FLT_MIN)
                debug_out = abs(lms) / max_lms;
        }
        else if (iDebugMode == 7)
        {
            if (any(IsNan3(color)) || any(IsInf3(color)))
            {
                debug_out = float3(1.0, 1.0, 1.0);
            }
            else
            {
                float3 neg = float3(
                    color.r < 0.0 ? 1.0 : 0.0,
                    color.g < 0.0 ? 1.0 : 0.0,
                    color.b < 0.0 ? 1.0 : 0.0
                );
                float any_neg = neg.r + neg.g + neg.b;

                if (any_neg > 0.0)
                {
                    debug_out = float3(
                        max(neg.r, neg.b),
                        max(neg.g, neg.b),
                        max(neg.r, neg.g)
                    );
                }
                else
                {
                    debug_out = float3(0.0, 0.15, 0.0);
                }
            }
        }
        else if (iDebugMode == 8)
        {
            if (bEnableKhronosNeutral)
            {
                float3 normalized = pre_khronos_color / max(whitePt, FLT_MIN);
                float targetPeak = (space <= 1) ? 1.0 : max(1.0, fDisplayPeakNits / whitePt);
                float ratio = ComputeCompressionRatio(normalized, targetPeak, fCompressionStart);
                debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(ratio));
            }
            else
            {
                debug_out = float3(0.2, 0.2, 0.2);
            }
        }

        fragColor = float4(EncodeDebug(debug_out, space), src.a);
        return;
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 7: ENCODE & OUTPUT
    // ---------------------------------------------------------------------------------------------
    float3 encoded = EncodeFromLinear(color, space);

    [flatten]
    if (space <= 1) encoded = saturate(encoded);

    fragColor = float4(encoded, src.a);
}

// =================================================================================================
// 11. Technique Definition
// =================================================================================================

technique PhotorealHDR_Mastering_V597 <
    ui_label = "Photoreal HDR V5.9.7 (Director's Cut)";
    ui_tooltip = "Photorealistic grading for SDR and HDR.\n\n"
                 "Pipeline:\n"
                 "  1. Exposure (linear EV shift)\n"
                 "  2. LMS White Balance (exponential cone-domain)\n"
                 "  3. Subtractive Black Point (C1 smooth floor, crush-free)\n"
                 "  4. Filmic Contrast + Tonal EQ (stop-domain, C1 rational recovery)\n"
                 "  5. Biological Highlight Bleaching (Troland photopigment depletion)\n"
                 "  6. Khronos PBR Neutral Highlight Compression\n"
                 "  7. MacLeod-Boynton Isoluminant Purity\n"
                 "  8. Analytical MB Gamut Guard (hue-preserving, zero loops)\n\n"
                 "Design: Precision and quality over performance.\n"
                 "  - True IEEE 754 math (no SFU approximations)\n"
                 "  - IEC/SMPTE exact standard constants\n"
                 "  - Physiological MacLeod-Boynton chromaticity\n"
                 "  - Vivid-color protection with gamut awareness\n"
                 "  - Khronos hue invariance (proven, not approximated)\n"
                 "  - Troland biological bleaching (intensity-driven, hue-preserving)\n"
                 "  - Analytical gamut guard (exact ray-boundary intersection)\n\n"
                 "Debug: 9 visualization modes (luminance, zones, bleaching, purity,\n"
                 "        hue, LMS, negative/WCG, compression).\n\n"
                 "Companion shader: Bilateral Contrast v8.4.5+";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_PhotorealHDR;
    }
}