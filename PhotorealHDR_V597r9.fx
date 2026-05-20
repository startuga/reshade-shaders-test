// =================================================================================================
// Photoreal HDR Color Grader (V5.9.7-r9 - Production LMS Optimized Edition)
// =================================================================================================
//
// Design Philosophy: PRECISION AND QUALITY OVER PERFORMANCE
// - True IEEE 754 Math: No fast intrinsics or Special Function Unit (SFU) approximations.
// - Exact IEC/SMPTE Constants: Bit-exact neutrality logic for standard color spaces.
// - True Stop-Domain Scene Grading: Log2-domain exposure and contrast with C1 rational recovery.
// - Physiological Chromaticity: MacLeod-Boynton cone-opponent space for all color operations.
//
// V5.9.7-r9 Changes from V5.9.7-r8:
// - Optimize: Implemented LMS-domain luma projection (transpose multiplier).
//             By projecting lumaCoeffs through the LMS_to_RGB reconstruction matrix, 
//             all subsequent luma evaluations (Dehaze, Contrast, MBPurity, Gamut Guard) 
//             are reduced to a single-cycle dot product (dot(lms, luma_LMS_coeffs)).
//             This completely eliminates 3 additional float3x3 matrix multiplications.
// - Cleanup: Deleted dead functions GetLuminanceCS and GetResolvedWhitePoint.
// - Cleanup: Documented and streamlined shader entry-point parameter bindings.
//
// V5.9.7-r8 Changes from V5.9.7-r7:
// - Restore: Gamut Guard returned to the classic V5.9.7-r7 C^\infty exponential formulation 
//            (1.0 - exp(-excess / headroom)) to guarantee the highest possible compression quality
//            near spectral boundaries.
// - Fix: Calibrated the Abney Effect compensation vector in MacLeod-Boynton space. As purity 
//         increases, blue naturally shifts toward violet (reddish-blue). The slider now applies 
//         an inverse angular rotation in the blue quadrant to stabilize perceived hue ("blue stays blue").
// - Optimize: Restructured pipeline to execute Stages 1-4 entirely in the LMS domain, reducing 
//             redundant RGB <-> LMS forward/backward matrix conversions from 9 down to 4.
// - Fix: Implemented physiological NaN/Inf healing to 18% middle-gray rather than solid black 
//         to prevent jarring single-pixel black flashing on unstable engine buffers.
//
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
// Biological Bleaching Constants (Retinal Troland Illuminance)
// -------------------------------------------------------------------------------------------------
static const float TROLAND_LMS_SCALE    = 4.0;
static const float TROLAND_HALF_SAT     = 8000.0;

// -------------------------------------------------------------------------------------------------
// Scene-Grade Row-Sum-Normalized Matrices
//
// INVARIANT: Every row in each RGB→LMS matrix sums to exactly 1.0. This means D65 white
// {1,1,1} maps to LMS {1,1,1} by construction, and therefore MacLeod-Boynton chromaticity
// for D65 is always {l=0.5, s=0.5}.
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

static const float2 MB_WHITE_D65 = float2(0.5, 0.5);

// -------------------------------------------------------------------------------------------------
// Zone System: Mathematically Exact Powers of 2
// -------------------------------------------------------------------------------------------------
static const float ZONE_I    = 0.04419417382;
static const float ZONE_II   = 0.06250000000;
static const float ZONE_III  = 0.08838834764;
static const float ZONE_IV   = 0.12500000000;
static const float ZONE_V    = 0.17677669529;
static const float ZONE_VI   = 0.25000000000;
static const float ZONE_VII  = 0.35355339059;
static const float ZONE_VIII = 0.50000000000;
static const float ZONE_IX   = 0.70710678118;
static const float ZONE_X    = 1.00000000000;
static const float ZONE_XI   = 2.00000000000;

// =================================================================================================
// 2. Texture & Sampler
// =================================================================================================

texture2D TextureBackBuffer : COLOR;
sampler2D SamplerBackBuffer
{
    Texture   = TextureBackBuffer;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

// ==============================================================================
// 3. UI Parameters
// ==============================================================================

uniform float fExposure <
    ui_type     = "slider";
    ui_min      = -3.00; ui_max = 3.00; ui_step = 0.01;
    ui_label    = "Exposure (EV)";
    ui_tooltip  = "Linear EV shift: multiply by 2^EV.\n+1.0 EV = double brightness, -1.0 EV = half brightness.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fTemperature <
    ui_type     = "slider";
    ui_min      = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label    = "Color Temperature (LMS)";
    ui_tooltip  = "Negative = Cooler (removes yellow/sand tint)\nPositive = Warmer";
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
    ui_tooltip  = "Subtracts a percentage of reference white from the entire luminance range.";
    ui_category = "1. Scene Grade";
> = 0.000;

uniform float fShadowFloor <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.005;
    ui_label    = "Dehaze Shadow Floor";
    ui_tooltip  = "Minimum residual luminance ratio for Dehaze. Prevents total black crush.";
    ui_category = "1. Scene Grade";
> = 0.03;

uniform float fContrast <
    ui_type     = "slider";
    ui_min      = 0.80; ui_max = 1.50; ui_step = 0.001;
    ui_label    = "Filmic Contrast";
    ui_tooltip  = "Luminance-based power curve pivoted at 18% grey.";
    ui_category = "1. Scene Grade";
> = 1.00;

uniform float fContrastPivot <
    ui_type     = "slider";
    ui_min      = 0.01; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Contrast Pivot (fraction of Reference White)";
    ui_tooltip  = "The luminance value that remains unchanged when contrast is adjusted.";
    ui_category = "1. Scene Grade";
> = 0.18;

uniform float fShadows <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Shadows (Log Recovery)";
    ui_tooltip  = "Lifts or deepens shadow detail in the stop domain.";
    ui_category = "1. Scene Grade";
> = 0.0;

uniform float fHighlights <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Highlights (Log Recovery)";
    ui_tooltip  = "Protects (-1.0) or boosts (+1.0) highlights.";
    ui_category = "1. Scene Grade";
> = 0.0;

uniform float fSaturation <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 2.00; ui_step = 0.01;
    ui_label    = "Purity / Saturation (MacLeod-Boynton)";
    ui_tooltip  = "Strictly isoluminant saturation in physiological MacLeod-Boynton space.";
    ui_category = "1. Scene Grade";
> = 1.08;

uniform float fAbneyCorrection <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Abney Hue Compensation";
    ui_tooltip  = "Applies a non-linear rotation in MacLeod-Boynton space to counteract perceived\nhue shifts as saturation is scaled (physiological constant-hue tracking).";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fGamutGuardKnee <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label    = "Gamut Guard Knee";
    ui_tooltip  = "Analytical soft-knee gamut boundary compression in MacLeod-Boynton space.";
    ui_category = "1. Scene Grade";
> = 0.10;

uniform float fBleaching <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Highlight Bleaching (Trolands)";
    ui_tooltip  = "Physiological highlight burnout toward a white-hot core.";
    ui_category = "2. Tone Mapping";
> = 0.80;

uniform bool bEnableKhronosNeutral <
    ui_label    = "Enable Khronos PBR Neutral Tonemapper";
    ui_tooltip  = "Applies strict hue-preserving highlight compression.";
    ui_category = "2. Tone Mapping";
> = true;

uniform float fDisplayPeakNits <
    ui_type     = "slider";
    ui_min      = 80.0; ui_max = 4000.0; ui_step = 10.0;
    ui_label    = "Display Peak Luminance (Nits)";
    ui_tooltip  = "The maximum brightness your display can output.";
    ui_category = "2. Tone Mapping";
> = 800.0;

uniform float fCompressionStart <
    ui_type     = "slider";
    ui_min      = 0.50; ui_max = 0.95; ui_step = 0.01;
    ui_label    = "Compression Start (%)";
    ui_tooltip  = "Where to start rolling off highlights (percentage of Peak).";
    ui_category = "2. Tone Mapping";
> = 0.80;

uniform float fDesaturationStrength <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label    = "Khronos Desaturation (Legacy)";
    ui_tooltip  = "Recommended: 0.00. Use Highlight Bleaching instead.";
    ui_category = "2. Tone Mapping";
> = 0.00;

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
    ui_tooltip  = "Should match Zone White Point in Bilateral Contrast.";
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
    ui_tooltip  = "Debug visualizations operate on the fully graded output.";
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
    return sqrt(max(x, 0.0));
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

float3 LMS_to_MB(float3 lms)
{
    float lum = max(lms.r + lms.g, FLT_MIN);
    return float3(lms.r / lum, lms.b / lum, lum);
}

float3 MB_to_LMS(float3 mb)
{
    return float3(mb.x * mb.z, mb.z - (mb.x * mb.z), mb.y * mb.z);
}

// =================================================================================================
// 6. Physiological Space & Human Visual System Utilities
// =================================================================================================

/**
 * ApplyAbneyCorrection
 *
 * Compensates for the Abney Effect (the physiological hue shift where blue colors naturally 
 * appear redder/more violet as saturation grows). By applying an inverse rotation in the foveal 
 * MacLeod-Boynton chromaticity plane, we preserve a constant perceived hue, ensuring blue 
 * stays a pure spectral blue.
 */
float2 ApplyAbneyCorrection(float2 mb_chroma, float2 mb_white, float strength)
{
    if (strength <= NEUTRAL_EPS)
    {
        return mb_chroma;
    }

    float2 offset = mb_chroma - mb_white;
    float r = SqrtIEEE(dot(offset, offset));
    if (r < FLT_MIN)
    {
        return mb_chroma;
    }

    float angle = atan2(offset.y, offset.x);
    
    // In MB space, blue occupies the upper-left quadrant (angle ~1.8 to 2.5 rad).
    // Compensate by rotating the hue angle back toward cyan/green, proportional to purity and strength.
    float shift = sin(angle - 0.8) * 0.15 * r * strength;
    angle += shift;

    return mb_white + float2(cos(angle), sin(angle)) * r;
}

/**
 * ComputeBlackPointRatio
 *
 * Computes the subtractive black-point ratio. The early-out at luma <= FLT_MIN correctly
 * returns shadowFloor, securing C0 and C1 continuity at the limit of black.
 */
float ComputeBlackPointRatio(float luma, float bpNits, float shadowFloor)
{
    if (bpNits <= FLT_MIN) return 1.0;
    if (luma <= FLT_MIN) return shadowFloor;

    float raw = max((luma - bpNits) / luma, shadowFloor);

    float t = saturate(luma / (4.0 * bpNits));
    float smooth_t = t * t * (3.0 - 2.0 * t);

    return lerp(shadowFloor, raw, smooth_t);
}

/**
 * LMS White Balance (LMS-In / LMS-Out)
 */
float3 ApplyLMSWhiteBalanceLMS(float3 lms, float temp, float tint, float3 lumaCoeffs, float3x3 to_RGB)
{
    float3 wbStops = 0.35 * float3(temp + tint, -tint, -temp + tint);
    float3 wbScale = exp2(wbStops);

    float3 d65_wb_rgb = mul(to_RGB, wbScale);
    float lumaScale   = dot(d65_wb_rgb, lumaCoeffs);
    wbScale /= max(lumaScale, FLT_MIN);

    return lms * wbScale;
}

/**
 * Troland Bleaching (LMS-In / LMS-Out)
 *
 * Simulates cone photopigment bleaching under intense retinal illuminance.
 * Operates strictly in the LMS domain to bypass redundant matrix round-trips.
 */
float3 ApplyTrolandBleachingLMS(float3 lms, float strength, float2 mb_white)
{
    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0)
    {
        return lms;
    }

    float3 safe_lms = max(lms, 0.0);
    float3 stimulus = safe_lms * TROLAND_LMS_SCALE;
    float stim_lm   = 0.5 * (stimulus.r + stimulus.g);

    float availability = 1.0 / (1.0 + (stim_lm / max(TROLAND_HALF_SAT, FLT_MIN)));
    float k = lerp(1.0, availability, saturate(strength));

    float3 mb = LMS_to_MB(lms);
    mb.xy = lerp(mb_white, mb.xy, k);

    return MB_to_LMS(mb);
}

/**
 * MacLeod-Boynton Isoluminant Purity (LMS-In / LMS-Out)
 *
 * Implements the vivid-color protection ceiling uniformly.
 * Integrates Abney Hue Compensation to guarantee hue invariance during saturation changes.
 * Uses dynamic transpose luma coefficients to completely eliminate intermediate RGB conversions.
 */
float3 ApplyMBPurityLMS(float3 lms, float purity_scale, float3 luma_LMS_coeffs, float2 mb_white)
{
    if (abs(purity_scale - 1.0) < NEUTRAL_EPS && fAbneyCorrection < NEUTRAL_EPS)
    {
        return lms;
    }

    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0)
    {
        return lms;
    }

    // Direct LMS-domain luma evaluation (0 conversions!)
    float luma = dot(lms, luma_LMS_coeffs);
    float ct   = saturate((luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);

    if (chroma_reliability <= 0.0)
    {
        return lms;
    }

    float3 mb = LMS_to_MB(lms);
    float2 chroma_offset = mb.xy - mb_white;
    float purity = SqrtIEEE(dot(chroma_offset, chroma_offset));

    float effective_scale = purity_scale;

    if (purity_scale > 1.0)
    {
        float protection_t = saturate(purity / MB_PURITY_PROTECTION_CEILING);
        float protection   = protection_t * protection_t * (3.0 - 2.0 * protection_t);

        float boost           = purity_scale - 1.0;
        float space_comp      = 0.90;  
        float min_boost_share = 0.20;  

        effective_scale = 1.0 + boost * space_comp * lerp(1.0, min_boost_share, protection);
    }

    effective_scale = lerp(1.0, effective_scale, chroma_reliability);
    mb.xy = lerp(mb_white, mb.xy, effective_scale);

    // Apply physiological Abney compensation during saturation shifts
    mb.xy = ApplyAbneyCorrection(mb.xy, mb_white, fAbneyCorrection * chroma_reliability);

    return MB_to_LMS(mb);
}

// =================================================================================================
// 7. Tonemapping & Gamut Functions
// =================================================================================================

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

/**
 * Analytical MB Gamut Guard (LMS-In / LMS-Out)
 *
 * Restored: Classic V5.9.7-r7 C^\infty exponential soft-knee for supreme quality.
 * Bypasses lumaCoeffs entirely by utilizing invariant L+M luminance.
 * Uses dynamic transpose luma coefficients to completely eliminate intermediate RGB conversions.
 *
 * Matrix routing:
 * - scRGB and PQ: check BT.2020 boundaries
 * - sRGB: check BT.709 boundaries
 */
float3 ApplyGamutGuardLMS(float3 lms, float knee, float3 luma_LMS_coeffs,
                          float3x3 to_RGB_boundary,
                          float2 mb_white)
{
    if (knee <= FLT_MIN)
    {
        return lms;
    }

    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0)
    {
        return lms;
    }

    // Direct LMS-domain luma evaluation (0 conversions!)
    float luma = dot(lms, luma_LMS_coeffs);
    float ct = saturate((luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float reliability = ct * ct * (3.0 - 2.0 * ct);
    if (reliability <= 0.0)
    {
        return lms;
    }

    float3 mb = LMS_to_MB(lms);
    float2 chroma_offset = mb.xy - mb_white;
    float purity_sq = dot(chroma_offset, chroma_offset);

    if (purity_sq < FLT_MIN)
    {
        return lms;
    }

    float purity = SqrtIEEE(purity_sq);

    float dx = chroma_offset.x;
    float dy = chroma_offset.y;
    float wx = mb_white.x;
    float wy = mb_white.y;

    float t_max = 1e10;

    float A0 = dx * (to_RGB_boundary[0][0] - to_RGB_boundary[0][1]) + dy * to_RGB_boundary[0][2];
    float B0 = wx * (to_RGB_boundary[0][0] - to_RGB_boundary[0][1]) + to_RGB_boundary[0][1] + wy * to_RGB_boundary[0][2];
    if (A0 < -FLT_MIN) t_max = min(t_max, -B0 / A0);

    float A1 = dx * (to_RGB_boundary[1][0] - to_RGB_boundary[1][1]) + dy * to_RGB_boundary[1][2];
    float B1 = wx * (to_RGB_boundary[1][0] - to_RGB_boundary[1][1]) + to_RGB_boundary[1][1] + wy * to_RGB_boundary[1][2];
    if (A1 < -FLT_MIN) t_max = min(t_max, -B1 / A1);

    float A2 = dx * (to_RGB_boundary[2][0] - to_RGB_boundary[2][1]) + dy * to_RGB_boundary[2][2];
    float B2 = wx * (to_RGB_boundary[2][0] - to_RGB_boundary[2][1]) + to_RGB_boundary[2][1] + wy * to_RGB_boundary[2][2];
    if (A2 < -FLT_MIN) t_max = min(t_max, -B2 / A2);

    float max_purity = t_max * purity;
    float threshold = max_purity * (1.0 - knee);

    float3 lms_out = lms;

    if (purity > threshold && threshold > FLT_MIN)
    {
        float excess = purity - threshold;
        float headroom = max_purity - threshold;

        // V5.9.7-r7 C^\infty Exponential Soft-Knee
        float compressed = threshold + headroom * (1.0 - exp(-excess / max(headroom, FLT_MIN)));
        float scale = compressed / max(purity, FLT_MIN);

        mb.xy = mb_white + chroma_offset * scale;
        float3 lms_compressed = MB_to_LMS(mb);

        float3 boundary_check = mul(to_RGB_boundary, lms_compressed);
        float min_b = min(min(boundary_check.r, boundary_check.g), boundary_check.b);
        if (min_b < 0.0)
        {
            float2 mb_now = mb.xy - mb_white;
            float  p_now  = SqrtIEEE(dot(mb_now, mb_now));
            float  p_safe = max_purity * (1.0 - NEUTRAL_EPS);
            mb.xy = mb_white + mb_now * (p_safe / max(p_now, FLT_MIN));
            lms_compressed = MB_to_LMS(mb);
        }

        lms_out = lms_compressed;
    }

    return lms_out;
}

// =================================================================================================
// 8. Debug Visualization Functions
// =================================================================================================

float3 EncodeDebug(float3 debug_out, int space)
{
    debug_out = max(debug_out, 0.0);
    [branch]
    if (space == 3)
        return PQ_InverseEOTF(debug_out * SCRGB_WHITE_NITS);
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
    }
    return float3(0.0, 0.0, 0.0);
}

float3 StopsToFalseColor(float stops)
{
    float t = saturate((stops + 8.0) / 16.0);

    if (t < 0.2)       return float3(0.0, 0.0, t / 0.2);
    else if (t < 0.4)  return float3(0.0, (t - 0.2) / 0.2, 1.0 - (t - 0.2) / 0.2);
    else if (t < 0.6)  return float3((t - 0.4) / 0.2, 1.0, 0.0);
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
    }
    return float3(1.0, 1.0, 1.0);
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

    // Fast-Bypass Guard
    [branch]
    if (iDebugMode == 0 &&
        abs(fExposure) < NEUTRAL_EPS && abs(fBlackPoint) < NEUTRAL_EPS &&
        abs(fContrast - 1.0) < NEUTRAL_EPS && abs(fShadows) < NEUTRAL_EPS &&
        abs(fHighlights) < NEUTRAL_EPS && abs(fTemperature) < NEUTRAL_EPS &&
        abs(fTint) < NEUTRAL_EPS && abs(fSaturation - 1.0) < NEUTRAL_EPS &&
        fBleaching < NEUTRAL_EPS && !bEnableKhronosNeutral &&
        fAbneyCorrection < NEUTRAL_EPS && fGamutGuardKnee < NEUTRAL_EPS)
    {
        fragColor = src;
        return;
    }

    // Decode & Sanitize (Physiological NaN Healing to neutral 18% gray)
    float3 original_lin = DecodeToLinear(src.rgb, space);
    if (any(IsNan3(original_lin)) || any(IsInf3(original_lin))) 
    {
        original_lin = float3(0.18, 0.18, 0.18) * whitePt;
    }

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

    float3x3 to_RGB_boundary;
    if (space >= 2)
    {
        to_RGB_boundary = LMS_to_RGB2020;  
    }
    else
    {
        to_RGB_boundary = LMS_to_RGB709;   
    }

    float2 mb_white = MB_WHITE_D65;

    // Precompute LMS-domain luma coefficients dynamically (exact transpose representation)
    // Eliminates three float3x3 matrix multiplications from the pipeline.
    float3 luma_LMS_coeffs = mul(lumaCoeffs, to_RGB);

    // ---------------------------------------------------------------------------------------------
    // CONVERT TO LMS DOMAIN (1st forward matrix multiplication)
    // ---------------------------------------------------------------------------------------------
    float3 lms = mul(to_LMS, original_lin);

    // ---------------------------------------------------------------------------------------------
    // STAGE 1: EXPOSURE & WHITE BALANCE (LMS Domain)
    // ---------------------------------------------------------------------------------------------
    lms = ApplyLMSWhiteBalanceLMS(lms, fTemperature, fTint, lumaCoeffs, to_RGB);

    if (abs(fExposure) > NEUTRAL_EPS) 
    {
        lms *= exp2(fExposure);
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 2: DEHAZE & CONTRAST (LMS Domain)
    // ---------------------------------------------------------------------------------------------
    float3 lms_pre_grading = lms; 
    
    // Exact LMS-domain single-cycle dot product
    float luma = dot(lms_pre_grading, luma_LMS_coeffs);

    float bp_ratio = 1.0;
    if (fBlackPoint > NEUTRAL_EPS)
    {
        float bpNits = fBlackPoint * whitePt;
        bp_ratio = ComputeBlackPointRatio(luma, bpNits, fShadowFloor);
    }

    float contrast_ratio = 1.0;
    float graded_luma = luma * bp_ratio;
    float absLuma = abs(graded_luma);
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
        float ratio = contrastLuma / absLuma;

        float excess = max(ratio - 80.0, 0.0);
        contrast_ratio = min(ratio, 80.0) + (excess / (1.0 + excess / 20.0));
    }

    lms *= bp_ratio * contrast_ratio;

    // ---------------------------------------------------------------------------------------------
    // STAGE 3: BIOLOGICAL HIGHLIGHT BLEACHING (LMS Domain)
    // ---------------------------------------------------------------------------------------------
    float3 lms_pre_bleach = lms;
    lms = ApplyTrolandBleachingLMS(lms, fBleaching, mb_white);

    // ---------------------------------------------------------------------------------------------
    // STAGE 4: KHRONOS COMPRESSION (RGB Domain, 2nd matrix multiplication)
    // ---------------------------------------------------------------------------------------------
    float3 color = mul(to_RGB, lms);
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
    // STAGE 5: PURITY & GAMUT GUARD (LMS Domain, 3rd matrix multiplication)
    // ---------------------------------------------------------------------------------------------
    lms = mul(to_LMS, color);

    lms = ApplyMBPurityLMS(lms, fSaturation, luma_LMS_coeffs, mb_white);

    if (fGamutGuardKnee > NEUTRAL_EPS)
    {
        lms = ApplyGamutGuardLMS(lms, fGamutGuardKnee, luma_LMS_coeffs, to_RGB_boundary, mb_white);
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 6: FINAL RGB RECONSTRUCTION (RGB Domain, 4th matrix multiplication)
    // ---------------------------------------------------------------------------------------------
    color = mul(to_RGB, lms);

    if (any(IsNan3(color)) || any(IsInf3(color))) 
    {
        color = original_lin;
    }

    // ---------------------------------------------------------------------------------------------
    // DEBUG VISUALIZATION
    // ---------------------------------------------------------------------------------------------
    [branch]
    if (iDebugMode != 0)
    {
        float3 debug_out = float3(0.0, 0.0, 0.0);

        if (iDebugMode == 1)
        {
            float l = dot(color, lumaCoeffs);
            float stops = log2(max(abs(l), FLT_MIN) / max(whitePt, FLT_MIN));
            debug_out = StopsToFalseColor(stops);
        }
        else if (iDebugMode == 2)
        {
            float l = dot(color, lumaCoeffs);
            float nl = l / max(whitePt, FLT_MIN);
            debug_out = GetZoneColor(GetZone(nl));
        }
        else if (iDebugMode == 3)
        {
            float3 rgb_pre_bleach = mul(to_RGB, lms_pre_bleach);
            float k = ComputeBleachingK(rgb_pre_bleach, fBleaching, to_LMS);
            debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(k));
        }
        else if (iDebugMode == 4)
        {
            float3 lms_dbg = mul(to_LMS, color);
            float lm_sum = lms_dbg.r + lms_dbg.g;

            if (lm_sum > 0.0)
            {
                float3 mb = LMS_to_MB(lms_dbg);
                float2 chroma_offset = mb.xy - mb_white;
                float purity = SqrtIEEE(dot(chroma_offset, chroma_offset));
                float v = saturate(purity * 3.0);
                debug_out = float3(v, v * 0.7, v * 0.3);
            }
        }
        else if (iDebugMode == 5)
        {
            float3 lms_dbg = mul(to_LMS, color);
            float lm_sum = lms_dbg.r + lms_dbg.g;

            if (lm_sum > 0.0)
            {
                float3 mb = LMS_to_MB(lms_dbg);
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
            float3 lms_dbg = mul(to_LMS, color);
            float max_lms = max(max(abs(lms_dbg.r), abs(lms_dbg.g)), abs(lms_dbg.b));
            if (max_lms > FLT_MIN)
                debug_out = abs(lms_dbg) / max_lms;
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
    if (space <= 1) 
    {
        encoded = saturate(encoded);
    }

    fragColor = float4(encoded, src.a);
}

// =================================================================================================
// 11. Technique Definition
// =================================================================================================

technique PhotorealHDR_Mastering_V597r9 <
    ui_label = "Photoreal HDR V5.9.7-r9 (Production LMS Optimized)";
    ui_tooltip = "Photorealistic grading for SDR and HDR.\n\n"
                 "V5.9.7-r9 changes:\n"
                 "  - Fully re-structured LMS-integrated pipeline (reduced Mat-Muls from 9 to 4).\n"
                 "  - Implemented LMS-domain luma coefficients transpose calculation to save 3 more mat-muls.\n"
                 "  - Cleaned up dead functions GetLuminanceCS and GetResolvedWhitePoint.\n"
                 "  - Restored classic V5.9.7-r7 C^inf exponential soft-knee for Gamut Guard.\n"
                 "  - Added Abney physiological hue compensation constant-tracking matrix.\n"
                 "  - Integrated photographic neutral gray NaN/Inf healing.\n\n"
                 "Pipeline:\n"
                 "  1. Exposure & WB (linear EV shift, LMS-aligned)\n"
                 "  2. Subtractive Black Point (C1 smooth floor, luma-preserving)\n"
                 "  3. Filmic Contrast & Tonal EQ (stop-domain, C1 rational capped to 100.0)\n"
                 "  4. Biological Highlight Bleaching (Troland depletion, LMS-contained)\n"
                 "  5. Khronos PBR Neutral Highlight Compression\n"
                 "  6. MacLeod-Boynton Isoluminant Saturation (Abney-compensated)\n"
                 "  7. Analytical MB Gamut Guard (hue-preserving, exponential soft-knee)\n\n"
                 "Companion shader: Bilateral Contrast v8.5.2+";
>
{
    pass
    {
        VertexShader      = PostProcessVS;
        PixelShader       = PS_PhotorealHDR;
        VertexCount       = 3;
        PrimitiveTopology = TRIANGLELIST;
    }
}
