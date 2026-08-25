// =================================================================================================
// Photoreal HDR Color Grader (V6.1.6 - Reliability & Normalization Consistency Edition)
// =================================================================================================
//
// Design Philosophy: EXACT MATHEMATICAL RIGOR (OFFLINE / REFERENCE GRADE)
// - True IEEE 754 Math: Guarded linear paths, NaN healing, and unscaled scale-invariant solvers.
// - Exact IEC/SMPTE Constants: Analytically matched thresholds for bit-exact C0 continuity.
// - True Stop-Domain Scene Grading: Log2-domain exposure and contrast with C1 rational recovery.
// - Exact Ray-Tracing: 24-step unrolled binary search solver for exact boundaries (t_high = 8.0).
// - Physiological Chromaticity: MacLeod-Boynton cone-opponent space for all color operations.
// - Static D65 Anchor: Permanently anchored to (0.5, 0.5) to eliminate white-point hue drift.
// - Gamut-Relative Vibrance: Normalized pastel saturation scaling across all hues uniformly.
//
// V6.1.6 Audited Changes:
//   - FIX (Dead Gating): Chroma reliability ramp converted from absolute nits to a white-point-
//     relative domain (0.05% .. 0.2% of diffuse white). Shadow desaturation protection and
//     vibrance/Abney gating are now active in real image content instead of only FP noise.
//   - FIX (Identity Defaults): Gamut Guard Knee now defaults to 0.00. All-neutral sliders are a
//     true bit-transparent bypass (no background solver, no edge compression).
//   - FIX (Unclamped Normalization): The gamut boundary solve used as the vibrance/diminishing-
//     returns reference is now UNCONDITIONAL. Identical slider settings produce identical scaling
//     whether the guard is clamped or unclamped.
//   - FIX (Grayscale Precedence): Master Saturation = 0.00 forces exact R=G=B collapse and
//     overrides Vibrance. Vibrance gain is additionally gated by saturate(master saturation).
//   - FIX (Extended-Range Passthrough): Dehaze/black-point no longer crushes negative-luma
//     (wide-gamut scRGB) pixels toward black.
//   - CLEANUP: Khronos PBR Neutral math centralized (single source of truth for TM + debug).
//   - CLEANUP: Dead parameter removed from NRG-TM; scRGB boundary policy documented inline.
//   - DOCS: HLG OOTF omission and P3 matrix provenance explicitly documented.
//
// Carried from V6.1.5:
//   - Static D65 anchor (0.5, 0.5); zero hue bending across luminance gradients.
//   - Gamut-Relative Vibrance guarantees equal pastel response across hues.
//   - Strictly radial purity scaling: zero hue distortion across the full wheel.
//   - Selectable gamut targets (Auto, Rec.709, DCI-P3, Rec.2020, Unclamped).
//   - 3D Melanin-Hemoglobin skin locus protecting Fitzpatrick I-VI.
//   - Closed-form constant-luminance z-recovery (Delta-Y lock, ~1 ulp residual).
//
// References:
// - https://github.com/crosire/reshade-shaders/blob/slim/Shaders/ReShade.fxh
// - https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md
// - https://onlinelibrary.wiley.com/doi/10.1111/cgf.70136
// =================================================================================================

#include "ReShade.fxh"

// =================================================================================================
// 1. Constants & Definitions
// =================================================================================================

#if defined(__RESHADE__) && __RESHADE__ < 40800
    #error "Photoreal HDR requires ReShade 4.8.0 or newer."
#endif

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

static const float FLT_MIN              = 1.175494351e-38;
static const float SCRGB_WHITE_NITS     = 80.0;
static const float NEUTRAL_EPS          = 1e-6;
static const float PI                   = 3.14159265358979323846;

// -------------------------------------------------------------------------------------------------
// sRGB Constants (IEC 61966-2-1:1999) - Analytically Solved C0-Intersection Roots
// -------------------------------------------------------------------------------------------------
static const float SRGB_THRESHOLD_EOTF  = 0.040448236277123205;
static const float SRGB_THRESHOLD_OETF  = 0.003130668442501796;
static const float SRGB_GAMMA           = 2.4;
static const float SRGB_INV_GAMMA       = 0.41666666666666667; // 1 / 2.4 = 5 / 12

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
// V6.1.6: Chroma reliability thresholds are FRACTIONS OF DIFFUSE WHITE (not absolute nits).
// Below ~0.05% of reference white, cone-opponent signals drown in quantization noise; above
// ~0.2% they are fully trustworthy. This makes the gate active on real image content.
static const float CHROMA_RELIABILITY_START     = 5e-4;   // 0.05% of diffuse white
static const float CHROMA_STABILITY_THRESH      = 2e-3;   // 0.20% of diffuse white
static const float INV_CHROMA_RELIABILITY_SPAN  = 1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);

static const float3 Luma709             = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020            = float3(0.2627, 0.6780, 0.0593);

// -------------------------------------------------------------------------------------------------
// Biological Bleaching Constants (Retinal Troland Illuminance)
// -------------------------------------------------------------------------------------------------
static const float TROLAND_LMS_SCALE    = 4.0;
static const float TROLAND_HALF_SAT     = 8000.0;

// -------------------------------------------------------------------------------------------------
// Row-Sum-Normalized Color Matrices (D65 maps (1,1,1) in RGB to (1,1,1) in LMS)
// -------------------------------------------------------------------------------------------------
// Ottosson Hunt-Pointer-Estevez(D65) M1 for Rec.709. Row sums = 1 (gray-preserving).
static const float3x3 RGB709_to_LMS = float3x3(
    0.4122214708,  0.5363325363,  0.0514459929,
    0.2119034982,  0.6806995451,  0.1073969566,
    0.0883024619,  0.2817188376,  0.6299787005
);

// Analytic inverse of RGB709_to_LMS (verified: M_inv * M = I to ~1e-10; row sums = 1).
static const float3x3 LMS_to_RGB709 = float3x3(
     4.0767416621, -3.3077115913,  0.2309699292,
    -1.2684380046,  2.6097574011, -0.3413193965,
    -0.0041960863, -0.7034186147,  1.7076147010
);

// DCI-P3-D65 primaries mapped through the same HPE(D65) basis. Row sums = 1 (white-safe).
// NOTE: Validate round-trip identities (M_to_P3 * P3_to_LMS = I) on the CPU side before relying
// on it in mastering contexts; it is the least-exercised matrix in this file.
static const float3x3 LMS_to_P3D65 = float3x3(
     3.12776899, -2.25713580,  0.12936681,
    -1.09100905,  2.41333176, -0.32232271,
    -0.02601081, -0.50804133,  1.53405214
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
static const float ZONE_V    = 0.17677669529; // Grey point (17.68%)
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
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

// =================================================================================================
// 3. UI Parameters
// =================================================================================================

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
    ui_tooltip  = "Subtracts a percentage of reference white from the entire luminance range.\nNegative-luma (extended-range scRGB WCG) pixels bypass dehaze untouched.";
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
> = 0.17677669529;

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
    ui_tooltip  = "Strictly isoluminant saturation in physiological MacLeod-Boynton space.\n"
                  "Set to 1.0 for neutral pass.\n"
                  "Set to 0.0 for exact R=G=B monochrome - this OVERRIDES Vibrance.";
    ui_category = "1. Scene Grade";
> = 1.00;

uniform float fVibrance <
    ui_type     = "slider";
    ui_min      = -1.00; ui_max = 2.00; ui_step = 0.01;
    ui_label    = "Smart Saturation (Vibrance)";
    ui_tooltip  = "Intelligently boosts muted pastels and sky/blue tones uniformly using Gamut-Relative Weber-Fechner curves.\n"
                  "Protects saturated colors from clipping.\n"
                  "Scaling reference is identical in clamped and unclamped gamut modes.\n"
                  "Has no effect when master Saturation is 0.0.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fSkinProtection <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Skin Tone Protection";
    ui_tooltip  = "Protects human skin tones (Fitzpatrick I-VI) from smart saturation and master saturation shifts\nusing a 3D volumetric locus in MacLeod-Boynton space.\n1.0 = Full Protection, 0.0 = Off.";
    ui_category = "1. Scene Grade";
> = 0.85;

uniform float fAbneyCorrection <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Abney Hue Compensation";
    ui_tooltip  = "Applies a non-linear rotation in MacLeod-Boynton space to counteract perceived\nhue shifts as saturation is scaled.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform int iGamutTarget <
    ui_type     = "combo";
    ui_label    = "Gamut Guard Target Limit";
    ui_items    = "Auto (Container Gamut)\0Rec. 709 (SDR Standard)\0DCI-P3 (Cinema)\0Rec. 2020 (UHD Display)\0Bypass / Unclamped (Infinite Gamut)\0";
    ui_tooltip  = "Selects the physical gamut boundary to compress/clamp against.\n"
                  "- Auto: Matches container (Rec.709 in SDR, Rec.2020 in HDR).\n"
                  "- scRGB note: 'Auto' guards against the Rec.2020 volume because scRGB expresses\n"
                  "  wide gamut via negative/extended 709-channel excursions; the guard bounds\n"
                  "  chromaticity while the linear container carries the signal losslessly.\n"
                  "- Bypass / Unclamped: Disables boundary clamping completely. Saturation/vibrance\n"
                  "  scaling references remain solved identically (consistency guarantee).";
    ui_category = "1. Scene Grade";
> = 0;

// V6.1.6: Default 0.00 so that factory-reset settings are a TRUE bypass (no solver, no knee).
uniform float fGamutGuardKnee <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label    = "Gamut Guard Knee";
    ui_tooltip  = "Analytical soft-knee gamut boundary compression in MacLeod-Boynton space.\n"
                  "Set to 0.0 for pure hard clamp.\n"
                  "Default 0.0 keeps all-neutral settings a bit-transparent bypass.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fBleaching <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Highlight Bleaching (Trolands)";
    ui_tooltip  = "Physiological highlight desaturation toward a white-hot core. Set to 0.0 for bypass.";
    ui_category = "2. Tone Mapping";
> = 0.00;

uniform float fConeResponseExponent <
    ui_type     = "slider";
    ui_min      = 0.50; ui_max = 2.00; ui_step = 0.01;
    ui_label    = "Cone Response Exponent (n)";
    ui_tooltip  = "Adjusts the steepness of the highlight compression curve.\n"
                  "Values < 1.0 flatten and open the shoulder for a wider, softer highlight roll-off.\n"
                  "Values > 1.0 steepen the shoulder for a punchier transition.";
    ui_category = "2. Tone Mapping";
> = 1.00;

uniform float fHighlightsCurvature <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Highlight Compression Curvature (h)";
    ui_tooltip  = "Decoupled technical shoulder shape of the tone mapper.\n"
                  "Set to 0.0 to enable Adaptive Reference Shoulder mode.";
    ui_category = "2. Tone Mapping";
> = 0.00;

uniform float fHueRestore <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "MacLeod-Boynton Hue Restore";
    ui_tooltip  = "Blends back the original scene hue direction in compressed highlights while keeping the tonemapped purity and luminance intact.";
    ui_category = "2. Tone Mapping";
> = 0.00;

uniform int iToneMapperMode <
    ui_type     = "combo";
    ui_label    = "Tone Mapping Operator";
    ui_items    = "Bypass\0Khronos PBR Neutral\0Non-Riemannian Geodesic (NRG-TM)\0";
    ui_tooltip  = "NRG-TM operates physiologically inside the LMS/MB spaces and counteracts Bezold-Brücke hue shifts.";
    ui_category = "2. Tone Mapping";
> = 0;

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
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Desaturation Strength";
    ui_tooltip  = "Controls opponent channel depletion in NRG-TM highlight shoulders.";
    ui_category = "2. Tone Mapping";
> = 0.15;

uniform float fMConeCrosstalk <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Spectral M-Cone Crosstalk";
    ui_tooltip  = "Bends highly saturated red/blue highlights toward green/yellow (M-cone region)\n"
                  "to mimic human visual confusion/crosstalk near spectral boundaries.\n"
                  "Prevents harsh, out-of-gamut clipping in sky and sunsets.";
    ui_category = "3. Advanced HVS";
> = 0.00;

uniform int iColorSpaceOverride <
    ui_type     = "combo";
    ui_label    = "Color Space Override";
    ui_items    = "Auto (Default via ReShade)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0HLG (HDR)\0";
    ui_tooltip  = "Container format detection override.";
    ui_category = "System";
> = 0;

uniform float fWhitePoint <
    ui_type     = "slider";
    ui_min      = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_label    = "Reference White (Nits)";
    ui_tooltip  = "Paper white anchor for HDR mapping.";
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
                  "Compression Map\0"
                  "Skin Protection Mask\0";
    ui_tooltip  = "Debug visualizations operate on the fully graded output.";
    ui_category = "Debug";
> = 0;

// =================================================================================================
// 4. True Math Utilities (IEEE 754 Compliant)
// =================================================================================================

float PowNonNegPreserveZero(float x, float e)
{
    return (x <= 0.0) ? 0.0 : pow(x, e);
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

bool3 IsNan3(float3 v) { return (asuint(v) & 0x7FFFFFFFu) > 0x7F800000u; }
bool3 IsInf3(float3 v) { return (asuint(v) & 0x7FFFFFFFu) == 0x7F800000u; }

// =================================================================================================
// 5. Color Science & EOTF Utilities
// =================================================================================================

float3 sRGB_EOTF(float3 V)
{
    float3 abs_V  = abs(V);
    float3 lin_lo = abs_V / 12.92;
    float3 lin_hi = PowNonNegPreserveZero3((abs_V + 0.055) / 1.055, SRGB_GAMMA);
    float3 out_lin = (abs_V <= SRGB_THRESHOLD_EOTF) ? lin_lo : lin_hi;
    return sign(V) * out_lin;
}

float3 sRGB_OETF(float3 L)
{
    float3 abs_L  = abs(L);
    float3 enc_lo = abs_L * 12.92;
    float3 enc_hi = 1.055 * PowNonNegPreserveZero3(abs_L, SRGB_INV_GAMMA) - 0.055;
    float3 out_enc = (abs_L <= SRGB_THRESHOLD_OETF) ? enc_lo : enc_hi;
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

// NOTE: Intentionally NO inverse-OOTF/OOTF pair is applied for HLG. Game backbuffers are
// display-referred already; applying the BT.2100 scene-referred system gamma (~1.2 @ 1000 nit)
// here would double-apply it. If you feed true scene-referred HLG, insert the OOTF explicitly.
float3 HLG_EOTF(float3 x)
{
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.55991073;
    
    float3 r;
    r.r = (x.r <= 0.5) ? (x.r * x.r) / 3.0 : (exp((x.r - c) / a) + b) / 12.0;
    r.g = (x.g <= 0.5) ? (x.g * x.g) / 3.0 : (exp((x.g - c) / a) + b) / 12.0;
    r.b = (x.b <= 0.5) ? (x.b * x.b) / 3.0 : (exp((x.b - c) / a) + b) / 12.0;
    
    return r * 1000.0;
}

float3 HLG_OETF(float3 x)
{
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.55991073;
    
    float3 E = max(x / 1000.0, 0.0);
    float3 r;
    r.r = (E.r <= 1.0 / 12.0) ? sqrt(3.0 * E.r) : a * log(max(12.0 * E.r - b, FLT_MIN)) + c;
    r.g = (E.g <= 1.0 / 12.0) ? sqrt(3.0 * E.g) : a * log(max(12.0 * E.g - b, FLT_MIN)) + c;
    r.b = (E.b <= 1.0 / 12.0) ? sqrt(3.0 * E.b) : a * log(max(12.0 * E.b - b, FLT_MIN)) + c;
    
    return r;
}

float3 DecodeToLinear(float3 encoded, int space)
{
    [branch] if (space == 4) return HLG_EOTF(encoded);
    [branch] if (space == 3) return PQ_EOTF(encoded);
    [branch] if (space == 2) return encoded * SCRGB_WHITE_NITS;
    return sRGB_EOTF(encoded) * SCRGB_WHITE_NITS;
}

float3 EncodeFromLinear(float3 lin, int space)
{
    [branch] if (space == 4) return HLG_OETF(lin);
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
 * Evaluate3DSkinLocusMB
 *
 * Evaluates the 3D volumetric skin-tone confidence in MacLeod-Boynton cone-opponent space.
 * Covers all Fitzpatrick skin types (I-VI) based on physiological Melanin-Hemoglobin curves.
 */
float Evaluate3DSkinLocusMB(float luma_norm, float3 mb)
{
    // 1. L / (L+M) Gate (Red/Green Cone Balance): Human skin sits within [0.510, 0.590]
    float l_gate = smoothstep(0.505, 0.515, mb.x) * (1.0 - smoothstep(0.585, 0.620, mb.x));

    // 2. S / (L+M) Gate correlated with Lightness (Melanin-Hemoglobin axis)
    // Fair skin (high luma) has high S (0.35-0.45); Dark skin (low luma) has lower S (0.20-0.30)
    float expected_s_min = clamp(0.16 + 0.16 * luma_norm, 0.16, 0.32);
    float s_gate = smoothstep(expected_s_min - 0.03, expected_s_min + 0.02, mb.y) * (1.0 - smoothstep(0.44, 0.48, mb.y));

    // 3. Lightness Gate: covers deep shadows to highlights
    float luma_gate = smoothstep(0.010, 0.035, luma_norm) * (1.0 - smoothstep(0.95, 1.30, luma_norm));

    return saturate(l_gate * s_gate * luma_gate);
}

/**
 * ComputeBlackPointRatio
 */
float ComputeBlackPointRatio(float luma, float bpNits, float shadowFloor)
{
    float raw = max((luma - bpNits) / max(luma, FLT_MIN), shadowFloor);
    float t = saturate(luma / max(4.0 * bpNits, FLT_MIN));
    float smooth_t = t * t * (3.0 - 2.0 * t);
    return lerp(shadowFloor, raw, smooth_t);
}

/**
 * SolveGamutBoundaryExact (24-Step Solver with full Rec.2020 coverage t_high = 8.0)
 */
float SolveGamutBoundaryExact(float2 chroma_direction, float3 luma_LMS_coeffs, float3x3 to_RGB_boundary, float2 mb_white)
{
    float t_low = 0.0;
    float t_high = 8.0; // Encompasses pure Rec.2020 Blue (distance 5.21 in MB space)
    
    [unroll]
    for (int iter = 0; iter < 24; iter++)
    {
        float t = 0.5 * (t_low + t_high);
        float x = mb_white.x + t * chroma_direction.x;
        float y = mb_white.y + t * chroma_direction.y;
        
        float denom = x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
        
        if (denom <= FLT_MIN)
        {
            t_high = t;
        }
        else
        {
            float3 test_lms = float3(x, 1.0 - x, y);
            float3 rgb_point = mul(to_RGB_boundary, test_lms);
            float min_rgb = min(min(rgb_point.r, rgb_point.g), rgb_point.b);
            
            if (min_rgb < 0.0 || !(min_rgb >= 0.0))
                t_high = t;
            else
                t_low = t;
        }
    }
    return t_low;
}

/**
 * Troland Bleaching (LMS-In / LMS-Out)
 */
float3 ApplyTrolandBleachingLMS(float3 lms, float strength, float3 luma_LMS_coeffs)
{
    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0 || strength <= NEUTRAL_EPS) return lms;
    
    float3 safe_lms = max(lms, 0.0);
    float3 stimulus = safe_lms * TROLAND_LMS_SCALE;
    float stim_lm = 0.5 * (stimulus.r + stimulus.g);
    
    float availability = 1.0 / (1.0 + (stim_lm / max(TROLAND_HALF_SAT, FLT_MIN)));
    float k = lerp(1.0, availability, saturate(strength));
    
    float luma = dot(lms, luma_LMS_coeffs);
    float denom = dot(float3(1.0, 1.0, 1.0), luma_LMS_coeffs);
    float3 neutral = float3(1.0, 1.0, 1.0) * (luma / max(denom, FLT_MIN));
    
    return lerp(neutral, lms, k);
}

/**
 * ApplyMBPurityAndGamutGuardLMS (V6.1.6 - Consistent-Normalization Edition)
 *
 * Fixes over V6.1.5:
 *  - Chroma reliability gate evaluated against WHITE-RELATIVE luminance (was absolute nits,
 *    which made the gate inert on all visible content).
 *  - Boundary solve for the normalization reference (relative_purity) is now UNCONDITIONAL,
 *    so vibrance demand curves and diminishing-returns behave identically in Unclamped mode.
 *  - Master Saturation = 0 forces exact monochrome collapse regardless of Vibrance.
 */
float3 ApplyMBPurityAndGamutGuardLMS(
    float3 lms, 
    float purity_scale, 
    float vibrance_amount,
    float skin_protection,
    int gamut_target_mode,
    float knee, 
    float abney_correction, 
    float3 luma_LMS_coeffs, 
    float3x3 to_RGB_boundary, 
    float2 mb_white, 
    float whitePt)
{
    // Inhibit bypass if an explicit gamut clamp is requested
    bool is_unclamped = (gamut_target_mode == 4);
    bool force_gamut_clamp = (gamut_target_mode == 1 || gamut_target_mode == 2 || gamut_target_mode == 3);

    if (!force_gamut_clamp &&
        abs(purity_scale - 1.0) < NEUTRAL_EPS && 
        abs(vibrance_amount) < NEUTRAL_EPS && 
        knee < NEUTRAL_EPS && 
        abney_correction < NEUTRAL_EPS)
    {
        return lms;
    }

    float lm_sum = lms.r + lms.g;
    if (lm_sum <= FLT_MIN) return lms;

    float luma = dot(lms, luma_LMS_coeffs);
    if (luma <= FLT_MIN) return lms; // Guard against crushed/negative luminance

    // V6.1.6: White-relative chroma reliability (0.05% .. 0.2% of diffuse white ramp)
    float ct = saturate((luma / max(whitePt, FLT_MIN) - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);

    if (chroma_reliability <= 0.0 && purity_scale >= 1.0 && vibrance_amount >= 0.0 && !force_gamut_clamp) return lms;

    float3 mb = LMS_to_MB(lms);
    float2 chroma_offset = mb.xy - mb_white;
    float purity = SqrtIEEE(dot(chroma_offset, chroma_offset));

    // True monochrome collapse.
    // V6.1.6: Master purity zero now OVERRIDES vibrance (exact R=G=B guaranteed).
    if (purity < 1e-6 || purity_scale <= NEUTRAL_EPS)
    {
        mb.xy = mb_white;
        float denom = mb.x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + mb.y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
        mb.z = luma / max(denom, FLT_MIN);
        return MB_to_LMS(mb);
    }

    float2 chroma_dir = chroma_offset / max(purity, FLT_MIN);
    float relative_lightness = luma / max(whitePt, FLT_MIN);

    // V6.1.6: ALWAYS solve the boundary along the current chroma direction. This value is the
    // normalization reference for the Weber-Fechner demand curve AND the diminishing-returns
    // coefficient, so it must not depend on whether clamping is enabled. Consistency guarantee:
    // identical slider settings produce identical scaling in every gamut-target mode.
    float initial_max_purity = SolveGamutBoundaryExact(chroma_dir, luma_LMS_coeffs, to_RGB_boundary, mb_white);
    float relative_purity = saturate(purity / max(initial_max_purity, FLT_MIN));

    // --- 1. SMART SATURATION (VIBRANCE) & 3D SKIN PROTECTION ---
    float effective_scale = purity_scale;

    // 3D Skin Protection Evaluation
    float skin_confidence = Evaluate3DSkinLocusMB(relative_lightness, mb);
    float effective_skin_mask = saturate(skin_confidence * skin_protection);

    // Protect skin from master saturation boosts
    if (effective_scale > 1.0 && effective_skin_mask > NEUTRAL_EPS)
    {
        float master_boost = effective_scale - 1.0;
        master_boost *= (1.0 - effective_skin_mask * 0.85);
        effective_scale = 1.0 + master_boost;
    }

    // Smart Saturation (Vibrance)
    // V6.1.6: Gain is gated by saturate(purity_scale) so vibrance smoothly couples to (and dies
    // with) the master saturation control instead of resurrecting color from grayscale.
    if (abs(vibrance_amount) > NEUTRAL_EPS)
    {
        // Normalized Weber-Fechner Demand based on Relative Purity (Uniform across all hues!)
        float saturation_demand = exp(-3.5 * pow(relative_purity, 1.25));
        float vibrance_gain = vibrance_amount * saturation_demand
                            * saturate(purity_scale)
                            * (1.0 - effective_skin_mask) * chroma_reliability;
        effective_scale *= max(1.0 + vibrance_gain, 0.0);
    }

    // Non-Riemannian Diminishing Returns for Extreme Boosts (Uniform Relative Scaling)
    if (effective_scale > 1.0)
    {
        float diminishing_returns_coeff = 0.35;
        float boost = effective_scale - 1.0;
        boost = boost / (1.0 + diminishing_returns_coeff * relative_purity * boost);
        effective_scale = 1.0 + boost;
    }

    effective_scale = lerp(1.0, effective_scale, chroma_reliability);
    
    // Strictly radial purity scaling (preserving exact chromaticity angle and preventing hue distortion)
    float2 scaled_chroma_offset = chroma_offset * effective_scale;

    // --- 2. ABNEY HUE COMPENSATION ---
    if (abney_correction > NEUTRAL_EPS)
    {
        float angle = atan2(chroma_offset.y, chroma_offset.x);
        float abney_profile = 0.15 * sin(2.0 * angle + 0.4) * (1.0 + 0.3 * cos(angle));
        float shift = abney_profile * relative_purity * abney_correction * chroma_reliability;
        angle += shift;

        float scaled_purity_val = SqrtIEEE(dot(scaled_chroma_offset, scaled_chroma_offset));
        scaled_chroma_offset = float2(cos(angle), sin(angle)) * scaled_purity_val;
    }

    // --- 3. SYNCHRONIZED GAMUT GUARD COMPRESSION & CLAMP (Skipped if Unclamped) ---
    float scaled_purity = SqrtIEEE(dot(scaled_chroma_offset, scaled_chroma_offset));
    
    if (!is_unclamped && scaled_purity > FLT_MIN)
    {
        // Re-solve boundary along the NEW, post-Abney rotated direction
        float2 new_chroma_dir = scaled_chroma_offset / scaled_purity;
        float max_purity = (abney_correction > NEUTRAL_EPS)
            ? SolveGamutBoundaryExact(new_chroma_dir, luma_LMS_coeffs, to_RGB_boundary, mb_white)
            : initial_max_purity;

        if (max_purity > FLT_MIN)
        {
            // Soft-Knee Gamut Compression
            if (knee > FLT_MIN)
            {
                float threshold = max_purity * (1.0 - knee);
                if (scaled_purity > threshold && threshold > FLT_MIN)
                {
                    float excess = scaled_purity - threshold;
                    float headroom = max_purity - threshold;
                    float compressed = threshold + headroom * (1.0 - exp(-excess / max(headroom, FLT_MIN)));
                    scaled_chroma_offset = new_chroma_dir * compressed;
                    scaled_purity = compressed;
                }
            }

            // Strict Hard Boundary Clamp
            float p_safe = max_purity * (1.0 - NEUTRAL_EPS);
            if (scaled_purity > p_safe)
            {
                scaled_chroma_offset = new_chroma_dir * p_safe;
                scaled_purity = p_safe;
            }

            // Analytical RGB boundary validation with exact closed-form z-recovery
            mb.xy = mb_white + scaled_chroma_offset;
            float denom = mb.x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + mb.y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
            mb.z = luma / max(denom, FLT_MIN);

            float3 lms_test = MB_to_LMS(mb);
            float3 boundary_check = mul(to_RGB_boundary, lms_test);
            float min_b = min(min(boundary_check.r, boundary_check.g), boundary_check.b);

            if (min_b < 0.0)
            {
                // Pull back along the ray to the safe boundary
                float p_exact = SolveGamutBoundaryExact(new_chroma_dir, luma_LMS_coeffs, to_RGB_boundary, mb_white) * (1.0 - NEUTRAL_EPS);
                scaled_chroma_offset = new_chroma_dir * min(scaled_purity, p_exact);
            }
        }
    }

    mb.xy = mb_white + scaled_chroma_offset;

    // Closed-form rational recovery of LMS lightness z on constant luminance plane Y (Delta-Y lock,
    // ~1 ulp floating-point residual through the round-trip)
    float denom = mb.x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + mb.y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
    mb.z = luma / max(denom, FLT_MIN);

    return MB_to_LMS(mb);
}

/**
 * ApplyMConeCrosstalkLMS
 */
float3 ApplyMConeCrosstalkLMS(float3 lms, float strength, float3 lms_before_tm, float3 luma_LMS_coeffs, float2 mb_white, float whitePt)
{
    if (strength <= NEUTRAL_EPS) return lms;

    float original_Y = dot(lms, luma_LMS_coeffs);
    if (original_Y <= 0.0) return lms;

    float3 global_white_lms = float3(1.0, 1.0, 1.0) * whitePt;
    float3 drive = abs(lms_before_tm) / max(global_white_lms, FLT_MIN);

    float l_over_m = max(drive.x - drive.y, 0.0);
    float s_over_m = max(drive.z - drive.y, 0.0);

    float ls_mixed_share = l_over_m + s_over_m > FLT_MIN ? min(l_over_m, s_over_m) / (l_over_m + s_over_m) : 0.0;
    float cone_complementary_gate = smoothstep(0.02, 0.15, ls_mixed_share);

    float3 mb = LMS_to_MB(lms);
    float2 offset = mb.xy - mb_white;

    float rg = offset.x;
    float yv = offset.y;
    float chroma = max(length(offset), FLT_MIN);
    float rg_pos = saturate(rg / chroma);
    float yv_pos = saturate(yv / chroma);
    float mixed = rg_pos * yv_pos > FLT_MIN ? min(abs(rg), abs(yv)) / max(max(abs(rg), abs(yv)), FLT_MIN) : 0.0;
    float acc_complementary_gate = saturate(2.0 * rg_pos * yv_pos * mixed);

    float spectral_confidence = 1.0 - max(cone_complementary_gate, acc_complementary_gate);

    float lm_share = drive.x + drive.y > FLT_MIN ? drive.y / (drive.x + drive.y) : 0.0;
    float l_bias = saturate(l_over_m / max(drive.x, FLT_MIN)) * saturate(lm_share / 0.25);
    float s_bias = 0.15 * saturate(s_over_m / max(drive.z, FLT_MIN));

    float m_bias_weight = spectral_confidence * (0.12 * l_bias + 0.025 * s_bias) * strength;

    float target_radius = length(offset);
    float2 m_offset = -mb_white;
    float m_radius = length(m_offset);

    if (target_radius > FLT_MIN && m_radius > FLT_MIN)
    {
        float2 bent_offset = lerp(
            offset,
            m_offset * (target_radius / m_radius),
            saturate(m_bias_weight)
        );

        bent_offset *= target_radius / max(length(bent_offset), FLT_MIN);
        mb.xy = mb_white + bent_offset;

        float denom = mb.x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + mb.y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
        mb.z = original_Y / max(denom, FLT_MIN);
        return MB_to_LMS(mb);
    }

    return lms;
}

// =================================================================================================
// 7. Tonemapping Functions
// =================================================================================================

/**
 * ComputeKhronosParams (V6.1.6)
 *
 * Single source of truth for the Khronos PBR Neutral shoulder geometry. Used by BOTH the tone
 * mapper and the debug compression-map visualization, eliminating the drift risk of duplicated
 * formulas.
 */
void ComputeKhronosParams(
    float3 safeColor,
    float  targetPeak,
    float  compressionStart,
    out float offset,
    out float peak,
    out float startComp,
    out float d,
    out float newPeak)
{
    float x = min(safeColor.r, min(safeColor.g, safeColor.b));

    offset    = x < 0.08 ? x - 6.25 * x * x : 0.04;
    peak      = max(safeColor.r, max(safeColor.g, safeColor.b)) - offset;
    startComp = (targetPeak * compressionStart) - 0.04;
    d         = targetPeak - startComp;
    newPeak   = targetPeak - (d * d) / (peak + d - startComp);
}

float3 ApplyKhronosPBRNeutral(float3 color, float targetPeak, float compressionStart, float desatStrength)
{
    float3 safeColor = max(color, 0.0);

    float offset, peak, startComp, d, newPeak;
    ComputeKhronosParams(safeColor, targetPeak, compressionStart, offset, peak, startComp, d, newPeak);

    [branch]
    if (peak >= startComp && startComp > 0.0)
    {
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

float CompressLumaPhysiological(float Y, float targetPeak, float startComp, float exponent, float highlights)
{
    if (Y < startComp) return Y;
    
    float d = targetPeak - startComp;
    float x = Y - startComp;
    
    float h = max(exponent * highlights, 1e-6);
    float ratio = x / max(d, FLT_MIN);
    float compressed_x = x / pow(1.0 + pow(max(ratio, 0.0), h), 1.0 / h);
    
    return startComp + compressed_x;
}

/**
 * ApplyNonRiemannianGeodesicToneMapper (3D Cusp-Preserving Color Volume Edition)
 * 
 * Accurately respects the physical display gamut cusp:
 * - Compresses max(R,G,B) <= targetPeakNits
 * - Preserves exact chromaticity ratios (R/M, G/M, B/M) with ZERO pink/cyan washout
 * - Saturated Red correctly targets <= 210 nits, Blue <= 47 nits, White <= 800 nits
 *
 * V6.1.6: Removed unused mb_white parameter.
 */
float3 ApplyNonRiemannianGeodesicToneMapper(
    float3 lms, 
    float targetPeak, 
    float compressionStart, 
    float desatStrength, 
    float coneExponent, 
    float h_input, 
    float3 luma_LMS_coeffs, 
    float3x3 to_RGB, 
    float3x3 to_LMS,
    float whitePt)
{
    float Y = dot(lms, luma_LMS_coeffs);
    if (Y <= 0.0) return lms;

    float targetPeakNits = targetPeak * whitePt;
    float startCompNits  = targetPeakNits * compressionStart;

    // Convert to display RGB to evaluate the true physical color volume
    float3 rgb = mul(to_RGB, lms);
    float max_channel = max(rgb.r, max(rgb.g, rgb.b));

    if (max_channel <= FLT_MIN) return lms;

    // 1. Cusp-Preserving Tone Compression on the Max Driver Channel
    // This maintains exact chromaticity ratios without distorting color purity.
    float compressed_max = max_channel;
    if (max_channel > startCompNits)
    {
        float d = targetPeakNits - startCompNits;
        float x = max_channel - startCompNits;
        float h = max(coneExponent * h_input, 1e-6);
        float ratio = x / max(d, FLT_MIN);
        float compressed_x = x / pow(1.0 + pow(max(ratio, 0.0), h), 1.0 / h);
        compressed_max = startCompNits + compressed_x;
    }

    // Exact scale factor that maps max(R,G,B) <= targetPeakNits with zero hue/purity distortion
    float cusp_scale = compressed_max / max(max_channel, FLT_MIN);
    float3 rgb_cusp_mapped = rgb * cusp_scale;

    // 2. Controlled Specular Bleaching (Only for extreme over-bright highlights)
    // Saturated colored lights retain their purity; only extreme speculars bleach to white.
    if (desatStrength > NEUTRAL_EPS)
    {
        float overexposure = max_channel / max(targetPeakNits, FLT_MIN);
        
        // Bleach factor activates only when light exceeds display peak significantly
        float bleach_gate = saturate((overexposure - 1.0) / 3.0);
        float bleach_k = desatStrength * bleach_gate;

        float3 white_target = float3(compressed_max, compressed_max, compressed_max);
        rgb_cusp_mapped = lerp(rgb_cusp_mapped, white_target, bleach_k);

        // Re-normalize to prevent the blend from exceeding targetPeakNits
        float post_max = max(rgb_cusp_mapped.r, max(rgb_cusp_mapped.g, rgb_cusp_mapped.b));
        if (post_max > targetPeakNits)
        {
            rgb_cusp_mapped *= (targetPeakNits / post_max);
        }
    }

    return mul(to_LMS, rgb_cusp_mapped);
}

// =================================================================================================
// 8. Debug Visualization Functions
// =================================================================================================

float3 EncodeDebug(float3 debug_out, int space)
{
    debug_out = max(debug_out, 0.0);
    [branch]
    if (space == 4)
    {
        return HLG_OETF(lerp(100.0, 600.0, saturate(debug_out)));
    }
    else if (space == 3)
    {
        return PQ_InverseEOTF(lerp(100.0, 600.0, saturate(debug_out)));
    }
    else if (space == 2)
    {
        // V6.1.6: scRGB is a LINEAR container - emitting raw 0..1 renders a nearly invisible
        // overlay (<= 80 nits with no perceptual curve). Remap onto a visible 4..200 nit band
        // (relative to the 80-nit scRGB white) so debug patterns read like the other spaces.
        return lerp(0.05, 2.5, saturate(debug_out));
    }
    else
    {
        return sRGB_OETF(saturate(debug_out));
    }
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
    return saturate(abs(frac(hue + float3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
}

float ComputeBleachingKLMS(float3 lms, float strength)
{
    if (strength <= NEUTRAL_EPS) return 1.0;
    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0) return 1.0;

    float3 safe_lms = max(lms, 0.0);
    float3 stimulus = safe_lms * TROLAND_LMS_SCALE;
    float stim_lm   = 0.5 * (stimulus.r + stimulus.g);
    float availability = 1.0 / (1.0 + (stim_lm / max(TROLAND_HALF_SAT, FLT_MIN)));
    return lerp(1.0, availability, saturate(strength));
}

/**
 * ComputeCompressionRatio (V6.1.6)
 * Now delegates to ComputeKhronosParams - guaranteed to stay in sync with the tone mapper.
 */
float ComputeCompressionRatio(float3 color, float targetPeak, float compressionStart)
{
    float3 safeColor = max(color, 0.0);

    float offset, peak, startComp, d, newPeak;
    ComputeKhronosParams(safeColor, targetPeak, compressionStart, offset, peak, startComp, d, newPeak);

    if (peak >= startComp && startComp > 0.0)
    {
        return newPeak / max(peak, FLT_MIN);
    }
    return 1.0;
}

// =================================================================================================
// 9. Custom Vertex Shader
// =================================================================================================

struct VS_Output
{
    float4 vpos : SV_Position;
    float2 texcoord : TEXCOORD0;
    nointerpolation float3 wbScale : TEXCOORD1;
    nointerpolation float3 luma_LMS_coeffs : TEXCOORD2;
};

VS_Output VS_PhotorealHDR(uint id : SV_VertexID)
{
    VS_Output output;
    
    output.texcoord.x = (id == 2) ? 2.0 : 0.0;
    output.texcoord.y = (id == 1) ? 2.0 : 0.0;
    output.vpos = float4(output.texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    
    float3 lumaCoeffs;
    float3x3 to_RGB;
    if (space >= 3)
    {
        lumaCoeffs = Luma2020;
        to_RGB     = LMS_to_RGB2020;
    }
    else
    {
        lumaCoeffs = Luma709;
        to_RGB     = LMS_to_RGB709;
    }

    float3 luma_LMS = mul(lumaCoeffs, to_RGB);
    output.luma_LMS_coeffs = luma_LMS;

    float3 wbStopsLMS = 0.35 * float3(
        fTemperature + fTint,
        -fTint,
        -fTemperature + fTint
    );
    float3 wbScaleLMS = exp2(wbStopsLMS);
    
    float lumaScale = dot(wbScaleLMS, luma_LMS);
    float refLuma   = dot(float3(1.0, 1.0, 1.0), luma_LMS);
    output.wbScale  = wbScaleLMS * (refLuma / max(lumaScale, FLT_MIN));

    return output;
}

// =================================================================================================
// 10. Main Pipeline Shader
// =================================================================================================

void PS_PhotorealHDR(VS_Output input, out float4 fragColor : SV_Target)
{
    int2 pos   = int2(input.vpos.xy);
    float4 src = tex2Dfetch(SamplerBackBuffer, pos);

    int space         = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float whitePt     = (space <= 1) ? SCRGB_WHITE_NITS : fWhitePoint;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;

    [branch]
    if (iDebugMode == 0 &&
        abs(fExposure) < NEUTRAL_EPS && abs(fBlackPoint) < NEUTRAL_EPS &&
        abs(fContrast - 1.0) < NEUTRAL_EPS && abs(fShadows) < NEUTRAL_EPS &&
        abs(fHighlights) < NEUTRAL_EPS && abs(fTemperature) < NEUTRAL_EPS &&
        abs(fTint) < NEUTRAL_EPS && abs(fSaturation - 1.0) < NEUTRAL_EPS &&
        abs(fVibrance) < NEUTRAL_EPS &&
        fBleaching < NEUTRAL_EPS && iToneMapperMode == 0 &&
        fAbneyCorrection < NEUTRAL_EPS && fGamutGuardKnee < NEUTRAL_EPS &&
        fMConeCrosstalk < NEUTRAL_EPS && fHueRestore < NEUTRAL_EPS &&
        iGamutTarget == 0)
    {
        fragColor = src;
        return;
    }

    // Decode & Sanitize
    float3 original_lin = DecodeToLinear(src.rgb, space);
    bool is_invalid = any(IsNan3(original_lin)) || any(IsInf3(original_lin));
    original_lin = is_invalid ? (0.18 * whitePt).xxx : original_lin;

    float3x3 to_LMS, to_RGB;
    float3x3 to_RGB_boundary;

    [branch]
    if (space >= 3)
    {
        to_LMS = RGB2020_to_LMS;
        to_RGB = LMS_to_RGB2020;
        to_RGB_boundary = LMS_to_RGB2020;
    }
    else if (space == 2)
    {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
        // scRGB policy (V6.1.6, documented): the container is open-ended - wide gamut is carried
        // via negative/extended 709-channel excursions and survives the encode path unclamped.
        // The GUARD therefore bounds chromaticity against the Rec.2020 volume rather than the
        // 709 primaries, preserving legitimate WCG while preventing spectral garbage.
        to_RGB_boundary = LMS_to_RGB2020; 
    }
    else
    {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
        to_RGB_boundary = LMS_to_RGB709;
    }

    // Select Gamut Guard Boundary Matrix
    [branch]
    if (iGamutTarget == 1)      to_RGB_boundary = LMS_to_RGB709;
    else if (iGamutTarget == 2) to_RGB_boundary = LMS_to_P3D65;
    else if (iGamutTarget == 3) to_RGB_boundary = LMS_to_RGB2020;

    float2 mb_white = MB_WHITE_D65;

    // ---------------------------------------------------------------------------------------------
    // CONVERT TO LMS DOMAIN
    // ---------------------------------------------------------------------------------------------
    float3 lms = mul(to_LMS, original_lin);
    
    // ---------------------------------------------------------------------------------------------
    // STAGE 1: EXPOSURE & WHITE BALANCE
    // ---------------------------------------------------------------------------------------------
    lms *= input.wbScale;

    if (abs(fExposure) > NEUTRAL_EPS) 
    {
        lms *= exp2(fExposure);
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 2: DEHAZE & CONTRAST
    // ---------------------------------------------------------------------------------------------
    float3 lms_pre_grading = lms; 
    float luma = dot(lms_pre_grading, input.luma_LMS_coeffs);

    // V6.1.6: Extended-range passthrough - negative-luma pixels (legitimate scRGB WCG excursions)
    // bypass dehaze entirely instead of being crushed toward black by the shadow floor.
    float bp_ratio = 1.0;
    if (fBlackPoint > NEUTRAL_EPS && luma > 0.0)
    {
        float bpNits = fBlackPoint * whitePt;
        bp_ratio = ComputeBlackPointRatio(luma, bpNits, fShadowFloor);
    }

    float contrast_ratio = 1.0;
    float graded_luma = max(luma * bp_ratio, FLT_MIN);
    float absLuma = graded_luma;
    
    // V6.1.6: Contrast stage likewise skips non-positive luma (log-domain undefined).
    [branch]
    if (absLuma > FLT_MIN && luma > 0.0)
    {
        float pivot = fContrastPivot * whitePt;
        float logRatio = log2(absLuma / pivot);

        float x = logRatio * fContrast;
        float S = fShadows * 3.0;
        float H = fHighlights * 3.0;
        float rational_factor = (x * x) / (x * x + 6.0);
        
        float blend_t = saturate(0.5 + x * 4.0);
        float recovery = lerp(S, H, blend_t);
        x += recovery * rational_factor;

        float contrastLuma = pivot * exp2(x);
        float ratio = contrastLuma / absLuma;

        // Asymptotic soft limiter: monotonic, approaches a x100 ceiling (never exceeds it).
        // (Documented honestly: this does NOT hard-stop at x80; it converges to x100.)
        float excess = max(ratio - 80.0, 0.0);
        contrast_ratio = min(ratio, 80.0) + (excess / (1.0 + excess / 20.0));
    }

    lms *= bp_ratio * contrast_ratio;

    // ---------------------------------------------------------------------------------------------
    // STAGE 3: BIOLOGICAL HIGHLIGHT BLEACHING
    // ---------------------------------------------------------------------------------------------
    float3 lms_pre_bleach = lms;
    lms = ApplyTrolandBleachingLMS(lms, fBleaching, input.luma_LMS_coeffs);

    // ---------------------------------------------------------------------------------------------
    // STAGE 4: TONE MAPPING
    // ---------------------------------------------------------------------------------------------
    float3 color = float3(0.0, 0.0, 0.0);
    float3 pre_khronos_color = float3(0.0, 0.0, 0.0);
    float tone_comp_ratio = 1.0;

    float targetPeak = (space <= 1) ? 1.0 : max(1.0, fDisplayPeakNits / whitePt);

    [branch]
    if (iToneMapperMode == 1) // Khronos PBR Neutral
    {
        color = mul(to_RGB, lms);
        pre_khronos_color = color;
        color /= max(whitePt, FLT_MIN);

        color = ApplyKhronosPBRNeutral(color, targetPeak, fCompressionStart, fDesaturationStrength);

        color *= whitePt;
        lms = mul(to_LMS, color);
    }
    else if (iToneMapperMode == 2) // Non-Riemannian Geodesic Tone Mapper (NRG-TM)
    {
        float3 lms_before = lms;
        
        float h;
        if (abs(fHighlightsCurvature) < NEUTRAL_EPS)
        {
            float peak_over_anchor = max(targetPeak, 1.0 + FLT_MIN);
            float reference_one_side_range_log10 = 3.7 * 0.5;
            float actual_above_adaptation_range_log10 = max(log2(peak_over_anchor) * 0.3010299956639812, FLT_MIN);
            h = max(reference_one_side_range_log10 / actual_above_adaptation_range_log10, 1.0);
        }
        else
        {
            h = exp2(fHighlightsCurvature);
        }
        
        lms = ApplyNonRiemannianGeodesicToneMapper(
            lms, 
            targetPeak, 
            fCompressionStart, 
            fDesaturationStrength, 
            fConeResponseExponent, 
            h, 
            input.luma_LMS_coeffs, 
            to_RGB, 
            to_LMS,
            whitePt
        );        
        float Y_before = dot(lms_before, input.luma_LMS_coeffs);
        float Y_after  = dot(lms, input.luma_LMS_coeffs);
        tone_comp_ratio = Y_after / max(Y_before, FLT_MIN);
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 5: PHYSIOLOGICAL HUE RESTORATION & SPECTRAL CROSSTALK
    // ---------------------------------------------------------------------------------------------
    if (fHueRestore > NEUTRAL_EPS)
    {
        float3 mb_src = LMS_to_MB(lms_pre_bleach);
        float2 source_offset = mb_src.xy - mb_white;

        float3 mb_tgt = LMS_to_MB(lms);
        float2 target_offset = mb_tgt.xy - mb_white;

        float target_radius = length(target_offset);
        float source_len = length(source_offset);

        if (target_radius > FLT_MIN && source_len > FLT_MIN)
        {
            float2 blended_offset = lerp(
                target_offset,
                source_offset * (target_radius / source_len),
                saturate(fHueRestore)
            );

            blended_offset *= target_radius / max(length(blended_offset), FLT_MIN);
            mb_tgt.xy = mb_white + blended_offset;

            float denom = mb_tgt.x * (input.luma_LMS_coeffs.r - input.luma_LMS_coeffs.g) + mb_tgt.y * input.luma_LMS_coeffs.b + input.luma_LMS_coeffs.g;
            mb_tgt.z = dot(lms, input.luma_LMS_coeffs) / max(denom, FLT_MIN);

            lms = MB_to_LMS(mb_tgt);
        }
    }

    lms = ApplyMConeCrosstalkLMS(lms, fMConeCrosstalk, lms_pre_bleach, input.luma_LMS_coeffs, mb_white, whitePt);

    // ---------------------------------------------------------------------------------------------
    // STAGE 6: PURITY, SMART SATURATION & GAMUT GUARD
    // ---------------------------------------------------------------------------------------------
    lms = ApplyMBPurityAndGamutGuardLMS(
        lms, 
        fSaturation, 
        fVibrance,
        fSkinProtection,
        iGamutTarget,
        fGamutGuardKnee, 
        fAbneyCorrection, 
        input.luma_LMS_coeffs, 
        to_RGB_boundary, 
        mb_white, 
        whitePt
    );

    // ---------------------------------------------------------------------------------------------
    // STAGE 7: FINAL RGB RECONSTRUCTION
    // ---------------------------------------------------------------------------------------------
    color = mul(to_RGB, lms);

    is_invalid = any(IsNan3(color)) || any(IsInf3(color));
    color = is_invalid ? original_lin : color;

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
            float k = ComputeBleachingKLMS(lms_pre_bleach, fBleaching);
            debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(k));
        }
        else if (iDebugMode == 4)
        {
            float3 lms_dbg = mul(to_LMS, color);
            float lm_sum = lms_dbg.r + lms_dbg.g;

            if (lm_sum > 0.0)
            {
                float3 mb_dbg = LMS_to_MB(lms_dbg);
                float2 chroma_offset = mb_dbg.xy - mb_white;
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
                float3 mb_dbg = LMS_to_MB(lms_dbg);
                float2 chroma_offset = mb_dbg.xy - mb_white;
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
                debug_out = (any_neg > 0.0) ? neg : float3(0.0, 0.15, 0.0);
            }
        }
        else if (iDebugMode == 8)
        {
            if (iToneMapperMode == 1)
            {
                float3 normalized = pre_khronos_color / max(whitePt, FLT_MIN);
                float ratio = ComputeCompressionRatio(normalized, targetPeak, fCompressionStart);
                debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(ratio));
            }
            else if (iToneMapperMode == 2)
            {
                debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(tone_comp_ratio));
            }
            else
            {
                debug_out = float3(0.2, 0.2, 0.2);
            }
        }
        else if (iDebugMode == 9)
        {
            float3 lms_dbg = mul(to_LMS, color);
            float lm_sum = lms_dbg.r + lms_dbg.g;
            if (lm_sum > FLT_MIN)
            {
                float3 mb_dbg = LMS_to_MB(lms_dbg);
                float skin_mask = Evaluate3DSkinLocusMB(dot(lms_dbg, input.luma_LMS_coeffs) / max(whitePt, FLT_MIN), mb_dbg);
                debug_out = lerp(float3(0.0, 0.1, 0.3), float3(1.0, 0.2, 0.8), skin_mask);
            }
        }

        fragColor = float4(EncodeDebug(debug_out, space), src.a);
        return;
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 8: ENCODE & OUTPUT
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

technique PhotorealHDR_Mastering_V616 <
    ui_label = "Photoreal HDR V6.1.6 (Consistent-Normalization Edition)";
    ui_tooltip = "Reference-grade offline color grading executing the exact mathematics of non-Riemannian color space.\n\n"
                 "V6.1.6 Audited Changes:\n"
                 "  - FIX: Chroma reliability gate now white-relative (was absolute nits / inert).\n"
                 "  - FIX: Factory-default settings are a true bit-transparent bypass (knee = 0).\n"
                 "  - FIX: Vibrance normalization identical in clamped and unclamped gamut modes.\n"
                 "  - FIX: Master Saturation = 0 forces exact monochrome, overriding Vibrance.\n"
                 "  - FIX: Negative-luma scRGB WCG pixels bypass dehaze/contrast untouched.\n"
                 "  - CLEANUP: Single-source Khronos shoulder math (TM + debug share one formula).\n\n"
                 "Carried from V6.1.5:\n"
                 "  - Static D65 anchor (0.5, 0.5); zero hue bending across luminance gradients.\n"
                 "  - Gamut-Relative Vibrance: equal pastel response across blues/greens/reds.\n"
                 "  - Zero hue-distortion radial purity scaling across the full color wheel.\n"
                 "  - GAMUT GUARD TARGET: Auto, Rec.709, DCI-P3, Rec.2020, Unclamped.\n"
                 "  - 3D SKIN PROTECTION: Melanin-Hemoglobin locus, Fitzpatrick I-VI.\n"
                 "  - Closed-form Delta-Y lock (~1 ulp residual) across all color operations.\n\n"
                 "Companion shader: Bilateral Contrast";
>
{
    pass
    {
        VertexShader      = VS_PhotorealHDR;
        PixelShader       = PS_PhotorealHDR;
        VertexCount       = 3;
        PrimitiveTopology = TRIANGLELIST;
    }
}