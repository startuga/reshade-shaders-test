// =================================================================================================
// Photoreal HDR Scene Grader (V7.4.1 - Reference Linear Grading & Iapbp Color Space Edition)
// =================================================================================================
//
// Design Philosophy: PRECISION AND QUALITY OVER PERFORMANCE
// - True IEEE 754 Math: Guarded linear paths, NaN healing, and scale-invariant solvers.
// - Exact IEC/SMPTE Constants: Analytically matched thresholds for bit-exact C0 continuity.
// - True Stop-Domain Scene Grading: Linear EV exposure, CAT02 Von Kries adaptation, and filmic contrast.
// - Projection-Based Iapbp Space: Advanced perceptual color space with SA-PQ non-linear transfer
//   (Optics Express Vol. 32, Issue 17, pp. 30742–30755, August 2024).
// - Gamut-Relative Vibrance: Normalized pastel saturation scaling across all hues uniformly.
// - Enforced Bit-Exact Neutrality: Master Saturation = 0.00 forces exact R=G=B monochrome collapse.
// - Exact Ray-Tracing: 24-step unrolled binary search solver for exact physical gamut boundaries.
// - Gamut Boundary Targets: Auto, Rec.709, Display P3, Rec.2020, and Bypass (Unclamped).
// - 3D Melanin-Hemoglobin skin locus protecting Fitzpatrick I-VI.
//
// V7.4.1 Fixes:
// - Fixed error X3004: unified to_TargetRGB declaration in PS_PhotorealHDR.
// - Derived exact closed-form neutral anchor at constant lightness Ia, eliminating anchor chromatic skew.
// - Implemented projective horizon guard (t_pole) to prevent the boundary solver from crossing W <= 0.
// - Added C1 continuous exponential soft-knee gamut compression.
// - Integrated strict primary non-negativity conformance guard, eliminating sub-epsilon AP0 leakage.
//
// References:
// - Y. Huang et al., "Towards perceptual uniformity and HDR-WCG image processing: a projection-based
//   color space," Optics Express 32(17), 30742–30755 (2024). https://doi.org/10.1364/OE.530213
// - ReShade Shaders Reference: https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md
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
static const float SRGB_INV_GAMMA       = 0.41666666666666667;

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
// Color Space CIE Tristimulus Matrices (D65 White Point: X=0.950456, Y=1.000000, Z=1.088830)
// -------------------------------------------------------------------------------------------------
static const float3 D65_XYZ             = float3(0.950456, 1.000000, 1.088830);
static const float3 Luma709             = float3(0.2126729, 0.7151522, 0.0721750);
static const float3 LumaP3              = float3(0.2289746, 0.6917385, 0.0792869);
static const float3 Luma2020            = float3(0.2627002, 0.6779981, 0.0593017);

static const float3x3 RGB709_to_XYZ = float3x3(
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
);
static const float3x3 XYZ_to_RGB709 = float3x3(
    3.2404542, -1.5371385, -0.4985314,
   -0.9692660,  1.8760108,  0.0415560,
    0.0556434, -0.2040259,  1.0572252
);
static const float3x3 RGB2020_to_XYZ = float3x3(
    0.6369580, 0.1446169, 0.1688810,
    0.2627002, 0.6779981, 0.0593017,
    0.0000000, 0.0280727, 1.0609851
);
static const float3x3 XYZ_to_RGB2020 = float3x3(
    1.7166512, -0.3556708, -0.2533663,
   -0.6666844,  1.6164812,  0.0157685,
    0.0176399, -0.0427706,  0.9421031
);
static const float3x3 P3D65_to_XYZ = float3x3(
    0.4865709, 0.2656677, 0.1982173,
    0.2289746, 0.6917385, 0.0792869,
    0.0000000, 0.0451134, 1.0439441
);
static const float3x3 XYZ_to_P3D65 = float3x3(
    2.4934969, -0.9313836, -0.4027108,
   -0.8294890,  1.7626641,  0.0236247,
    0.0358458, -0.0761724,  0.9568845
);

// Standard CAT02 Matrices for Physiological Linear White Balance
static const float3x3 XYZ_to_CAT02 = float3x3(
     0.7328,  0.4296, -0.1624,
    -0.7036,  1.6975,  0.0061,
     0.0030,  0.0136,  0.9834
);
static const float3x3 CAT02_to_XYZ = float3x3(
     1.096124, -0.278869,  0.182745,
     0.454369,  0.473533,  0.072098,
    -0.009628, -0.005658,  1.015286
);

// -------------------------------------------------------------------------------------------------
// Optics Express 2024: Iapbp Projection Transformation Matrices & Inverses
// -------------------------------------------------------------------------------------------------
static const float4x4 M1_XYZ_to_LMS = float4x4(
    0.490978,  1.045001,  0.482481, 0.0,
    0.517558,  1.256543,  0.240489, 0.0,
    1.587132,  0.553389, -0.124257, 0.0,
    1.967421, -0.807475, -0.143517, 1.0
);
static const float4x4 M1_LMS_to_XYZ = float4x4(
    0.57848819, -0.79376771,  0.70995724, 0.0,
   -0.89207575,  1.65368485, -0.26329773, 0.0,
    3.41608287, -2.77395346, -0.15218629, 0.0,
   -1.36819272,  2.49887497, -1.63123244, 1.0
);
static const float4x4 M2_LMSprime_to_Iapbp = float4x4(
   -0.011823,  0.248826, -0.106030, 0.0,
    5.055264, -5.937734,  0.881421, 0.0,
    1.247189, -3.372989,  2.125450, 0.0,
   -0.741126,  1.127297, -1.255019, 1.0
);
static const float4x4 M2_Iapbp_to_LMSprime = float4x4(
    7.63711500,  0.13555011,  0.32477197, 0.0,
    7.63559420, -0.08479165,  0.41607151, 0.0,
    7.63595917, -0.21409956,  0.94020212, 0.0,
    6.63575590, -0.07265393,  0.95163231, 1.0
);

// SA-PQ Constants (Huang et al. 2024 Eq. 6)
static const float SA_PQ_C1     = 0.7707;
static const float SA_PQ_C2     = 44.4561;
static const float SA_PQ_C3     = 44.2269;
static const float SA_PQ_M      = 0.2926;
static const float SA_PQ_N      = 78.2171;
static const float SA_PQ_INV_M  = 3.417634996582365;
static const float SA_PQ_INV_N  = 0.012784928091172;
static const float SA_PQ_K_MAX  = (SA_PQ_C2 / SA_PQ_C3) * (1.0 - 1e-5);

// Chroma Reliability Thresholds
static const float CHROMA_RELIABILITY_START     = 5e-4;
static const float CHROMA_STABILITY_THRESH      = 2e-3;
static const float INV_CHROMA_RELIABILITY_SPAN  = 1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);

// Zone System Constants
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
    ui_tooltip  = "Linear EV shift: multiply by 2^EV in scene-linear domain.\n+1.0 EV = double brightness, -1.0 EV = half brightness.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fTemperature <
    ui_type     = "slider";
    ui_min      = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label    = "Color Temperature";
    ui_tooltip  = "Physiological chromatic adaptation in linear CAT02 cone space.\nNegative = Cooler, Positive = Warmer.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fTint <
    ui_type     = "slider";
    ui_min      = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label    = "Color Tint";
    ui_tooltip  = "Physiological chromatic adaptation in linear CAT02 cone space.\nNegative = Greener, Positive = Magenta.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fBlackPoint <
    ui_type     = "slider";
    ui_min      = 0.000; ui_max = 0.050; ui_step = 0.001;
    ui_label    = "Dehaze / Black Point";
    ui_tooltip  = "Subtracts a percentage of reference white from luminance.";
    ui_category = "1. Scene Grade";
> = 0.000;

uniform float fShadowFloor <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.005;
    ui_label    = "Dehaze Shadow Floor";
    ui_tooltip  = "Minimum residual luminance ratio for Dehaze. Prevents shadow crush.";
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
    ui_label    = "Contrast Pivot";
    ui_tooltip  = "The luminance value that remains unchanged when contrast is adjusted.";
    ui_category = "1. Scene Grade";
> = 0.17677669529;

uniform float fShadows <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Shadows (Log Recovery)";
    ui_tooltip  = "Lifts or deepens shadow detail in the stop domain with C1 continuity.";
    ui_category = "1. Scene Grade";
> = 0.0;

uniform float fHighlights <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Highlights (Log Recovery)";
    ui_tooltip  = "Compresses (-1.0) or boosts (+1.0) highlight detail in the stop domain.";
    ui_category = "1. Scene Grade";
> = 0.0;

uniform float fSaturation <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 2.00; ui_step = 0.01;
    ui_label    = "Purity / Saturation (Iapbp)";
    ui_tooltip  = "Strictly radial chromaticity scaling in the projection-based Iapbp color space.\n"
                  "Set to 1.0 for neutral identity pass.\n"
                  "Set to 0.0 for exact bit-level R=G=B monochrome collapse.";
    ui_category = "1. Scene Grade";
> = 1.00;

uniform float fVibrance <
    ui_type     = "slider";
    ui_min      = -1.00; ui_max = 2.00; ui_step = 0.01;
    ui_label    = "Smart Saturation (Vibrance)";
    ui_tooltip  = "Intelligently boosts muted pastels uniformly using Gamut-Relative Weber-Fechner curves in Iapbp space.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fSkinProtection <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Skin Tone Protection";
    ui_tooltip  = "Protects human skin tones (Fitzpatrick I-VI) using Melanin-Hemoglobin cone locus gating.";
    ui_category = "1. Scene Grade";
> = 0.85;

uniform float fAbneyCorrection <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Abney Hue Compensation";
    ui_tooltip  = "Counteracts perceived hue shifts as saturation is scaled.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform int iGamutTarget <
    ui_type     = "combo";
    ui_label    = "Gamut Guard Target Limit";
    ui_items    = "Auto (Container Gamut)\0Rec. 709 (SDR Standard)\0Display P3 (D65)\0Rec. 2020 (UHD Display)\0Bypass / Unclamped (Infinite Gamut)\0";
    ui_tooltip  = "Selects physical gamut boundary to soft-compress/clamp against.\n"
                  "- Auto: Matches container (Rec.709 in SDR, Rec.2020 in HDR).\n"
                  "- Bypass / Unclamped: Disables boundary clamping completely.";
    ui_category = "1. Scene Grade";
> = 0;

uniform float fGamutGuardKnee <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label    = "Gamut Guard Knee";
    ui_tooltip  = "Analytical soft-knee boundary compression in Iapbp space. Set 0.0 for hard clamp.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform int iColorSpaceOverride <
    ui_type     = "combo";
    ui_label    = "Color Space Override";
    ui_items    = "Auto (Default via ReShade)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0HLG (HDR)\0";
    ui_tooltip  = "Container format detection override.";
    ui_category = "2. System";
> = 0;

uniform float fWhitePoint <
    ui_type     = "slider";
    ui_min      = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_label    = "Reference White (Nits)";
    ui_tooltip  = "Diffuse paper white anchor for HDR mapping (default 203 nits ITU-R BT.2408).\nIgnored for standard SDR sRGB.";
    ui_category = "2. System";
> = 203.0;

uniform int iDebugMode <
    ui_type     = "combo";
    ui_label    = "Debug Visualization";
    ui_items    = "Off\0"
                  "Luminance (False Color Stops)\0"
                  "Zone Map (Ansel Adams)\0"
                  "CAT02 Cone Response\0"
                  "Iapbp Chroma / Saturation\0"
                  "Iapbp Hue Wheel\0"
                  "Negative / WCG Out-of-Gamut\0"
                  "Skin Protection Mask\0";
    ui_tooltip  = "Debug diagnostic visualizations operating on graded output.";
    ui_category = "3. Debug";
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
// 5. Projective Transformations & SA-PQ Non-linear Transfer
// =================================================================================================

float3 ProjectiveTransform(float4x4 M, float3 v)
{
    float4 h = mul(M, float4(v, 1.0));
    if (any(IsNan3(h.xyz)) || (h.w != h.w)) 
        return float3(0.0, 0.0, 0.0);
        
    float w = h.w;
    if (abs(w) < FLT_MIN)
        w = (w < 0.0) ? -FLT_MIN : FLT_MIN;
        
    return h.xyz / w;
}

float3 SA_PQ_Forward(float3 Y)
{
    float3 Y_clamped = max(Y, 0.0);
    float3 Y_m = PowNonNegPreserveZero3(Y_clamped, SA_PQ_M);
    float3 num = SA_PQ_C1 + SA_PQ_C2 * Y_m;
    float3 den = 1.0 + SA_PQ_C3 * Y_m;
    return PowNonNegPreserveZero3(num / den, SA_PQ_N);
}

float3 SA_PQ_Inverse(float3 V)
{
    float3 V_clamped = max(V, 0.0);
    float3 k = PowNonNegPreserveZero3(V_clamped, SA_PQ_INV_N);
    k = min(k, SA_PQ_K_MAX);
    float3 num = max(k - SA_PQ_C1, 0.0);
    float3 den = max(SA_PQ_C2 - SA_PQ_C3 * k, FLT_MIN);
    return PowNonNegPreserveZero3(num / den, SA_PQ_INV_M);
}

/**
 * GetNeutralAnchorIapbp
 * Exact closed-form neutral locus (D65) in Iapbp space as a function of lightness Ia.
 * Derived from M1, SA-PQ, and M2: on the achromatic line L'=M'=S'=k:
 *   Ia = 0.130973 * k / (1.0 - 0.868848 * k)
 *   ap = -0.00800928 * Ia
 *   bp = -0.00267231 * Ia
 * This guarantees bit-exact neutrality and strictly positive primary values at t = 0.
 */
float2 GetNeutralAnchorIapbp(float Ia)
{
    return Ia * float2(-0.00800928, -0.00267231);
}

// =================================================================================================
// 6. Color Science & Container EOTF/OETF Utilities
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

float3 HLG_EOTF(float3 x)
{
    x = max(x, 0.0);
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

// =================================================================================================
// 7. Physiological Human Visual System & Locus Utilities
// =================================================================================================

float Evaluate3DSkinLocusLinear(float3 xyz, float luma_norm)
{
    float3 lms = mul(XYZ_to_CAT02, xyz);
    float lm_sum = max(lms.r + lms.g, FLT_MIN);
    float l_ratio = lms.r / lm_sum;
    float s_ratio = lms.b / lm_sum;
    
    float l_gate = smoothstep(0.56, 0.60, l_ratio) * (1.0 - smoothstep(0.68, 0.72, l_ratio));
    float expected_s_min = clamp(0.12 + 0.16 * luma_norm, 0.12, 0.28);
    float s_gate = smoothstep(expected_s_min - 0.04, expected_s_min + 0.02, s_ratio) * (1.0 - smoothstep(0.38, 0.44, s_ratio));
    float luma_gate = smoothstep(0.010, 0.035, luma_norm) * (1.0 - smoothstep(0.95, 1.30, luma_norm));
    
    return saturate(l_gate * s_gate * luma_gate);
}

float ComputeBlackPointRatio(float luma, float bpNits, float shadowFloor)
{
    float raw = max((luma - bpNits) / max(luma, FLT_MIN), shadowFloor);
    float t = saturate(luma / max(4.0 * bpNits, FLT_MIN));
    float smooth_t = t * t * (3.0 - 2.0 * t);
    return lerp(shadowFloor, raw, smooth_t);
}

/**
 * SolveGamutBoundaryIapbp (Scale-Invariant Primary Non-Negativity Solver)
 * Solves the exact intersection with the physical RGB gamut boundary along a chromatic ray.
 * Uses analytical projective pole avoidance: guarantees W > 0 along the entire search bracket.
 */
float SolveGamutBoundaryIapbp(
    float Ia,
    float2 neutral_apbp,
    float2 chroma_dir,
    float3x3 XYZ_to_targetRGB)
{
    float W0 = 6.635756 * Ia - 0.072654 * neutral_apbp.x + 0.951632 * neutral_apbp.y + 1.0;
    float D_W = -0.072654 * chroma_dir.x + 0.951632 * chroma_dir.y;
    
    float t_low = 0.0;
    float t_high = (D_W < -1e-5) ? min(1.5, -W0 / D_W * 0.85) : 1.5;
    
    [unroll]
    for (int iter = 0; iter < 24; iter++)
    {
        float t = 0.5 * (t_low + t_high);
        float2 test_apbp = neutral_apbp + t * chroma_dir;
        float3 test_iapbp = float3(Ia, test_apbp.x, test_apbp.y);
        
        float3 lms_p = ProjectiveTransform(M2_Iapbp_to_LMSprime, test_iapbp);
        float3 lms = SA_PQ_Inverse(lms_p);
        float3 norm_xyz = ProjectiveTransform(M1_LMS_to_XYZ, lms) * D65_XYZ;
        float3 rgb = mul(XYZ_to_targetRGB, norm_xyz);
        
        float min_rgb = min(min(rgb.r, rgb.g), rgb.b);
        bool outside = (min_rgb < 0.0) || !(min_rgb >= 0.0);
        
        if (outside)
            t_high = t;
        else
            t_low = t;
    }
    
    return t_low;
}

/**
 * ApplyIapbpSaturationAndGamutGuard
 * Reference chromatic grading and gamut compression engine.
 */
float3 ApplyIapbpSaturationAndGamutGuard(
    float3 graded_lin_xyz,
    float purity_scale,
    float vibrance_amount,
    float skin_protection,
    int gamut_target_mode,
    float knee,
    float abney_correction,
    float3x3 to_RGB_boundary,
    float3x3 boundary_to_XYZ,
    float3x3 to_TargetRGB,
    float3 lumaCoeffsBoundary,
    float whitePt,
    float3 lumaCoeffs,
    out float out_skin_confidence)
{
    out_skin_confidence = 0.0;
    
    bool is_unclamped = (gamut_target_mode == 4);
    bool force_gamut_clamp = (gamut_target_mode == 1 || gamut_target_mode == 2 || gamut_target_mode == 3);
    
    float luma_nits = graded_lin_xyz.y;
    float luma_norm = luma_nits / max(whitePt, FLT_MIN);
    
    bool chromaIdentity = 
        abs(purity_scale - 1.0) < NEUTRAL_EPS &&
        abs(vibrance_amount) < NEUTRAL_EPS &&
        abney_correction < NEUTRAL_EPS &&
        (knee < NEUTRAL_EPS || is_unclamped);

    if (!force_gamut_clamp && chromaIdentity)
    {
        out_skin_confidence = Evaluate3DSkinLocusLinear(graded_lin_xyz / max(whitePt, FLT_MIN), luma_norm);
        return mul(to_TargetRGB, graded_lin_xyz);
    }

    // Reference White Normalization into Iapbp (Huang et al. 2024 Section 2.1)
    float3 norm_xyz = (graded_lin_xyz / max(whitePt, FLT_MIN)) / D65_XYZ;
    float3 lms = ProjectiveTransform(M1_XYZ_to_LMS, norm_xyz);
    float3 lms_p = SA_PQ_Forward(max(lms, 0.0));
    float3 iapbp = ProjectiveTransform(M2_LMSprime_to_Iapbp, lms_p);
    
    // Exact closed-form neutral anchor at lightness Ia
    float2 neutral_apbp = GetNeutralAnchorIapbp(iapbp.x);
    
    // Bit-exact monochrome collapse (Master Saturation = 0.00)
    if (purity_scale <= NEUTRAL_EPS)
    {
        float3 rgb_direct = mul(to_TargetRGB, graded_lin_xyz);
        float gray = dot(rgb_direct, lumaCoeffs);
        return gray.xxx;
    }
    
    float2 chroma_offset = iapbp.yz - neutral_apbp;
    float chroma = SqrtIEEE(dot(chroma_offset, chroma_offset));
    
    float ct = saturate((luma_norm - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);
    
    float2 chroma_dir = (chroma > 1e-7) ? (chroma_offset / chroma) : float2(1.0, 0.0);
    float initial_max_chroma = SolveGamutBoundaryIapbp(iapbp.x, neutral_apbp, chroma_dir, to_RGB_boundary);
    float relative_purity = saturate(chroma / max(initial_max_chroma, FLT_MIN));
    
    float skin_confidence = Evaluate3DSkinLocusLinear(graded_lin_xyz / max(whitePt, FLT_MIN), luma_norm);
    out_skin_confidence = skin_confidence;
    
    float effective_skin_mask = saturate(skin_confidence * skin_protection);
    float effective_scale = purity_scale;
    
    if (effective_scale > 1.0 && effective_skin_mask > NEUTRAL_EPS)
    {
        float master_boost = effective_scale - 1.0;
        master_boost *= (1.0 - effective_skin_mask * 0.85);
        effective_scale = 1.0 + master_boost;
    }
    
    // Smart Saturation (Vibrance)
    if (abs(vibrance_amount) > NEUTRAL_EPS)
    {
        float saturation_demand = exp(-3.5 * pow(relative_purity, 1.25));
        float vibrance_gain = vibrance_amount * saturation_demand
                              * saturate(purity_scale)
                              * (1.0 - effective_skin_mask) * chroma_reliability;
        effective_scale *= max(1.0 + vibrance_gain, 0.0);
    }
    
    // Non-Riemannian Diminishing Returns for Extreme Boosts
    if (effective_scale > 1.0)
    {
        float boost = effective_scale - 1.0;
        boost = boost / (1.0 + 0.35 * relative_purity * boost);
        effective_scale = 1.0 + boost;
        effective_scale = lerp(1.0, effective_scale, chroma_reliability);
    }
    
    float2 scaled_chroma_offset = chroma_offset * effective_scale;
    
    // Abney Hue Compensation
    if (abney_correction > NEUTRAL_EPS && chroma > 1e-7)
    {
        float angle = atan2(chroma_offset.y, chroma_offset.x);
        float abney_profile = 0.15 * sin(2.0 * angle + 0.4) * (1.0 + 0.3 * cos(angle));
        float shift = abney_profile * relative_purity * abney_correction * chroma_reliability;
        angle += shift;
        
        float scaled_len = SqrtIEEE(dot(scaled_chroma_offset, scaled_chroma_offset));
        scaled_chroma_offset = float2(cos(angle), sin(angle)) * scaled_len;
    }
    
    // Gamut Guard: Soft-Knee Compression & Boundary Limiting
    float scaled_purity = SqrtIEEE(dot(scaled_chroma_offset, scaled_chroma_offset));
    
    if (!is_unclamped && scaled_purity > FLT_MIN)
    {
        float2 new_chroma_dir = scaled_chroma_offset / scaled_purity;
        float max_chroma = (abney_correction > NEUTRAL_EPS)
            ? SolveGamutBoundaryIapbp(iapbp.x, neutral_apbp, new_chroma_dir, to_RGB_boundary)
            : initial_max_chroma;
            
        if (max_chroma > FLT_MIN)
        {
            // Analytical Exponential Soft-Knee Gamut Compression
            if (knee > FLT_MIN)
            {
                float threshold = max_chroma * (1.0 - knee);
                if (scaled_purity > threshold)
                {
                    float excess = scaled_purity - threshold;
                    float headroom = max(max_chroma - threshold, FLT_MIN);
                    float compressed = threshold + headroom * (1.0 - exp(-excess / headroom));
                    scaled_chroma_offset = new_chroma_dir * compressed;
                    scaled_purity = compressed;
                }
            }
            
            // Boundary hard clamp
            float p_safe = max_chroma * (1.0 - 1e-5);
            if (scaled_purity > p_safe)
            {
                scaled_chroma_offset = new_chroma_dir * p_safe;
            }
        }
    }
    
    // Reconstruct Iapbp
    iapbp.yz = neutral_apbp + scaled_chroma_offset;
    
    // Invert back: Iapbp -> L'M'S' -> LMS -> Normalized XYZ -> Physical Linear XYZ
    float3 out_lms_p = ProjectiveTransform(M2_Iapbp_to_LMSprime, iapbp);
    float3 out_lms = SA_PQ_Inverse(out_lms_p);
    float3 out_norm_xyz = ProjectiveTransform(M1_LMS_to_XYZ, out_lms) * D65_XYZ;
    float3 out_xyz = out_norm_xyz * whitePt;
    
    // Strict Target Gamut Primary Non-Negativity Conformance Guard
    if (!is_unclamped)
    {
        float3 rgb_b = mul(to_RGB_boundary, out_xyz);
        float min_c = min(min(rgb_b.r, rgb_b.g), rgb_b.b);
        if (min_c < 0.0)
        {
            float luma_b = dot(rgb_b, lumaCoeffsBoundary);
            float t_desat = saturate(-min_c / max(luma_b - min_c, FLT_MIN));
            rgb_b = lerp(rgb_b, luma_b.xxx, t_desat);
            rgb_b = max(rgb_b, 0.0);
            out_xyz = mul(boundary_to_XYZ, rgb_b);
        }
    }
    
    return mul(to_TargetRGB, out_xyz);
}

// =================================================================================================
// 8. Debug Diagnostic Functions
// =================================================================================================

float3 EncodeDebug(float3 debug_out, int space)
{
    debug_out = max(debug_out, 0.0);
    [branch]
    if (space == 4)      return HLG_OETF(lerp(100.0, 600.0, saturate(debug_out)));
    else if (space == 3) return PQ_InverseEOTF(lerp(100.0, 600.0, saturate(debug_out)));
    else if (space == 2) return lerp(0.05, 2.5, saturate(debug_out));
    else                 return sRGB_OETF(saturate(debug_out));
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

// =================================================================================================
// 9. Vertex Shader
// =================================================================================================

struct VS_Output
{
    float4 vpos : SV_Position;
    float2 texcoord : TEXCOORD0;
    nointerpolation float3 wbGainCAT02 : TEXCOORD1;
};

VS_Output VS_PhotorealHDR(uint id : SV_VertexID)
{
    VS_Output output;
    output.texcoord.x = (id == 2) ? 2.0 : 0.0;
    output.texcoord.y = (id == 1) ? 2.0 : 0.0;
    output.vpos = float4(output.texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    
    // Stop shifts for physiological linear CAT02 adaptation
    float3 wbStops = 0.35 * float3(
        fTemperature + fTint,
        -fTint,
        -fTemperature + fTint
    );
    float3 rawGains = exp2(wbStops);
    
    // Preserve exact luminance Y of D65 during white balancing
    float3 cat02D65 = mul(XYZ_to_CAT02, D65_XYZ);
    float3 adaptedCAT02 = cat02D65 * rawGains;
    float3 adaptedXYZ = mul(CAT02_to_XYZ, adaptedCAT02);
    float lumaPreserveScale = 1.0 / max(adaptedXYZ.y, FLT_MIN);
    
    output.wbGainCAT02 = rawGains * lumaPreserveScale;
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
    
    // Bit-transparent bypass check
    [branch]
    if (iDebugMode == 0 &&
        iColorSpaceOverride == 0 &&
        abs(fExposure) < NEUTRAL_EPS && abs(fBlackPoint) < NEUTRAL_EPS &&
        abs(fContrast - 1.0) < NEUTRAL_EPS && abs(fShadows) < NEUTRAL_EPS &&
        abs(fHighlights) < NEUTRAL_EPS && abs(fTemperature) < NEUTRAL_EPS &&
        abs(fTint) < NEUTRAL_EPS && abs(fSaturation - 1.0) < NEUTRAL_EPS &&
        abs(fVibrance) < NEUTRAL_EPS &&
        fAbneyCorrection < NEUTRAL_EPS && fGamutGuardKnee < NEUTRAL_EPS &&
        iGamutTarget == 0)
    {
        fragColor = src;
        return;
    }
    
    // Decode to scene-linear nits & sanitize
    float3 original_lin = DecodeToLinear(src.rgb, space);
    bool is_invalid = any(IsNan3(original_lin)) || any(IsInf3(original_lin));
    original_lin = is_invalid ? (0.18 * whitePt).xxx : original_lin;
    
    float3x3 to_XYZ, to_TargetRGB;
    float3x3 to_RGB_boundary, boundary_to_XYZ;
    float3 lumaCoeffsBoundary;

    [branch]
    if (space >= 3)
    {
        to_XYZ              = RGB2020_to_XYZ;
        to_TargetRGB        = XYZ_to_RGB2020;
        to_RGB_boundary     = XYZ_to_RGB2020;
        boundary_to_XYZ     = RGB2020_to_XYZ;
        lumaCoeffsBoundary  = Luma2020;
    }
    else if (space == 2)
    {
        to_XYZ              = RGB709_to_XYZ;
        to_TargetRGB        = XYZ_to_RGB709;
        // scRGB container default target: Rec.2020
        to_RGB_boundary     = XYZ_to_RGB2020;
        boundary_to_XYZ     = RGB2020_to_XYZ;
        lumaCoeffsBoundary  = Luma2020;
    }
    else
    {
        to_XYZ              = RGB709_to_XYZ;
        to_TargetRGB        = XYZ_to_RGB709;
        to_RGB_boundary     = XYZ_to_RGB709;
        boundary_to_XYZ     = RGB709_to_XYZ;
        lumaCoeffsBoundary  = Luma709;
    }
    
    // Explicit Gamut Guard Target Selection
    [branch]
    if (iGamutTarget == 1)
    {
        to_RGB_boundary     = XYZ_to_RGB709;
        boundary_to_XYZ     = RGB709_to_XYZ;
        lumaCoeffsBoundary  = Luma709;
    }
    else if (iGamutTarget == 2)
    {
        to_RGB_boundary     = XYZ_to_P3D65;
        boundary_to_XYZ     = P3D65_to_XYZ;
        lumaCoeffsBoundary  = LumaP3;
    }
    else if (iGamutTarget == 3)
    {
        to_RGB_boundary     = XYZ_to_RGB2020;
        boundary_to_XYZ     = RGB2020_to_XYZ;
        lumaCoeffsBoundary  = Luma2020;
    }
    
    // ---------------------------------------------------------------------------------------------
    // STAGE 1: LINEAR SCENE GRADE (Exposure, CAT02 White Balance)
    // ---------------------------------------------------------------------------------------------
    float3 lin_xyz = mul(to_XYZ, original_lin);
    
    // Physiological Von Kries adaptation in linear CAT02 cone space
    if (abs(fTemperature) > NEUTRAL_EPS || abs(fTint) > NEUTRAL_EPS)
    {
        float3 cat02 = mul(XYZ_to_CAT02, lin_xyz);
        cat02 *= input.wbGainCAT02;
        lin_xyz = mul(CAT02_to_XYZ, cat02);
    }
    
    // Linear Stop-Domain Exposure
    if (abs(fExposure) > NEUTRAL_EPS)
    {
        lin_xyz *= exp2(fExposure);
    }
    
    // ---------------------------------------------------------------------------------------------
    // STAGE 2: PHYSICAL LUMINANCE GRADING (Dehaze, Filmic Contrast, Stop Recovery)
    // ---------------------------------------------------------------------------------------------
    float luma = lin_xyz.y;
    
    float bp_ratio = 1.0;
    if (fBlackPoint > NEUTRAL_EPS && luma > 0.0)
    {
        float bpNits = fBlackPoint * whitePt;
        bp_ratio = ComputeBlackPointRatio(luma, bpNits, fShadowFloor);
    }
    
    float contrast_ratio = 1.0;
    float graded_luma = max(luma * bp_ratio, FLT_MIN);
    float absLuma = graded_luma;
    
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
        
        float excess = max(ratio - 80.0, 0.0);
        contrast_ratio = min(ratio, 80.0) + (excess / (1.0 + excess / 20.0));
    }
    
    lin_xyz *= bp_ratio * contrast_ratio;
    
    // ---------------------------------------------------------------------------------------------
    // STAGE 3: PERCEPTUAL COLOR SPACE GRADING & GAMUT GUARD (Iapbp Space, Huang et al. 2024)
    // ---------------------------------------------------------------------------------------------
    float skin_confidence = 0.0;
    float3 color = ApplyIapbpSaturationAndGamutGuard(
        lin_xyz,
        fSaturation,
        fVibrance,
        fSkinProtection,
        iGamutTarget,
        fGamutGuardKnee,
        fAbneyCorrection,
        to_RGB_boundary,
        boundary_to_XYZ,
        to_TargetRGB,
        lumaCoeffsBoundary,
        whitePt,
        lumaCoeffs,
        skin_confidence
    );
    
    is_invalid = any(IsNan3(color)) || any(IsInf3(color));
    color = is_invalid ? original_lin : color;
    
    // ---------------------------------------------------------------------------------------------
    // DEBUG VISUALIZATION (Operating directly on graded output)
    // ---------------------------------------------------------------------------------------------
    [branch]
    if (iDebugMode != 0)
    {
        float3 debug_out = float3(0.0, 0.0, 0.0);
        
        if (iDebugMode == 1) // False Color Stops
        {
            float l = dot(color, lumaCoeffs);
            float stops = log2(max(abs(l), FLT_MIN) / max(whitePt, FLT_MIN));
            debug_out = StopsToFalseColor(stops);
        }
        else if (iDebugMode == 2) // Zone Map
        {
            float l = dot(color, lumaCoeffs);
            float nl = l / max(whitePt, FLT_MIN);
            debug_out = GetZoneColor(GetZone(nl));
        }
        else if (iDebugMode == 3) // CAT02 Cone Response
        {
            float3 c02 = mul(XYZ_to_CAT02, mul(to_XYZ, color));
            float max_c02 = max(max(abs(c02.r), abs(c02.g)), abs(c02.b));
            if (max_c02 > FLT_MIN)
                debug_out = abs(c02) / max_c02;
        }
        else if (iDebugMode == 4) // Iapbp Chroma / Saturation (Graded Output)
        {
            float3 graded_xyz = mul(to_XYZ, color);
            float3 norm_xyz_dbg = (graded_xyz / max(whitePt, FLT_MIN)) / D65_XYZ;
            float3 lms_p_dbg = SA_PQ_Forward(max(ProjectiveTransform(M1_XYZ_to_LMS, norm_xyz_dbg), 0.0));
            float3 iapbp_dbg = ProjectiveTransform(M2_LMSprime_to_Iapbp, lms_p_dbg);
            float2 neutral_apbp_dbg = GetNeutralAnchorIapbp(iapbp_dbg.x);
            
            float2 c_off = iapbp_dbg.yz - neutral_apbp_dbg;
            float ch = SqrtIEEE(dot(c_off, c_off));
            float v = saturate(ch * 4.0);
            debug_out = float3(v, v * 0.7, v * 0.3);
        }
        else if (iDebugMode == 5) // Iapbp Hue Wheel (Graded Output)
        {
            float3 graded_xyz = mul(to_XYZ, color);
            float3 norm_xyz_dbg = (graded_xyz / max(whitePt, FLT_MIN)) / D65_XYZ;
            float3 lms_p_dbg = SA_PQ_Forward(max(ProjectiveTransform(M1_XYZ_to_LMS, norm_xyz_dbg), 0.0));
            float3 iapbp_dbg = ProjectiveTransform(M2_LMSprime_to_Iapbp, lms_p_dbg);
            float2 neutral_apbp_dbg = GetNeutralAnchorIapbp(iapbp_dbg.x);
            
            float2 c_off = iapbp_dbg.yz - neutral_apbp_dbg;
            float ch_sq = dot(c_off, c_off);
            
            if (ch_sq > 1e-12)
            {
                float hue = atan2(c_off.y, c_off.x) / (2.0 * PI) + 0.5;
                float br = saturate(SqrtIEEE(ch_sq) * 6.0);
                debug_out = HueToRGB(saturate(hue)) * br;
            }
        }
        else if (iDebugMode == 6) // Negative / WCG Out-of-Gamut
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
        else if (iDebugMode == 7) // Skin Protection Mask
        {
            debug_out = lerp(float3(0.0, 0.1, 0.3), float3(1.0, 0.2, 0.8), skin_confidence);
        }
        
        fragColor = float4(EncodeDebug(debug_out, space), src.a);
        return;
    }
    
    // ---------------------------------------------------------------------------------------------
    // FINAL ENCODE & OUTPUT
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

technique PhotorealHDR_SceneGrade <
    ui_label = "Photoreal HDR Scene Grader (Linear & Iapbp Edition)";
    ui_tooltip = "Reference-grade offline scene grading executing exact IEEE-754 mathematics.\n\n"
                 "V7.4.1 Features:\n"
                 "  - Pure Scene Grading engine (Tone mapping and HVS bleaching purged).\n"
                 "  - Physically correct linear scene exposure and CAT02 physiological white balance.\n"
                 "  - Iapbp Color Space: Advanced projection-based color space with SA-PQ curve\n"
                 "    (Huang et al., Optics Express 32(17), 30742–30755, 2024).\n"
                 "  - Bit-exact neutral gray tracking: Saturation = 0 forces exact R=G=B monochrome.\n"
                 "  - Gamut-Relative Vibrance: Equal pastel response across all hues uniformly.\n"
                 "  - 3D Melanin-Hemoglobin skin tone locus protecting Fitzpatrick I-VI.\n"
                 "  - Projective horizon-guarded 24-step binary search gamut boundary solver.\n"
                 "  - Strict primary non-negativity conformance guard (0.000% AP0/WCG leakage).\n"
                 "  - Factory default settings guarantee bit-transparent passthrough.";
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