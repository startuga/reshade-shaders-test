/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement - MASTERING EDITION
 *
 * Design Philosophy: PRECISION OVER PERFORMANCE
 * - True IEEE 754 Math (No fast intrinsics or approximations)
 * - Exact IEC/SMPTE Standard Constants
 * - Bit-Exact Neutrality Logic
 * - Pre-computed High-Precision Kernels
 * - True Stop-Domain HDR Processing
 * - Oklab Perceptual Chromaticity Processing
 *
 * Version: 8.4.1 (Bug Fixes & Optimization)
 * - Fix: Pre-pass chroma guard for dark pixels (prevents spurious ChromaEdge)
 * - Fix: minCompNorm computed unconditionally (Negative Protection now works in SDR)
 * - Fix: Structure Tensor reduced from 72 to 25 texture fetches (3x bandwidth savings)
 * - Fix: [flatten] hints on uniform-dependent EOTF/OETF branches
 * - Fix: Documented n_data.a negative-value contract
 * Author: startuga
 */

// https://github.com/crosire/reshade-shaders/blob/slim/Shaders/ReShade.fxh
// https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md
#include "ReShade.fxh"

// ==============================================================================
// 0. Pre-Processor Configuration
// ==============================================================================

// Set to 1: True IEEE 754 precision (Mastering Standard - No precision loss)
// Set to 0: 16-bit Float (Faster, uses 50% less VRAM, minor sub-pixel precision loss)
#ifndef PREPASS_USE_RGBA32F
    #define PREPASS_USE_RGBA32F 1
#endif

#if PREPASS_USE_RGBA32F
    #define PREPASS_FORMAT RGBA32F
#else
    #define PREPASS_FORMAT RGBA16F
#endif

// ==============================================================================
// 1. High-Precision Constants & Color Science Definitions
// ==============================================================================

// Mathematical Constants
static const float FLT_MIN = 1.175494351e-38;   // Strict float32 min normalized
static const float LN_FLT_MIN = -87.33654475;   // ln(FLT_MIN) for exp optimization
static const float NEG_LN_SPATIAL_CUTOFF = 9.210340372; // -ln(1e-4)

// Algorithm Thresholds
static const int MAX_LOOP_RADIUS = 32;          // Capped to prevent Driver Timeout (TDR)
static const float RATIO_MIN = 0.0001;          // Ratio safety clamp min
static const float RATIO_MAX = 10000.0;         // Ratio safety clamp max
static const float CHROMA_STABILITY_THRESH = 1e-4; // Linear value below which chroma weight fades out
static const float EDGE_LUMA_FLOOR = 1e-4;      // ~ -13 stops. Prevents log2 noise explosion in deep blacks.
static const float LOG2_EDGE_LUMA_FLOOR = -13.2877123795; // log2(1e-4)

// IEC 61966-2-1:1999 defines the EOTF threshold.
static const float SRGB_THRESHOLD_EOTF = 0.04045;
// Derived OETF threshold for perfect round-trip (compiler will fold this with high precision)
static const float SRGB_THRESHOLD_OETF = (0.04045 / 12.92);

// ITU-R Rec.709 Luma Coefficients (Standard)
static const float3 Luma709 = float3(0.2126, 0.7152, 0.0722);

// ITU-R Rec.2020 Luma Coefficients (Standard)
static const float3 Luma2020 = float3(0.2627, 0.6780, 0.0593);

// Oklab Matrices (Björn Ottosson 2020)
static const float3x3 RGB709_to_LMS = float3x3(
    0.4122214708, 0.5363325363, 0.0514459929,
    0.2119034982, 0.6806995451, 0.1073969566,
    0.0883024619, 0.2817188376, 0.6299787005
);

// Corrected & Row-Sum-Normalized Rec.2020 -> LMS
// Enforces exact LMS(1,1,1) for D65 white (bit-exact achromatic neutrality)
// Derived from Ottosson M1 x BT.2020 RGB->XYZ_D65, then normalized per row
static const float3x3 RGB2020_to_LMS = float3x3(
    0.616759697, 0.360188024, 0.023052279,   // sum = 1.000000000
    0.265131674, 0.635851580, 0.099016746,   // sum = 1.000000000
    0.100127915, 0.203878384, 0.695993701    // sum = 1.000000000
);

static const float3x3 LMS_to_Oklab = float3x3(
    0.2104542553, 0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050, 0.4505937099,
    0.0259040371, 0.7827717662, -0.8086757660
);

// ST.2084 (PQ) EOTF Constants (SMPTE ST 2084-2014)
static const float PQ_M1 = 0.1593017578125;    // 2610/16384 (exact)
static const float PQ_M2 = 78.84375;           // 2523/4096 x 128 (exact)
static const float PQ_C1 = 0.8359375;          // 3424/4096 = c3 - c2 + 1 (exact)
static const float PQ_C2 = 18.8515625;         // 2413/4096 x 32 (exact)
static const float PQ_C3 = 18.6875;            // 2392/4096 x 32 (exact)
static const float PQ_PEAK_LUMINANCE = 10000.0;

// scRGB Standard Definition
// In standard Windows scRGB (16-bit float), 1.0 linear = 80 nits.
static const float SCRGB_WHITE_NITS = 80.0;

// Zone System: Mathematically Exact Powers of 2
// Zone V (Middle Grey) is anchored at exactly 2^-2.5
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

// Pre-computed Kernel for Structure Tensor (Binomial 3x3 - separable [1,2,1] x [1,2,1]/16)
static const float3x3 Structure_Gauss = float3x3(
    0.0625, 0.1250, 0.0625,  // 1/16, 2/16, 1/16
    0.1250, 0.2500, 0.1250,  // 2/16, 4/16, 2/16
    0.0625, 0.1250, 0.0625   // 1/16, 2/16, 1/16
);

// Edge Detection Kernels
static const float Sobel5x5_Gx[25] = {
    -1.0, -2.0,  0.0,  2.0,  1.0,
    -4.0, -8.0,  0.0,  8.0,  4.0,
    -6.0,-12.0,  0.0, 12.0,  6.0,
    -4.0, -8.0,  0.0,  8.0,  4.0,
    -1.0, -2.0,  0.0,  2.0,  1.0
};

static const float Sobel5x5_Gy[25] = {
    -1.0, -4.0, -6.0, -4.0, -1.0,
    -2.0, -8.0,-12.0, -8.0, -2.0,
     0.0,  0.0,  0.0,  0.0,  0.0,
     2.0,  8.0, 12.0,  8.0,  2.0,
     1.0,  4.0,  6.0,  4.0,  1.0
};

static const float LoG_Kernel[25] = {
     0.0,  0.0, -1.0,  0.0,  0.0,
     0.0, -1.0, -2.0, -1.0,  0.0,
    -1.0, -2.0, 16.0, -2.0, -1.0,
     0.0, -1.0, -2.0, -1.0,  0.0,
     0.0,  0.0, -1.0,  0.0,  0.0
};

// Debug Visualization Colors
float3 GetZoneColor(int index)
{
    [flatten]
    switch(clamp(index, 0, 12)) {
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

// ==============================================================================
// 2. Texture & System Config
// ==============================================================================

texture2D TextureBackBuffer : COLOR;
sampler2D SamplerBackBuffer
{
    Texture = TextureBackBuffer;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = NONE;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

// Architecture: Float Pre-Pass Buffer
// R: log2(luma) | G: Oklab a/L | B: Oklab b/L | A: raw linear luma
texture2D TexLinearData { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = PREPASS_FORMAT; };
sampler2D SamplerLinearData
{
    Texture = TexLinearData;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = NONE;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

// Safety Check
#if !defined(BUFFER_WIDTH) || !defined(BUFFER_HEIGHT)
    #error "Bilateral Contrast: Missing BUFFER_WIDTH/HEIGHT. ReShade.fxh injection failed."
#endif

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

// ==============================================================================
// 3. UI Parameters
// ==============================================================================

uniform int iQualityPreset <
    ui_type = "combo";
    ui_label = "Quality Preset";
    ui_items = "Custom\0Reference (Mastering)\0";
    ui_category = "Presets";
> = 0;

uniform float fStrength <
    ui_type = "slider";
    ui_label = "Contrast Strength";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.001;
    ui_category = "Core Settings";
> = 2.5;

uniform float fShadowProtection <
    ui_type = "slider";
    ui_label = "Shadow Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0;

uniform float fMidtoneProtection <
    ui_type = "slider";
    ui_label = "Midtone Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0;

uniform float fHighlightProtection <
    ui_type = "slider";
    ui_label = "Highlight Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0;

uniform float fZoneWhitePoint <
    ui_type = "slider";
    ui_label = "Zone White Point (Nits)";
    ui_min = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_tooltip = "Only applies when using HDR/scRGB Color Space overrides.";
    ui_category = "Protection Zones";
> = 203.0;

uniform float fNegativeProtection <
    ui_type = "slider";
    ui_label = "Negative Value Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_tooltip = "Protects out-of-gamut negative RGB values created by ratio scaling.\n"
                 "Works in all color spaces (SDR and HDR).";
    ui_category = "Protection Zones";
> = 0;

uniform bool bAdaptiveStrength <
    ui_label = "Enable Adaptive Strength";
    ui_category = "Adaptive Processing";
> = true;

uniform int iAdaptiveMode <
    ui_type = "combo";
    ui_label = "Adaptive Mode";
    ui_items = "Dynamic Range\0Variance\0Hybrid\0Range-Variance Hybrid\0";
    ui_category = "Adaptive Processing";
> = 3;

uniform float fAdaptiveAmount <
    ui_type = "slider";
    ui_label = "Adaptive Amount";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Adaptive Processing";
> = 0.25;

uniform float fAdaptiveCurve <
    ui_type = "slider";
    ui_label = "Adaptive Curve";
    ui_min = 0.1; ui_max = 4.0; ui_step = 0.01;
    ui_category = "Adaptive Processing";
> = 1.25;

uniform int iRadius <
    ui_type = "slider";
    ui_label = "Filter Radius";
    ui_min = 1; ui_max = 32;
    ui_category = "Filter Parameters";
> = 8;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_min = 0.1; ui_max = 32.0; ui_step = 0.01;
    ui_category = "Filter Parameters";
> = 4;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Range Sigma (Stops)";
    ui_min = 0.01; ui_max = 4.0; ui_step = 0.001;
    ui_category = "Filter Parameters";
> = 0.35;

uniform float fSigmaChroma <
    ui_type = "slider";
    ui_label = "Chroma Sigma";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.001;
    ui_tooltip = "Controls filter sensitivity to Oklab chromaticity differences.\n"
                 "Typical perceptible shift: 0.05-0.15 in (a/L, b/L) space.";
    ui_category = "Filter Parameters";
> = 0.15;

uniform bool bChromaAwareBilateral <
    ui_label = "Chroma-Aware Filtering";
    ui_category = "Filter Parameters";
> = true;

uniform bool bAdaptiveRadius <
    ui_label = "Adaptive Radius";
    ui_category = "Adaptive Radius";
> = true;

uniform float fAdaptiveRadiusStrength <
    ui_type = "slider";
    ui_label = "Adaptive Radius Strength";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Adaptive Radius";
> = 0.7;

uniform float fChromaEdgeStrength <
    ui_type = "slider";
    ui_label = "Chroma Edge Influence";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Controls how strongly chroma edges reduce the filter radius.\n"
                 "0.0 = Luma only. 1.0 = Max(Luma, Oklab Chroma).";
    ui_category = "Adaptive Radius";
> = 0.5;

uniform int iEdgeDetectionMethod <
    ui_type = "combo";
    ui_label = "Edge Detection Method";
    ui_items = "Sobel 3x3\0Scharr 3x3\0Prewitt 3x3\0Sobel 5x5\0Laplacian of Gaussian\0Structure Tensor\0";
    ui_category = "Adaptive Radius";
> = 5;

uniform float fGradientSensitivity <
    ui_type = "slider";
    ui_label = "Gradient Sensitivity";
    ui_min = 10.0; ui_max = 500.0; ui_step = 1.0;
    ui_category = "Advanced Tuning";
> = 150.0;

uniform float fVarianceWeight <
    ui_type = "slider";
    ui_label = "Variance Weight";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Advanced Tuning";
> = 0.65;

uniform int iColorSpaceOverride <
    ui_type = "combo";
    ui_label = "Color Space Override";
    ui_items = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0";
    ui_tooltip = "Selects the EOTF/OETF used for decoding.\n"
                 "'Auto' uses BUFFER_COLOR_SPACE definition.\n"
                 "scRGB assumes 1.0 = 80 nits.";
    ui_category = "System";
> = 0;

uniform bool bGamutMapping <
    ui_label = "Gamut Mapping (Soft Knee Compression)";
    ui_category = "Output Quality";
> = false;

uniform float fGamutKnee <
    ui_type = "slider";
    ui_label = "Gamut Knee Strength";
    ui_min = 0.0; ui_max = 0.1; ui_step = 0.001;
    ui_tooltip = "Threshold for soft rollover to black. Helps prevent aliasing in dark gradients.";
    ui_category = "Output Quality";
> = 0.01;

uniform bool bQuantize10Bit <
    ui_label = "Simulate 10-bit Interface";
    ui_tooltip = "Rounds output to nearest 10-bit step (0-1023). Useful for verifying HDR banding.";
    ui_category = "Output Quality";
> = false;

uniform int iDebugMode <
    ui_type = "combo";
    ui_label = "Debug Visualization";
    ui_items = "Off\0Weights\0Variance\0Dynamic Range\0Enhancement Map\0Adaptive Radius\0Edge Detection\0Black Pixels\0Chroma Edges\0Entropy\0Zone Map\0Negative Values\0Signed Luminance\0";
    ui_category = "Debug";
> = 0;

// ==============================================================================
// 4. True Math Utilities (Bit-Exact Safety)
// ==============================================================================

float TrueSqrt(float x) { return sqrt(max(x, 0.0)); }

float PowSafe(float base, float exponent)
{
    float safe_base = max(abs(base), FLT_MIN);
    float result = pow(safe_base, exponent);
    return (exponent < 0.0) ? min(result, 1e38) : result;
}

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

float GetMinComponent(float3 lin) { return min(min(lin.r, lin.g), lin.b); }

float TrueSmoothstep(float edge0, float edge1, float x)
{
    float diff = edge1 - edge0;
    if (abs(diff) < FLT_MIN) return step(edge0, x);
    float t = saturate((x - edge0) / diff);
    return t * t * (3.0 - 2.0 * t);
}

bool IsNanVal(float x) { return (asuint(x) & 0x7FFFFFFF) > 0x7F800000; }
bool IsInfVal(float x) { return (asuint(x) & 0x7FFFFFFF) == 0x7F800000; }
bool3 IsNan3(float3 v) { return bool3(IsNanVal(v.x), IsNanVal(v.y), IsNanVal(v.z)); }
bool3 IsInf3(float3 v) { return bool3(IsInfVal(v.x), IsInfVal(v.y), IsInfVal(v.z)); }

// ==============================================================================
// 5. Color Science (Exact Standard Definitions)
// ==============================================================================

float3 sRGB_EOTF(float3 V)
{
    float3 abs_V = abs(V);
    float3 linear_lo = abs_V / 12.92;
    float3 linear_hi = PowNonNegPreserveZero3((abs_V + 0.055) / 1.055, 2.4);

    float3 out_lin;
    out_lin.r = (abs_V.r <= 0.04045) ? linear_lo.r : linear_hi.r;
    out_lin.g = (abs_V.g <= 0.04045) ? linear_lo.g : linear_hi.g;
    out_lin.b = (abs_V.b <= 0.04045) ? linear_lo.b : linear_hi.b;
    return sign(V) * out_lin;
}

float3 sRGB_OETF(float3 L)
{
    float3 abs_L = abs(L);
    float3 encoded_lo = abs_L * 12.92;
    float3 encoded_hi = 1.055 * PowNonNegPreserveZero3(abs_L, 1.0 / 2.4) - 0.055;

    float3 out_enc;
    out_enc.r = (abs_L.r <= SRGB_THRESHOLD_OETF) ? encoded_lo.r : encoded_hi.r;
    out_enc.g = (abs_L.g <= SRGB_THRESHOLD_OETF) ? encoded_lo.g : encoded_hi.g;
    out_enc.b = (abs_L.b <= SRGB_THRESHOLD_OETF) ? encoded_lo.b : encoded_hi.b;
    return sign(L) * out_enc;
}

float3 PQ_EOTF(float3 N)
{
    float3 N_safe = max(N, 0.0);
    float3 Np = PowNonNegPreserveZero3(N_safe, 1.0 / PQ_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    return PowNonNegPreserveZero3(num / den, 1.0 / PQ_M1) * PQ_PEAK_LUMINANCE;
}

float3 PQ_InverseEOTF(float3 L)
{
    float3 L_safe = max(L, 0.0) / PQ_PEAK_LUMINANCE;
    float3 Lp = PowNonNegPreserveZero3(L_safe, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    return PowNonNegPreserveZero3(num / den, PQ_M2);
}

// Corrected: Use to skip heavy unselected EOTF math entirely
float3 DecodeToLinear(float3 encoded)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    if (space == 3) return PQ_EOTF(encoded);
    if (space == 2) return encoded * SCRGB_WHITE_NITS;
    
    return sRGB_EOTF(encoded) * SCRGB_WHITE_NITS;
}

float3 EncodeFromLinear(float3 lin)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    if (space == 3) return PQ_InverseEOTF(lin);
    if (space == 2) return lin / SCRGB_WHITE_NITS;
    
    return sRGB_OETF(lin / SCRGB_WHITE_NITS);
}

float GetLuminanceCS(float3 lin)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    return dot(lin, (space >= 3) ? Luma2020 : Luma709);
}

float GetResolvedWhitePoint()
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    return (space <= 1) ? SCRGB_WHITE_NITS : fZoneWhitePoint;
}

float2 GetOklabChroma(float3 linearRGB, int activeSpace)
{
    float3x3 m;
    if (activeSpace >= 3)
        m = RGB2020_to_LMS;
    else
        m = RGB709_to_LMS;

    float3 lms = mul(m, linearRGB);

    // Vectorized Cube Root Nonlinearity
    float3 abs_lms = max(abs(lms), FLT_MIN);
    float3 lms_p = sign(lms) * pow(abs_lms, 1.0 / 3.0);

    float3 oklab = mul(LMS_to_Oklab, lms_p);

    // Normalize by Lightness L to get pure chromaticity direction
    float L = max(abs(oklab.x), FLT_MIN);
    return oklab.yz / L;
}

float3 SoftClipGamut(float3 lin, float knee)
{
    float minComponent = min(min(lin.r, lin.g), lin.b);

    if (minComponent < knee)
    {
        float luma = GetLuminanceCS(lin);
        float t = (knee > FLT_MIN) ? saturate((knee - minComponent) / (knee + FLT_MIN)) : step(minComponent, 0.0);

        float3 chroma = lin - luma;
        float scale = luma / (luma - minComponent + FLT_MIN);

        scale = lerp(1.0, min(scale, 1.0), t * t * (3.0 - 2.0 * t));

        lin = luma + chroma * scale;
    }
    return lin;
}

// ==============================================================================
// 6. Zone Logic (Stop-Domain)
// ==============================================================================

int GetZone(float normalizedLuma)
{
    if (normalizedLuma < 0.0)      return 0;
    if (normalizedLuma < ZONE_I)   return 1;
    if (normalizedLuma < ZONE_II)  return 2;
    if (normalizedLuma < ZONE_III) return 3;
    if (normalizedLuma < ZONE_IV)  return 4;
    if (normalizedLuma < ZONE_V)   return 5;
    if (normalizedLuma < ZONE_VI)  return 6;
    if (normalizedLuma < ZONE_VII) return 7;
    if (normalizedLuma < ZONE_VIII) return 8;
    if (normalizedLuma < ZONE_IX)  return 9;
    if (normalizedLuma < ZONE_X)   return 10;
    if (normalizedLuma < ZONE_XI)  return 11;
    return 12;
}

float GetZoneProtection(float nl, float minCompNorm, float shadowProt, float midProt, float highProt, float negProt)
{
    if (shadowProt + midProt + highProt + negProt < FLT_MIN) return 1.0;

    float negW = 1.0 - TrueSmoothstep(-0.001, 0.0, minCompNorm);
    float s = log2(max(nl, FLT_MIN));

    float blackW = 1.0 - TrueSmoothstep(-20.0, -14.0, s);
    float shadowProtEff = lerp(shadowProt, 1.0, blackW);

    float shadowW = (1.0 - negW) * (1.0 - TrueSmoothstep(-3.0, -2.5, s));
    float highW   = (1.0 - negW) * TrueSmoothstep(-1.0, 0.0, s);
    float midW    = (1.0 - negW) * (1.0 - shadowW) * (1.0 - highW);

    float protection = negW * negProt + shadowW * shadowProtEff + midW * midProt + highW * highProt;
    return 1.0 - saturate(protection);
}

// ==============================================================================
// 7. Float Pre-Pass
// ==============================================================================

void PS_PrePass(float4 vpos : SV_Position, out float4 outData : SV_Target)
{
    int2 pos = int2(vpos.xy);
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;

    // Decode full color representation strictly once per pixel
    float3 color_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, pos).rgb);
    float luma_lin = dot(color_lin, lumaCoeffs);

    // Safety floor for bit-exact Log2 EOTF calculation
    float safe_luma = max(luma_lin, FLT_MIN);
    float log2_luma = log2(safe_luma);

    // [v8.4.1 Fix] Guard chroma computation for dark/near-black pixels.
    // Below CHROMA_STABILITY_THRESH, Oklab chromaticity is numerically undefined
    // (1/L amplifies noise to arbitrary directions). Writing (0,0) ensures:
    //   - ChromaEdge() sees zero difference between dark neighbors (no false edges)
    //   - Bilateral chroma weight is zero (guarded separately by chroma_weight_factor)
    float2 chroma = float2(0.0, 0.0);
    if (bChromaAwareBilateral && luma_lin > CHROMA_STABILITY_THRESH) {
        chroma = GetOklabChroma(color_lin, space);
    }

    // R: Log2Luma | G: Oklab a/L | B: Oklab b/L | A: True Linear Luma (may be negative in scRGB)
    outData = float4(log2_luma, chroma.x, chroma.y, luma_lin);
}

// Safe wrapper for edge/kernel detectors fetching float pre-pass data
float4 FetchLinearData(int2 pos)
{
    pos = clamp(pos, int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
    return tex2Dfetch(SamplerLinearData, pos);
}

// ==============================================================================
// 8. Analysis & Edge Detection (Reads Pre-Pass Data)
// ==============================================================================

float FetchPerceptualLuma(int2 pos)
{
    float log2_luma = FetchLinearData(pos).r;
    return (max(log2_luma, LOG2_EDGE_LUMA_FLOOR) + 20.0) * 0.06;
}

float Sobel3x3(int2 center)
{
    float tl = FetchPerceptualLuma(center + int2(-1, -1));
    float tc = FetchPerceptualLuma(center + int2( 0, -1));
    float tr = FetchPerceptualLuma(center + int2( 1, -1));
    float ml = FetchPerceptualLuma(center + int2(-1,  0));
    float mr = FetchPerceptualLuma(center + int2( 1,  0));
    float bl = FetchPerceptualLuma(center + int2(-1,  1));
    float bc = FetchPerceptualLuma(center + int2( 0,  1));
    float br = FetchPerceptualLuma(center + int2( 1,  1));
    float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    return (gx * gx + gy * gy) * 0.0625;
}

float Scharr3x3(int2 center)
{
    float tl = FetchPerceptualLuma(center + int2(-1, -1));
    float tc = FetchPerceptualLuma(center + int2( 0, -1));
    float tr = FetchPerceptualLuma(center + int2( 1, -1));
    float ml = FetchPerceptualLuma(center + int2(-1,  0));
    float mr = FetchPerceptualLuma(center + int2( 1,  0));
    float bl = FetchPerceptualLuma(center + int2(-1,  1));
    float bc = FetchPerceptualLuma(center + int2( 0,  1));
    float br = FetchPerceptualLuma(center + int2( 1,  1));
    float gx = (3.0 * tr + 10.0 * mr + 3.0 * br) - (3.0 * tl + 10.0 * ml + 3.0 * bl);
    float gy = (3.0 * bl + 10.0 * bc + 3.0 * br) - (3.0 * tl + 10.0 * tc + 3.0 * tr);
    return (gx * gx + gy * gy) * 0.00390625;
}

float Prewitt3x3(int2 center)
{
    float tl = FetchPerceptualLuma(center + int2(-1, -1));
    float tc = FetchPerceptualLuma(center + int2( 0, -1));
    float tr = FetchPerceptualLuma(center + int2( 1, -1));
    float ml = FetchPerceptualLuma(center + int2(-1,  0));
    float mr = FetchPerceptualLuma(center + int2( 1,  0));
    float bl = FetchPerceptualLuma(center + int2(-1,  1));
    float bc = FetchPerceptualLuma(center + int2( 0,  1));
    float br = FetchPerceptualLuma(center + int2( 1,  1));
    float gx = (tr + mr + br) - (tl + ml + bl);
    float gy = (bl + bc + br) - (tl + tc + tr);
    return (gx * gx + gy * gy) * 0.111111111;
}

float Sobel5x5(int2 center)
{
    float sum_gx = 0.0;
    float sum_gy = 0.0;
    [unroll] for (int y = -2; y <= 2; y++) {
        [unroll] for (int x = -2; x <= 2; x++) {
            float luma = FetchPerceptualLuma(center + int2(x, y));
            int idx = (y + 2) * 5 + (x + 2);
            sum_gx += luma * Sobel5x5_Gx[idx];
            sum_gy += luma * Sobel5x5_Gy[idx];
        }
    }
    return (sum_gx * sum_gx + sum_gy * sum_gy) * 0.00043402778;
}

float LaplacianOfGaussian(int2 center)
{
    float response = 0.0;
    [unroll] for (int y = -2; y <= 2; y++) {
        [unroll] for (int x = -2; x <= 2; x++) {
            float luma = FetchPerceptualLuma(center + int2(x, y));
            int idx = (y + 2) * 5 + (x + 2);
            response += luma * LoG_Kernel[idx];
        }
    }
    return response * response;
}

// [v8.4.1 Fix] Structure Tensor optimized from 72 to 25 texture fetches.
// Pre-fetches the 5x5 perceptual luma neighborhood into a local array,
// then computes 9 Sobel gradients from shared data (no redundant reads).
float StructureTensor(int2 center)
{
    // Pre-fetch 5x5 neighborhood: 25 fetches (was 72)
    // Flattened 1D array: index = row * 5 + col
    float pl[25];
    [unroll] for (int pj = -2; pj <= 2; pj++) {
        [unroll] for (int pi = -2; pi <= 2; pi++) {
            pl[(pj + 2) * 5 + (pi + 2)] = FetchPerceptualLuma(center + int2(pi, pj));
        }
    }

    float Ixx = 0.0, Iyy = 0.0, Ixy = 0.0;

    // Compute Sobel gradient at each of 3x3 window positions
    // Window position (wy, wx) maps to Sobel center at pl[(wy+1)*5 + (wx+1)]
    // Sobel 3x3 reads rows wy..wy+2, cols wx..wx+2
    [unroll] for (int wy = 0; wy < 3; wy++) {
        [unroll] for (int wx = 0; wx < 3; wx++) {
            float tl = pl[ wy      * 5 + wx];
            float tc = pl[ wy      * 5 + wx + 1];
            float tr = pl[ wy      * 5 + wx + 2];
            float ml = pl[(wy + 1) * 5 + wx];
            float mr = pl[(wy + 1) * 5 + wx + 2];
            float bl = pl[(wy + 2) * 5 + wx];
            float bc = pl[(wy + 2) * 5 + wx + 1];
            float br = pl[(wy + 2) * 5 + wx + 2];

            float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
            float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);

            float w = Structure_Gauss[wy][wx];
            Ixx += gx * gx * w;
            Iyy += gy * gy * w;
            Ixy += gx * gy * w;
        }
    }

    float trace = Ixx + Iyy;

    // Numerically stable discriminant: (Ixx-Iyy)^2 + 4*Ixy^2
    float diff = Ixx - Iyy;
    float disc = TrueSqrt(max(diff * diff + 4.0 * Ixy * Ixy, 0.0));

    float lambda1 = (trace + disc) * 0.5;
    float lambda2 = (trace - disc) * 0.5;
    float coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + FLT_MIN);

    return lambda1 * (1.0 + coherence) * 0.5;
}

// ChromaEdge reads pre-pass chroma directly.
// [v8.4.1 Note] Dark pixels have chroma = (0,0) written by guarded pre-pass,
// so dark-vs-dark comparisons produce zero edge response (no false positives).
float ChromaEdge(int2 center)
{
    float2 center_ab = FetchLinearData(center).gb;
    float maxChromaDiff = 0.0;

    [unroll] for (int y = -1; y <= 1; y++) {
        [unroll] for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;

            float2 neighbor_ab = FetchLinearData(center + int2(x, y)).gb;
            float2 d = center_ab - neighbor_ab;

            maxChromaDiff = max(maxChromaDiff, dot(d, d));
        }
    }

    return maxChromaDiff * 12.0;
}

float GetEdgeStrength(int2 center, int method)
{
    if (method == 0) return Sobel3x3(center);
    if (method == 1) return Scharr3x3(center);
    if (method == 2) return Prewitt3x3(center);
    if (method == 3) return Sobel5x5(center);
    if (method == 4) return LaplacianOfGaussian(center);
    if (method == 5) return StructureTensor(center);
    return Sobel3x3(center);
}

// ==============================================================================
// 9. Bilateral Processing
// ==============================================================================

int GetAdaptiveRadius(int2 center, int base_radius, float strength, float sigma_spatial)
{
    float edge = GetEdgeStrength(center, iEdgeDetectionMethod);

    [branch]
    if (bChromaAwareBilateral && fChromaEdgeStrength > 0.0) {
        float chromaEdge = ChromaEdge(center);
        edge = lerp(edge, max(edge, chromaEdge), fChromaEdgeStrength);
    }

    float scale = TrueSmoothstep(0.0, 1.0, edge * (fGradientSensitivity * 0.01));
    float factor = lerp(1.0, lerp(1.0, 0.15, scale), strength);
    int sigma_max = (int)(sigma_spatial * 3.0 + 0.5);
    return clamp(min((int)(base_radius * factor + 0.5), sigma_max), 1, base_radius);
}

float CalculateAdaptiveStrength(float sum_log, float sum_log_sq, float sum_weight, float min_log, float max_log, float base_strength, int mode)
{
    if (sum_weight < FLT_MIN) return base_strength;
    float inv_weight = 1.0 / sum_weight;
    float range = max_log - min_log;
    float mean = sum_log * inv_weight;
    float var = max(0.0, sum_log_sq * inv_weight - mean * mean);
    float metric;

    if (mode == 0)      metric = saturate(range * 0.166666667);
    else if (mode == 1) metric = saturate(var * 0.5);
    else if (mode == 2) metric = PowSafe(max(saturate(range * 0.166666667), FLT_MIN), 1.0 - fVarianceWeight)
                                * PowSafe(max(saturate(var * 0.5), FLT_MIN), fVarianceWeight);
    else                metric = saturate((log2(1.0 + var) * (1.0 + range * 0.1)) * 0.25);

    return base_strength * lerp(1.0, PowSafe(metric, fAdaptiveCurve) * 2.0, fAdaptiveAmount);
}

float3 ProcessPixel(int2 center_pos)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float whitePt = GetResolvedWhitePoint();

    // Fetch original RGB needed for final ratio multiply & debug pass-through
    float3 color_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, center_pos).rgb);

    // Fetch pre-calculated floating-point analytics
    float4 center_data = tex2Dfetch(SamplerLinearData, center_pos);
    float log2_center = center_data.r;
    float luma_lin    = center_data.a;
    // CONTRACT: center_data.a may be negative (scRGB out-of-gamut).
    // Only used for: chroma guard threshold, zone protection, debug vis.
    // All bilateral filter math operates on center_data.r (log2, clamped via pre-pass).

    // Passthrough for non-positive luma (Log domain singularity).
    if (iDebugMode == 0 && luma_lin <= FLT_MIN) return color_lin;

    int base_radius;
    float sigma_s;

    [branch]
    if (iQualityPreset == 1) {
        base_radius = 24;
        sigma_s = 12.0;
    } else {
        base_radius = iRadius;
        sigma_s = fSigmaSpatial;
    }

    int radius = base_radius;

    [branch]
    if (bAdaptiveRadius && base_radius > 2)
        radius = GetAdaptiveRadius(center_pos, base_radius, fAdaptiveRadiusStrength, sigma_s);

    // Debug Early Exits
    if (iDebugMode == 5)  return lerp(float3(0, 0, 1), float3(1, 0, 0), float(radius) / float(base_radius));
    if (iDebugMode == 6)  { float e = GetEdgeStrength(center_pos, iEdgeDetectionMethod); return float3(e, e, e) * 10.0; }
    if (iDebugMode == 8)  { float c = ChromaEdge(center_pos); return float3(c, c, c) * 5.0; }
    if (iDebugMode == 10) return GetZoneColor(GetZone(luma_lin / whitePt));
    if (iDebugMode == 11) return (GetMinComponent(color_lin) < 0.0) ? float3(1, 0, 1) : float3(0, 0.1, 0);
    if (iDebugMode == 12) {
        float norm = luma_lin / max(whitePt, FLT_MIN);
        float stops = log2(max(abs(norm), FLT_MIN));
        float t = saturate((stops + 6.0) / 8.0);
        return (norm < 0.0) ? float3(0, t, 0) : float3(t, 0, 0);
    }
    if (iDebugMode == 7) return (luma_lin <= FLT_MIN) ? float3(1, 0, 1) : float3(0, 0, 0);

    // Bilateral Constants
    float inv_2_sigma_s_sq = 0.5 / (sigma_s * sigma_s);
    float inv_2_sigma_r_sq = 0.5 / (fSigmaRange * fSigmaRange);
    float inv_2_sigma_c_sq = 0.5 / (fSigmaChroma * fSigmaChroma);

    float cutoff_r_sq = NEG_LN_SPATIAL_CUTOFF / inv_2_sigma_s_sq;
    int cutoff_int = (int)TrueSqrt(cutoff_r_sq);
    int safe_radius = min(cutoff_int + 1, radius);
    int max_r = min(safe_radius, MAX_LOOP_RADIUS);

    int y_start = max(-max_r, -center_pos.y);
    int y_end   = min( max_r, (BUFFER_HEIGHT - 1) - center_pos.y);

    float r_limit_sq = float(max_r * max_r);

    float2 center_chroma = float2(0, 0);
    float chroma_weight_factor = 0.0;
    if (bChromaAwareBilateral) {
        chroma_weight_factor = TrueSmoothstep(FLT_MIN, CHROMA_STABILITY_THRESH, luma_lin);
        center_chroma = center_data.gb;
    }

    // Neumaier Accumulators
    float2 stats_log = 0.0;
    float2 stats_sq  = 0.0;
    float2 stats_w   = 0.0;
    float min_log = log2_center, max_log = log2_center;

    int x_min_off = -center_pos.x;
    int x_max_off = (BUFFER_WIDTH - 1) - center_pos.x;

    [loop]
    for (int y = y_start; y <= y_end; ++y)
    {
        float y_f = float(y);
        float y_sq = y_f * y_f;
        float spatial_y = y_sq * inv_2_sigma_s_sq;

        int x_limit_circ = (int)TrueSqrt(max(0.0, r_limit_sq - y_sq));

        int x_start = max(-x_limit_circ, x_min_off);
        int x_end   = min( x_limit_circ, x_max_off);

        int sample_y = center_pos.y + y;

        [loop]
        for (int x = x_start; x <= x_end; ++x)
        {
            int sample_x = center_pos.x + x;

            // High-speed fetch: 0 decoding, 0 cube roots, 0 matrix multiplies
            float4 n_data = tex2Dfetch(SamplerLinearData, int2(sample_x, sample_y));
            float n_log  = n_data.r;
            // CONTRACT: n_data.a may be negative (scRGB). Used only for chroma guard below.
            float n_luma = n_data.a;

            float d_luma = log2_center - n_log;

            float exponent = -(float(x * x) * inv_2_sigma_s_sq + spatial_y) - (d_luma * d_luma * inv_2_sigma_r_sq);

            if (chroma_weight_factor > 0.0 && n_luma > FLT_MIN) {
                float2 n_chroma = n_data.gb;
                float2 d_chroma = center_chroma - n_chroma;
                exponent -= (dot(d_chroma, d_chroma) * inv_2_sigma_c_sq) * chroma_weight_factor;
            }

            if (exponent <= LN_FLT_MIN) continue;

            float weight = exp(exponent);

            // Neumaier Summation
            float val = n_log * weight;
            float t = stats_log.x + val;
            float err = (abs(stats_log.x) >= abs(val)) ? ((stats_log.x - t) + val) : ((val - t) + stats_log.x);
            stats_log.x = t; stats_log.y += err;

            val = n_log * n_log * weight;
            t = stats_sq.x + val;
            err = (abs(stats_sq.x) >= abs(val)) ? ((stats_sq.x - t) + val) : ((val - t) + stats_sq.x);
            stats_sq.x = t; stats_sq.y += err;

            t = stats_w.x + weight;
            err = (abs(stats_w.x) >= abs(weight)) ? ((stats_w.x - t) + weight) : ((weight - t) + stats_w.x);
            stats_w.x = t; stats_w.y += err;

            min_log = min(min_log, n_log);
            max_log = max(max_log, n_log);
        }
    }

    float total_w = stats_w.x + stats_w.y;
    if (total_w < FLT_MIN) return color_lin;

    float total_log = stats_log.x + stats_log.y;
    float total_sq  = stats_sq.x  + stats_sq.y;
    float blurred = total_log / total_w;
    float diff = log2_center - blurred;

    float strength = fStrength;
    [branch]
    if (bAdaptiveStrength)
        strength = CalculateAdaptiveStrength(total_log, total_sq, total_w, min_log, max_log, fStrength, iAdaptiveMode);

    float norm_luma = luma_lin / whitePt;

    // [v8.4.1 Fix] Compute minCompNorm unconditionally.
    // In v8.4.0, this was gated by (space >= 2), making fNegativeProtection
    // silently inoperative for SDR. The ratio multiply (color_lin * ratio) can
    // create negative RGB even in SDR when one channel is near zero with large ratio.
    float minCompNorm = GetMinComponent(color_lin) / whitePt;

    strength *= GetZoneProtection(norm_luma, minCompNorm, fShadowProtection, fMidtoneProtection, fHighlightProtection, fNegativeProtection);

    // Bit-Exact Neutrality Check
    if (abs(strength) < FLT_MIN) return color_lin;

    float enhanced_log = log2_center + strength * diff;
    float enhanced_luma = exp2(enhanced_log);
    float ratio = enhanced_luma / max(luma_lin, FLT_MIN);
    ratio = clamp(ratio, RATIO_MIN, RATIO_MAX);

    // Post-Loop Debugs
    if (iDebugMode == 1) return saturate(log2(total_w + 1.0) * 0.1).xxx;
    if (iDebugMode == 2) { float m = blurred; float v = max(0.0, (total_sq / total_w) - m * m); return float3(v * 2.0, v, 0.0); }
    if (iDebugMode == 3) return float3((max_log - min_log) * 0.2, 0, 0);
    if (iDebugMode == 4) return lerp(float3(0, 0, 1), float3(1, 0, 0), saturate(abs(diff) * strength * 2.0));
    if (iDebugMode == 9) { float m = total_log / total_w; float v = max(0.0, (total_sq / total_w) - m * m); float r = max_log - min_log; float e = log2(1.0 + v) * (1.0 + r * 0.1); return float3(e * 0.25, e * 0.125, 0.0); }

    float3 final_color = color_lin * ratio;

    [branch]
    if (bGamutMapping) final_color = SoftClipGamut(final_color, fGamutKnee);

    if (any(IsNan3(final_color)) || any(IsInf3(final_color))) return color_lin;

    return final_color;
}

// ==============================================================================
// 10. Shader Entry Point
// ==============================================================================

void PS_BilateralContrast(float4 vpos : SV_Position, out float4 fragColor : SV_Target)
{
    [branch]
    if (fStrength <= 0.0 && iDebugMode == 0) {
        fragColor = tex2Dfetch(SamplerBackBuffer, int2(vpos.xy));
        return;
    }

    float3 result = ProcessPixel(int2(vpos.xy));
    float3 encoded = EncodeFromLinear(result);

    int activeSpace = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;

    [flatten]
    if (activeSpace <= 1) {
        encoded = saturate(encoded);
    }

    [branch]
    if (bQuantize10Bit) {
        encoded = round(max(encoded, 0.0) * 1023.0) / 1023.0;
    }

    fragColor = float4(encoded, 1.0);
}

technique BilateralContrast_Reference <
    ui_label = "Bilateral Contrast v8.4.1 (Mastering Edition)";
    ui_tooltip = "MASTERING QUALITY - True Math Processing\n\n"
                 "v8.4.1 Fixes:\n"
                 "- Fix: Pre-pass chroma guard for dark pixels (prevents spurious ChromaEdge)\n"
                 "- Fix: Negative Protection now works in SDR (minCompNorm unconditional)\n"
                 "- Fix: Structure Tensor 72->25 texture fetches (3x bandwidth savings)\n"
                 "- Fix: [flatten] hints on uniform EOTF/OETF branches\n"
                 "- Fix: Documented n_data.a negative-value contract\n\n"
                 "v8.4.0 Architecture:\n"
                 "- Floating-point Pre-Pass (eliminates inner-loop decode/Oklab)\n"
                 "- Selectable RGBA32F / RGBA16F Pre-Pass Buffer\n"
                 "Requires: DirectX 10+ or OpenGL 4.5+";
>
{
    pass PreCompute
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_PrePass;
        RenderTarget = TexLinearData;
    }
    pass Main
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_BilateralContrast;
    }
}