/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement - COMPUTE LDS EDITION
 *
 * Design Philosophy: PRECISION OVER PERFORMANCE
 * - True IEEE 754 Math (No fast intrinsics or approximations)
 * - Exact IEC/SMPTE Standard Constants
 * - Bit-Exact Neutrality Logic
 * - Pre-computed High-Precision Kernels
 * - True Stop-Domain HDR Processing
 * - Oklab Perceptual Chromaticity Processing
 * - Compute Shader with groupshared LDS tile cache
 *
 * Version: 8.5.0 (Compute LDS Edition)
 * - Compute Shader path with 32x32 groupshared tile (16x16 threads + 8px halo)
 * - Loop fission: LDS inner tile + VRAM outer fallback for radii > 8
 * - All edge detection and chroma analysis via LDS (zero VRAM fetches in hot paths)
 * - Aligned: PQ debug encoding uses 80 nits reference (SCRGB_WHITE_NITS)
 * - Requires: DirectX 11+, OpenGL 4.3+, or Vulkan
 *
 * Author: startuga
 * Formatter: Strict Opinionated Style (Allman, 4-space, Aligned Macros)
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

static const float FLT_MIN                 = 1.175494351e-38;
static const float LN_FLT_MIN              = -87.33654475;
static const float NEG_LN_SPATIAL_CUTOFF   = 9.210340372;

static const int MAX_LOOP_RADIUS           = 32;
static const int LDS_TILE_SIZE             = 32;
static const int LDS_HALO                  = 8;
static const int LDS_RADIUS                = 8;

static const float RATIO_MIN               = 0.0001;
static const float RATIO_MAX               = 10000.0;
static const float CHROMA_STABILITY_THRESH = 1e-4;
static const float CHROMA_RELIABILITY_START= 5e-5;
static const float INV_CHROMA_RELIABILITY_SPAN = 1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);
static const float EDGE_LUMA_FLOOR         = 1e-4;
static const float LOG2_EDGE_LUMA_FLOOR    = -13.2877123795;

static const float SRGB_THRESHOLD_EOTF     = 0.04045;
static const float SRGB_THRESHOLD_OETF     = (0.04045 / 12.92);

static const float3 Luma709                = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020               = float3(0.2627, 0.6780, 0.0593);

// Oklab Matrices (Björn Ottosson 2020)
static const float3x3 RGB709_to_LMS = float3x3(
    0.4122214708, 0.5363325363, 0.0514459929,
    0.2119034982, 0.6806995451, 0.1073969566,
    0.0883024619, 0.2817188376, 0.6299787005
);

// Corrected & Row-Sum-Normalized Rec.2020 -> LMS
// Enforces exact LMS(1,1,1) for D65 white (bit-exact achromatic neutrality)
static const float3x3 RGB2020_to_LMS = float3x3(
    0.616759697, 0.360188024, 0.023052279,
    0.265131674, 0.635851580, 0.099016746,
    0.100127915, 0.203878384, 0.695993701
);

static const float3x3 LMS_to_Oklab = float3x3(
    0.2104542553,  0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050,  0.4505937099,
    0.0259040371,  0.7827717662, -0.8086757660
);

// ST.2084 (PQ) EOTF Constants (SMPTE ST 2084-2014)
static const float PQ_M1             = 0.1593017578125;
static const float PQ_M2             = 78.84375;
static const float PQ_C1             = 0.8359375;
static const float PQ_C2             = 18.8515625;
static const float PQ_C3             = 18.6875;
static const float PQ_PEAK_LUMINANCE = 10000.0;

// scRGB Standard Definition (1.0 linear = 80 nits)
static const float SCRGB_WHITE_NITS  = 80.0;

// Exact Photographic Zones
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

static const float3x3 Structure_Gauss = float3x3(
    0.0625, 0.1250, 0.0625,
    0.1250, 0.2500, 0.1250,
    0.0625, 0.1250, 0.0625
);

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

// ==============================================================================
// 2. Texture & System Config
// ==============================================================================

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

texture2D TexLinearData { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = PREPASS_FORMAT; };
sampler2D SamplerLinearData
{
    Texture   = TexLinearData;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

texture2D TexBilateralOut { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
storage2D StorageBilateralOut { Texture = TexBilateralOut; };
sampler2D SamplerBilateralOut
{
    Texture   = TexBilateralOut;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

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
> = 3.0;

uniform float fShadowProtection <
    ui_type = "slider";
    ui_label = "Shadow Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0.15;

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
> = 0.10;

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
    ui_tooltip = "Protects out-of-gamut negative RGB values created or preserved by ratio scaling.\nWorks in all color spaces (SDR and HDR).";
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
> = 0.35;

uniform float fAdaptiveCurve <
    ui_type = "slider";
    ui_label = "Adaptive Curve";
    ui_min = 0.1; ui_max = 4.0; ui_step = 0.01;
    ui_category = "Adaptive Processing";
> = 1.50;

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
> = 3.0;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Range Sigma (Stops)";
    ui_min = 0.01; ui_max = 4.0; ui_step = 0.001;
    ui_category = "Filter Parameters";
> = 0.45;

uniform float fSigmaChroma <
    ui_type = "slider";
    ui_label = "Chroma Sigma";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.001;
    ui_tooltip = "Controls filter sensitivity to Oklab chromaticity differences.\nTypical perceptible shift: 0.05-0.15 in (a/L, b/L) space.";
    ui_category = "Filter Parameters";
> = 0.12;

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
> = 0.80;

uniform float fChromaEdgeStrength <
    ui_type = "slider";
    ui_label = "Chroma Edge Influence";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Controls how strongly chroma edges reduce the filter radius.\n0.0 = Luma only. 1.0 = Max(Luma, Oklab Chroma).";
    ui_category = "Adaptive Radius";
> = 0.60;

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
> = 200.0;

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
    ui_tooltip = "Selects the EOTF/OETF used for decoding.\n'Auto' uses BUFFER_COLOR_SPACE definition.\nscRGB assumes 1.0 = 80 nits.";
    ui_category = "System";
> = 0;

uniform int iDebugMode <
    ui_type = "combo";
    ui_label = "Debug Visualization";
    ui_items = "Off\0Weights\0Variance\0Dynamic Range\0Enhancement Map\0Adaptive Radius\0Edge Detection\0Black Pixels\0Chroma Edges\0Entropy\0Zone Map\0Negative Values\0Signed Luminance\0";
    ui_category = "Debug";
> = 0;

// ==============================================================================
// 4. True Math Utilities (Bit-Exact Safety)
// ==============================================================================

float TrueSqrt(float x) 
{ 
    return sqrt(max(x, 0.0)); 
}

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

float GetMinComponent(float3 lin) 
{ 
    return min(min(lin.r, lin.g), lin.b); 
}

float TrueSmoothstep(float edge0, float edge1, float x)
{
    float diff = edge1 - edge0;
    if (abs(diff) < FLT_MIN) return step(edge0, x);
    float t = saturate((x - edge0) / diff);
    return t * t * (3.0 - 2.0 * t);
}

bool IsNanVal(float x)   { return (asuint(x) & 0x7FFFFFFF) > 0x7F800000; }
bool IsInfVal(float x)   { return (asuint(x) & 0x7FFFFFFF) == 0x7F800000; }
bool3 IsNan3(float3 v)   { return bool3(IsNanVal(v.x), IsNanVal(v.y), IsNanVal(v.z)); }
bool3 IsInf3(float3 v)   { return bool3(IsInfVal(v.x), IsInfVal(v.y), IsInfVal(v.z)); }

// ==============================================================================
// 5. Color Science (Exact Standard Definitions)
// ==============================================================================

float3 sRGB_EOTF(float3 V)
{
    float3 abs_V = abs(V);
    float3 linear_lo = abs_V / 12.92;
    float3 linear_hi = PowNonNegPreserveZero3((abs_V + 0.055) / 1.055, 2.4);

    float3 out_lin;
    out_lin.r = (abs_V.r <= SRGB_THRESHOLD_EOTF) ? linear_lo.r : linear_hi.r;
    out_lin.g = (abs_V.g <= SRGB_THRESHOLD_EOTF) ? linear_lo.g : linear_hi.g;
    out_lin.b = (abs_V.b <= SRGB_THRESHOLD_EOTF) ? linear_lo.b : linear_hi.b;
    
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
    N = saturate(N);
    float3 Np = PowNonNegPreserveZero3(N, 1.0 / PQ_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    
    return PowNonNegPreserveZero3(num / den, 1.0 / PQ_M1) * PQ_PEAK_LUMINANCE;
}

float3 PQ_InverseEOTF(float3 L)
{
    L = clamp(L, 0.0, PQ_PEAK_LUMINANCE);
    float3 Lp = PowNonNegPreserveZero3(L / PQ_PEAK_LUMINANCE, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    
    return saturate(PowNonNegPreserveZero3(num / den, PQ_M2));
}

float3 DecodeToLinear(float3 encoded)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;

    [branch]
    if (space == 3)
    {
        return PQ_EOTF(encoded);
    }

    [branch]
    if (space == 2)
    {
        return encoded * SCRGB_WHITE_NITS;
    }

    return sRGB_EOTF(encoded) * SCRGB_WHITE_NITS;
}

float3 EncodeFromLinear(float3 lin)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;

    [branch]
    if (space == 3)
    {
        return PQ_InverseEOTF(lin);
    }

    [branch]
    if (space == 2)
    {
        return lin / SCRGB_WHITE_NITS;
    }

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
    
    [branch]
    if (activeSpace >= 3)
    {
        m = RGB2020_to_LMS;
    }
    else
    {
        m = RGB709_to_LMS;
    }
    
    float3 lms = mul(m, linearRGB);
    float3 abs_lms = max(abs(lms), FLT_MIN);
    float3 lms_p = sign(lms) * pow(abs_lms, 1.0 / 3.0);
    float3 oklab = mul(LMS_to_Oklab, lms_p);
    
    float L = max(abs(oklab.x), FLT_MIN);
    return oklab.yz / L;
}

// ==============================================================================
// 6. Zone Logic (Stop-Domain)
// ==============================================================================

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

int GetZone(float normalizedLuma)
{
    if (normalizedLuma < 0.0)       return 0;
    if (normalizedLuma < ZONE_I)    return 1;
    if (normalizedLuma < ZONE_II)   return 2;
    if (normalizedLuma < ZONE_III)  return 3;
    if (normalizedLuma < ZONE_IV)   return 4;
    if (normalizedLuma < ZONE_V)    return 5;
    if (normalizedLuma < ZONE_VI)   return 6;
    if (normalizedLuma < ZONE_VII)  return 7;
    if (normalizedLuma < ZONE_VIII) return 8;
    if (normalizedLuma < ZONE_IX)   return 9;
    if (normalizedLuma < ZONE_X)    return 10;
    if (normalizedLuma < ZONE_XI)   return 11;
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

    float3 color_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, pos).rgb);
    float luma_lin = dot(color_lin, lumaCoeffs);

    float safe_luma = max(luma_lin, FLT_MIN);
    float log2_luma = log2(safe_luma);

    float2 chroma = float2(0.0, 0.0);
    if (bChromaAwareBilateral && luma_lin > CHROMA_RELIABILITY_START)
    {
        chroma = GetOklabChroma(color_lin, space);
    }

    outData = float4(log2_luma, chroma.x, chroma.y, luma_lin);
}

// ==============================================================================
// 8. Analysis & Edge Detection (LDS / Groupshared Optimized)
// ==============================================================================

// Note: groupshared float4 array implies 16-byte strides.
// On some AMD GCN/RDNA architectures, this may cause LDS bank conflicts.
// If LDS latency becomes a bottleneck, consider splitting into planar arrays (R, G, B, A).
groupshared float4 gs_LinearData[LDS_TILE_SIZE * LDS_TILE_SIZE];

#define GS_IDX(x, y) ((y) * LDS_TILE_SIZE + (x))

float FetchPerceptualLumaShared(int2 local_pos)
{
    float log2_luma = gs_LinearData[GS_IDX(local_pos.x, local_pos.y)].r;
    return (max(log2_luma, LOG2_EDGE_LUMA_FLOOR) + 20.0) * 0.06;
}

float Sobel3x3Shared(int2 local_center)
{
    float tl = FetchPerceptualLumaShared(local_center + int2(-1, -1));
    float tc = FetchPerceptualLumaShared(local_center + int2( 0, -1));
    float tr = FetchPerceptualLumaShared(local_center + int2( 1, -1));
    float ml = FetchPerceptualLumaShared(local_center + int2(-1,  0));
    float mr = FetchPerceptualLumaShared(local_center + int2( 1,  0));
    float bl = FetchPerceptualLumaShared(local_center + int2(-1,  1));
    float bc = FetchPerceptualLumaShared(local_center + int2( 0,  1));
    float br = FetchPerceptualLumaShared(local_center + int2( 1,  1));
    float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    return (gx * gx + gy * gy) * 0.0625;
}

float Scharr3x3Shared(int2 local_center)
{
    float tl = FetchPerceptualLumaShared(local_center + int2(-1, -1));
    float tc = FetchPerceptualLumaShared(local_center + int2( 0, -1));
    float tr = FetchPerceptualLumaShared(local_center + int2( 1, -1));
    float ml = FetchPerceptualLumaShared(local_center + int2(-1,  0));
    float mr = FetchPerceptualLumaShared(local_center + int2( 1,  0));
    float bl = FetchPerceptualLumaShared(local_center + int2(-1,  1));
    float bc = FetchPerceptualLumaShared(local_center + int2( 0,  1));
    float br = FetchPerceptualLumaShared(local_center + int2( 1,  1));
    float gx = (3.0 * tr + 10.0 * mr + 3.0 * br) - (3.0 * tl + 10.0 * ml + 3.0 * bl);
    float gy = (3.0 * bl + 10.0 * bc + 3.0 * br) - (3.0 * tl + 10.0 * tc + 3.0 * tr);
    return (gx * gx + gy * gy) * 0.00390625;
}

float Prewitt3x3Shared(int2 local_center)
{
    float tl = FetchPerceptualLumaShared(local_center + int2(-1, -1));
    float tc = FetchPerceptualLumaShared(local_center + int2( 0, -1));
    float tr = FetchPerceptualLumaShared(local_center + int2( 1, -1));
    float ml = FetchPerceptualLumaShared(local_center + int2(-1,  0));
    float mr = FetchPerceptualLumaShared(local_center + int2( 1,  0));
    float bl = FetchPerceptualLumaShared(local_center + int2(-1,  1));
    float bc = FetchPerceptualLumaShared(local_center + int2( 0,  1));
    float br = FetchPerceptualLumaShared(local_center + int2( 1,  1));
    float gx = (tr + mr + br) - (tl + ml + bl);
    float gy = (bl + bc + br) - (tl + tc + tr);
    return (gx * gx + gy * gy) * 0.111111111;
}

float Sobel5x5Shared(int2 local_center)
{
    float sum_gx = 0.0;
    float sum_gy = 0.0;
    
    [unroll] 
    for (int y = -2; y <= 2; y++) 
    {
        [unroll] 
        for (int x = -2; x <= 2; x++) 
        {
            float luma = FetchPerceptualLumaShared(local_center + int2(x, y));
            int idx = (y + 2) * 5 + (x + 2);
            sum_gx += luma * Sobel5x5_Gx[idx];
            sum_gy += luma * Sobel5x5_Gy[idx];
        }
    }
    return (sum_gx * sum_gx + sum_gy * sum_gy) * 0.00043402778;
}

float LaplacianOfGaussianShared(int2 local_center)
{
    float response = 0.0;
    
    [unroll] 
    for (int y = -2; y <= 2; y++) 
    {
        [unroll] 
        for (int x = -2; x <= 2; x++) 
        {
            float luma = FetchPerceptualLumaShared(local_center + int2(x, y));
            int idx = (y + 2) * 5 + (x + 2);
            response += luma * LoG_Kernel[idx];
        }
    }
    return response * response;
}

float StructureTensorShared(int2 local_center)
{
    float pl[25];
    
    [unroll] 
    for (int pj = -2; pj <= 2; pj++) 
    {
        [unroll] 
        for (int pi = -2; pi <= 2; pi++) 
        {
            pl[(pj + 2) * 5 + (pi + 2)] = FetchPerceptualLumaShared(local_center + int2(pi, pj));
        }
    }

    float Ixx = 0.0, Iyy = 0.0, Ixy = 0.0;

    [unroll] 
    for (int wy = 0; wy < 3; wy++) 
    {
        [unroll] 
        for (int wx = 0; wx < 3; wx++) 
        {
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
    float diff = Ixx - Iyy;
    float disc = TrueSqrt(max(diff * diff + 4.0 * Ixy * Ixy, 0.0));

    float lambda1 = (trace + disc) * 0.5;
    float lambda2 = (trace - disc) * 0.5;
    float coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + FLT_MIN);

    return lambda1 * (1.0 + coherence) * 0.5;
}

float ChromaEdgeShared(int2 local_center)
{
    float4 center_data = gs_LinearData[GS_IDX(local_center.x, local_center.y)];
    float2 center_ab = center_data.gb;

    float ct = saturate((center_data.a - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float center_reliability = ct * ct * (3.0 - 2.0 * ct);

    if (center_reliability <= 0.0) return 0.0;

    float maxChromaDiff = 0.0;

    [unroll] 
    for (int y = -1; y <= 1; y++) 
    {
        [unroll] 
        for (int x = -1; x <= 1; x++) 
        {
            if (x == 0 && y == 0) continue;

            float4 neighbor_data = gs_LinearData[GS_IDX(local_center.x + x, local_center.y + y)];
            float nt = saturate((neighbor_data.a - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
            float neighbor_reliability = nt * nt * (3.0 - 2.0 * nt);
            float chroma_reliability = center_reliability * neighbor_reliability;

            if (chroma_reliability > 0.0) 
            {
                float2 d = center_ab - neighbor_data.gb;
                maxChromaDiff = max(maxChromaDiff, dot(d, d) * chroma_reliability);
            }
        }
    }
    return maxChromaDiff * 12.0;
}

float GetEdgeStrengthShared(int2 local_center, int method)
{
    if (method == 0) return Sobel3x3Shared(local_center);
    if (method == 1) return Scharr3x3Shared(local_center);
    if (method == 2) return Prewitt3x3Shared(local_center);
    if (method == 3) return Sobel5x5Shared(local_center);
    if (method == 4) return LaplacianOfGaussianShared(local_center);
    if (method == 5) return StructureTensorShared(local_center);
    
    return Sobel3x3Shared(local_center);
}

// ==============================================================================
// 9. Bilateral Processing (Compute Shader Híbrido LDS)
// ==============================================================================

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
    else if (mode == 2) metric = PowSafe(max(saturate(range * 0.166666667), FLT_MIN), 1.0 - fVarianceWeight) * PowSafe(max(saturate(var * 0.5), FLT_MIN), fVarianceWeight);
    else                metric = saturate((log2(1.0 + var) * (1.0 + range * 0.1)) * 0.25);

    return base_strength * lerp(1.0, PowSafe(metric, fAdaptiveCurve) * 2.0, fAdaptiveAmount);
}

/**
 * WriteDebugOut: Helper function to accurately write debug data 
 * across both SDR and HDR (PQ/scRGB) output spaces.
 */
void WriteDebugOut(int2 pos, float3 dbg, float alpha)
{
    int activeSpace = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float3 encoded;

    [branch]
    if (activeSpace == 3)
    {
        encoded = PQ_InverseEOTF(dbg * SCRGB_WHITE_NITS);
    }
    else if (activeSpace == 2)
    {
        encoded = dbg;
    }
    else
    {
        encoded = sRGB_OETF(saturate(dbg));
    }

    tex2Dstore(StorageBilateralOut, pos, float4(encoded, alpha));
}

/**
 * BCE_ACCUMULATE Macro
 * 
 * Implicit Contract - Expects the following variables in the outer scope:
 * - float log2_center, inv_2_sigma_s_sq, inv_2_sigma_r_sq, inv_2_sigma_c_sq
 * - float spatial_y
 * - bool use_chroma
 * - float center_chroma_reliability
 * - float2 center_chroma
 * - float2 stats_log, stats_sq, stats_w
 * - float min_log, max_log
 */
#define BCE_ACCUMULATE(n_data, x_coord)                                                                                    \
{                                                                                                                          \
    float _n_log = (n_data).r;                                                                                             \
    float _n_luma = (n_data).a;                                                                                            \
    float _d_luma = log2_center - _n_log;                                                                                  \
    float _exponent = -(float((x_coord) * (x_coord)) * inv_2_sigma_s_sq + spatial_y) - (_d_luma * _d_luma * inv_2_sigma_r_sq); \
    [branch]                                                                                                               \
    if (use_chroma)                                                                                                        \
    {                                                                                                                      \
        float _nt = saturate((_n_luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);                          \
        float _chroma_reliability = center_chroma_reliability * (_nt * _nt * (3.0 - 2.0 * _nt));                           \
        if (_chroma_reliability > 0.0)                                                                                     \
        {                                                                                                                  \
            float2 _d_chroma = center_chroma - (n_data).gb;                                                                \
            _exponent -= (dot(_d_chroma, _d_chroma) * inv_2_sigma_c_sq) * _chroma_reliability;                             \
        }                                                                                                                  \
    }                                                                                                                      \
    if (_exponent > LN_FLT_MIN)                                                                                            \
    {                                                                                                                      \
        float _weight = exp(_exponent);                                                                                    \
        float _val = _n_log * _weight;                                                                                     \
        float _t = stats_log.x + _val;                                                                                     \
        stats_log.y += (abs(stats_log.x) >= abs(_val)) ? ((stats_log.x - _t) + _val) : ((_val - _t) + stats_log.x);        \
        stats_log.x = _t;                                                                                                  \
        _val = _n_log * _n_log * _weight;                                                                                  \
        _t = stats_sq.x + _val;                                                                                            \
        stats_sq.y += (abs(stats_sq.x) >= abs(_val)) ? ((stats_sq.x - _t) + _val) : ((_val - _t) + stats_sq.x);            \
        stats_sq.x = _t;                                                                                                   \
        _t = stats_w.x + _weight;                                                                                          \
        stats_w.y += (abs(stats_w.x) >= abs(_weight)) ? ((stats_w.x - _t) + _weight) : ((_weight - _t) + stats_w.x);       \
        stats_w.x = _t;                                                                                                    \
        min_log = min(min_log, _n_log);                                                                                    \
        max_log = max(max_log, _n_log);                                                                                    \
    }                                                                                                                      \
}

[shader("compute")]
[numthreads(16, 16, 1)]
void CS_BilateralContrast(uint3 id : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    int2 global_pos = int2(id.xy);

    // -------------------------------------------------------------
    // PHASE 1: COOPERATIVE GROUPSHARED (LDS) LOAD
    // -------------------------------------------------------------
    int2 base_pos = int2(gid.xy) * 16 - int2(LDS_HALO, LDS_HALO);

    [unroll]
    for (int i = 0; i < 2; ++i) 
    {
        [unroll]
        for (int j = 0; j < 2; ++j) 
        {
            int lx = tid.x + i * 16;
            int ly = tid.y + j * 16;
            int2 fetch_pos = base_pos + int2(lx, ly);
            
            fetch_pos = max(0, min(int2(BUFFER_WIDTH, BUFFER_HEIGHT) - 1, fetch_pos));
            gs_LinearData[GS_IDX(lx, ly)] = tex2Dfetch(SamplerLinearData, fetch_pos);
        }
    }
    barrier();

    // -------------------------------------------------------------
    // PHASE 2: SETUP & EARLY OUTS
    // -------------------------------------------------------------
    if (global_pos.x >= BUFFER_WIDTH || global_pos.y >= BUFFER_HEIGHT) return;

    float4 src = tex2Dfetch(SamplerBackBuffer, global_pos);

    if (fStrength <= 0.0 && iDebugMode == 0) 
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

    int2 local_center = int2(tid.xy) + int2(LDS_HALO, LDS_HALO);
    float4 center_data = gs_LinearData[GS_IDX(local_center.x, local_center.y)];

    float log2_center = center_data.r;
    float luma_lin    = center_data.a;
    float whitePt     = GetResolvedWhitePoint();
    float3 color_lin  = DecodeToLinear(src.rgb);

    if (iDebugMode == 0 && luma_lin <= FLT_MIN) 
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

    // -------------------------------------------------------------
    // PHASE 3: EDGE DETECTION (100% via LDS)
    // -------------------------------------------------------------
    int base_radius = (iQualityPreset == 1) ? 24 : iRadius;
    float sigma_s = (iQualityPreset == 1) ? 12.0 : fSigmaSpatial;
    int radius = base_radius;

    if (bAdaptiveRadius && base_radius > 2) 
    {
        float edge = GetEdgeStrengthShared(local_center, iEdgeDetectionMethod);
        
        if (bChromaAwareBilateral && fChromaEdgeStrength > 0.0) 
        {
            float chromaEdge = ChromaEdgeShared(local_center);
            edge = lerp(edge, max(edge, chromaEdge), fChromaEdgeStrength);
        }
        
        float scale = TrueSmoothstep(0.0, 1.0, edge * (fGradientSensitivity * 0.01));
        float factor = lerp(1.0, lerp(1.0, 0.15, scale), fAdaptiveRadiusStrength);
        int sigma_max = (int)(sigma_s * 3.0 + 0.5);
        radius = clamp(min((int)(base_radius * factor + 0.5), sigma_max), 1, base_radius);
    }

    // Fast-path for isolated Debug modes
    if (iDebugMode == 5) 
    {
        float3 dbg = lerp(float3(0, 0, 1), float3(1, 0, 0), float(radius) / float(base_radius));
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 6) 
    {
        float e = GetEdgeStrengthShared(local_center, iEdgeDetectionMethod);
        WriteDebugOut(global_pos, float3(e, e, e) * 10.0, src.a);
        return;
    }
    if (iDebugMode == 8) 
    {
        float c = ChromaEdgeShared(local_center);
        WriteDebugOut(global_pos, float3(c, c, c) * 5.0, src.a);
        return;
    }
    if (iDebugMode == 10) 
    {
        float3 dbg = GetZoneColor(GetZone(luma_lin / whitePt));
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 11) 
    {
        float3 dbg = (GetMinComponent(color_lin) < 0.0) ? float3(1, 0, 1) : float3(0, 0.1, 0);
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 12) 
    {
        float norm = luma_lin / max(whitePt, FLT_MIN);
        float stops = log2(max(abs(norm), FLT_MIN));
        float t = saturate((stops + 6.0) / 8.0);
        float3 dbg = (norm < 0.0) ? float3(0, t, 0) : float3(t, 0, 0);
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 7) 
    {
        float3 dbg = (luma_lin <= FLT_MIN) ? float3(1, 0, 1) : float3(0, 0, 0);
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }

    // -------------------------------------------------------------
    // PHASE 4: THE BILATERAL LOOP (HYBRID: L1 + VRAM)
    // -------------------------------------------------------------
    float inv_2_sigma_s_sq = 0.5 / (sigma_s * sigma_s);
    float inv_2_sigma_r_sq = 0.5 / (fSigmaRange * fSigmaRange);
    float inv_2_sigma_c_sq = 0.5 / (fSigmaChroma * fSigmaChroma);

    int cutoff_int  = (int)TrueSqrt(NEG_LN_SPATIAL_CUTOFF / inv_2_sigma_s_sq);
    int safe_radius = min(cutoff_int + 1, radius);
    int max_r       = min(safe_radius, MAX_LOOP_RADIUS);
    float r_limit_sq = float(max_r * max_r);

    float2 center_chroma = center_data.gb;
    float center_chroma_reliability = 0.0;
    bool use_chroma = false;

    if (bChromaAwareBilateral) 
    {
        float ct = saturate((luma_lin - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
        center_chroma_reliability = ct * ct * (3.0 - 2.0 * ct);
        use_chroma = (center_chroma_reliability > 0.0);
    }

    float2 stats_log = 0.0;
    float2 stats_sq  = 0.0;
    float2 stats_w   = 0.0;
    float min_log = log2_center;
    float max_log = log2_center;

    int y_start   = max(-max_r, -global_pos.y);
    int y_end     = min( max_r, (BUFFER_HEIGHT - 1) - global_pos.y);
    int x_min_off = -global_pos.x;
    int x_max_off = (BUFFER_WIDTH - 1) - global_pos.x;

    int r_lds = min(max_r, LDS_RADIUS);

    // Phase 4A: LDS-only inner loop (confined to LDS halo)
    [loop]
    for (int y = -r_lds; y <= r_lds; ++y)
    {
        float y_f = float(y);
        float spatial_y = y_f * y_f * inv_2_sigma_s_sq;
        
        int x_limit_circ = (int)TrueSqrt(max(0.0, r_limit_sq - y_f * y_f));
        int x_start = max(-x_limit_circ, -r_lds);
        int x_end   = min( x_limit_circ,  r_lds);
        int local_y = local_center.y + y;

        [loop]
        for (int x = x_start; x <= x_end; ++x)
        {
            int local_x = local_center.x + x;
            float4 n_data = gs_LinearData[GS_IDX(local_x, local_y)];
            BCE_ACCUMULATE(n_data, x);
        }
    }

    // Phase 4B: VRAM fallback for outer ring (radius > halo)
    [branch]
    if (max_r > LDS_RADIUS)
    {
        [loop]
        for (int y = y_start; y <= y_end; ++y)
        {
            float y_f = float(y);
            float spatial_y = y_f * y_f * inv_2_sigma_s_sq;
            int x_limit_circ = (int)TrueSqrt(max(0.0, r_limit_sq - y_f * y_f));

            int x_start = max(-x_limit_circ, x_min_off);
            int x_end   = min( x_limit_circ, x_max_off);
            int abs_y = abs(y);

            if (abs_y > LDS_RADIUS)
            {
                // Entire row is outside vertical LDS bounds
                [loop]
                for (int x = x_start; x <= x_end; ++x)
                {
                    float4 n_data = tex2Dfetch(SamplerLinearData, global_pos + int2(x, y));
                    BCE_ACCUMULATE(n_data, x);
                }
            }
            else
            {
                // Only left and right wings are outside horizontal LDS bounds
                int left_end = min(x_end, -LDS_RADIUS - 1);
                [loop]
                for (int x = x_start; x <= left_end; ++x)
                {
                    float4 n_data = tex2Dfetch(SamplerLinearData, global_pos + int2(x, y));
                    BCE_ACCUMULATE(n_data, x);
                }
                
                int right_start = max(x_start, LDS_RADIUS + 1);
                [loop]
                for (int x = right_start; x <= x_end; ++x)
                {
                    float4 n_data = tex2Dfetch(SamplerLinearData, global_pos + int2(x, y));
                    BCE_ACCUMULATE(n_data, x);
                }
            }
        }
    }

    // -------------------------------------------------------------
    // PHASE 5: FINAL EVALUATION & WRITE
    // -------------------------------------------------------------
    float total_w = stats_w.x + stats_w.y;
    
    if (total_w < FLT_MIN) 
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

    float total_log = stats_log.x + stats_log.y;
    float total_sq  = stats_sq.x  + stats_sq.y;
    float blurred   = total_log / total_w;
    float diff      = log2_center - blurred;

    float strength = fStrength;
    if (bAdaptiveStrength)
    {
        strength = CalculateAdaptiveStrength(total_log, total_sq, total_w, min_log, max_log, fStrength, iAdaptiveMode);
    }

    float norm_luma = luma_lin / whitePt;
    float minCompNorm = GetMinComponent(color_lin) / whitePt;
    
    strength *= GetZoneProtection(norm_luma, minCompNorm, fShadowProtection, fMidtoneProtection, fHighlightProtection, fNegativeProtection);

    if (abs(strength) < FLT_MIN) 
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

    float enhanced_log = log2_center + strength * diff;
    float enhanced_luma = exp2(enhanced_log);
    float ratio = clamp(enhanced_luma / max(luma_lin, FLT_MIN), RATIO_MIN, RATIO_MAX);
    float3 final_color = color_lin * ratio;

    if (any(IsNan3(final_color)) || any(IsInf3(final_color))) 
    {
        final_color = color_lin;
    }

    int activeSpace = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;

    // Post-loop Debugs
    if (iDebugMode == 1) 
    {
        float3 dbg = saturate(log2(total_w + 1.0) * 0.1).xxx;
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 2) 
    {
        float m = blurred;
        float v = max(0.0, (total_sq / total_w) - m * m);
        WriteDebugOut(global_pos, float3(v * 2.0, v, 0.0), src.a);
        return;
    }
    if (iDebugMode == 3) 
    {
        float3 dbg = float3((max_log - min_log) * 0.2, 0, 0);
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 4) 
    {
        float3 dbg = lerp(float3(0, 0, 1), float3(1, 0, 0), saturate(abs(diff) * strength * 2.0));
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 9) 
    {
        float m = total_log / total_w;
        float v = max(0.0, (total_sq / total_w) - m * m);
        float r = max_log - min_log;
        float e = log2(1.0 + v) * (1.0 + r * 0.1);
        WriteDebugOut(global_pos, float3(e * 0.25, e * 0.125, 0.0), src.a);
        return;
    }

    float3 encoded = EncodeFromLinear(final_color);
    if (activeSpace <= 1) 
    {
        encoded = saturate(encoded);
    }

    tex2Dstore(StorageBilateralOut, global_pos, float4(encoded, src.a));
}

// ==============================================================================
// 10. Output Blit & Technique
// ==============================================================================

void PS_OutputToScreen(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    fragColor = tex2D(SamplerBilateralOut, texcoord);
}

technique BilateralContrast_Reference <
    ui_label = "Bilateral Contrast v8.5.0 (Compute LDS Edition)";
    ui_tooltip = "MASTERING QUALITY - Compute Shader LDS Optimized\n\n"
                 "v8.5.0 Changes:\n"
                 "- Compute Shader path with groupshared LDS tile cache (32x32)\n"
                 "- Loop fission: LDS inner tile + VRAM outer fallback\n"
                 "- All edge detection and chroma analysis via LDS\n"
                 "- Aligned: PQ debug encoding uses 80 nits reference\n\n"
                 "Requires: DirectX 11+, OpenGL 4.3+, or Vulkan\n\n"
                 "Companion shader: Photoreal HDR V5.9.7-r6+";
>
{
    pass PreCompute
    {
        VertexShader      = PostProcessVS;
        PixelShader       = PS_PrePass;
        RenderTarget      = TexLinearData;
        VertexCount       = 3;
        PrimitiveTopology = TRIANGLELIST;
    }
    
    pass BilateralCompute
    {
        ComputeShader     = CS_BilateralContrast;
        DispatchSizeX     = (BUFFER_WIDTH + 15) / 16;
        DispatchSizeY     = (BUFFER_HEIGHT + 15) / 16;
    }
    
    pass Output
    {
        VertexShader      = PostProcessVS;
        PixelShader       = PS_OutputToScreen;
        VertexCount       = 3;
        PrimitiveTopology = TRIANGLELIST;
    }
}