/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement - COMPUTE LDS EDITION (OPTIMIZED)
 *
 * Design Philosophy: PRECISION AND QUALITY OR IMAGE QUALITY OVER PERFORMANCE
 * - True IEEE 754 Math (No fast intrinsics or approximations)
 * - Exact IEC/SMPTE Standard Constants
 * - Enforced Bit-Exact Neutrality Logic (verified passthrough path)
 * - Pre-computed High-Precision Kernels
 * - True Stop-Domain HDR Processing
 * - ICtCp (Rec. ITU-R BT.2100/BT.2124) Perceptual Chromaticity Processing
 * - Non-Riemannian-Inspired Perceptual Color Metric (Bujack et al., PNAS 2022)
 *   Applied to detail band extraction and chroma edge detection to prevent haloing.
 * - Compute Shader with groupshared LDS tile cache
 * - Planar LDS Layout Optimization (conflict-free column access)
 *
 * Version: 8.7.9 (Sign-Preserving Color Science Edition)
 * - Changed: HLG_EOTF explicitly evaluates sign(x) * (x * x) / 3.0.
 * - Changed: Removed unnecessary max(x, 0.0) across all transfer functions;
 *   replaced with symmetric sign preservation to protect wide-gamut negative RGB.
 * - Added: ui_category_toggle annotations per REFERENCE.md specification;
 *   toggling feature booleans automatically folds/unfolds the entire category.
 * - Added: Zero-cost __RESHADE_PERFORMANCE_MODE__ debug stripping.
 * - Retained: Gaussian bilateral range kernel to eliminate heavy-tailed Cauchy
 *   light leakage, eradicating specular white points and dark edge halos.
 * - Retained: Bujack diminishing-returns detail band saturation with safe
 *   PowNonNegPreserveZero to prevent driver NaNs at diff == 0.
 * - Retained: Direct stop-domain ratio scaling exp2(strength * compressed_diff);
 *   eliminates FP16 prepass quantization drift and saves an exp2 and div per pixel.
 * - Retained: Corrected Laplacian of Gaussian normalization constant 0.00390625 (1/256).
 *
 * Requires: DirectX 11+, OpenGL 4.3+, or Vulkan
 * Conforms to: ReShade FX Reference Specification (crosire/reshade-shaders/REFERENCE.md)
 * Tested against: ReShade 6.8.0 (__RESHADE__ 60800)
 *
 * Author: startuga
 * Formatter: Strict Opinionated Style (Allman, 4-space, Aligned Macros)
 */

#include "ReShade.fxh"

// ==============================================================================
// 0. Compilation Guard & Pre-Processor Configuration
// ==============================================================================

#if __RESHADE__ < 40800
    #error "Bilateral Contrast requires ReShade 4.8.0 or newer for Compute Shader support."
#endif

// Set to 1: True IEEE 754 precision (Mastering Standard - No precision loss)
// Set to 0: 16-bit Float (Faster, uses 50% less VRAM, minor sub-pixel precision loss)
// Note: Visible in ReShade UI Preprocessor Definitions dialogue (>= 8 characters).
#ifndef PREPASS_USE_RGBA32F
    #define PREPASS_USE_RGBA32F 0
#endif

#if PREPASS_USE_RGBA32F
    #define PREPASS_FORMAT RGBA32F
#else
    #define PREPASS_FORMAT RGBA16F
#endif

// Set to 1 on devices that only provide the Vulkan minimum of 16384 bytes of
// groupshared memory. Reduces the LDS footprint from 16.9 KB to exactly 16 KB by
// removing the bank-conflict padding.
#ifndef BCE_COMPAT_VULKAN_MIN_LDS
    #define BCE_COMPAT_VULKAN_MIN_LDS 0
#endif

#if BCE_COMPAT_VULKAN_MIN_LDS
    #define LDS_STRIDE 32 // 4 x 1024 x 4 B = 16384 B exactly (Vulkan spec minimum)
#else
    #define LDS_STRIDE 33 // (y*33 + x) mod 32 = (y+x) mod 32: conflict-free columns
#endif

// ==============================================================================
// 1. High-Precision Constants & Color Science Definitions
// ==============================================================================

static const float BCE_FLT_MIN             = 1.175494351e-38;
static const float BCE_LN_FLT_MIN          = -87.33654475;
static const float NEG_LN_SPATIAL_CUTOFF   = 9.210340372;

static const int MAX_LOOP_RADIUS           = 32;
static const int LDS_TILE_SIZE             = 32;
static const int LDS_HALO                  = 8;
static const int LDS_RADIUS                = LDS_HALO;

static const float RATIO_MIN               = 0.0001;
static const float RATIO_MAX               = 10000.0;

// Chroma reliability fade-in, expressed in NORMALIZED luma (fraction of active white point).
static const float BCE_CHROMA_REL_START    = 4.8828125e-4;      // 2^-11 of active white
static const float BCE_CHROMA_REL_FULL     = 1.953125e-3;       // 2^-9  of active white
static const float BCE_INV_CHROMA_REL_SPAN = 2048.0 / 3.0;      // 1 / (FULL - START), exact in binary
static const float EDGE_LUMA_FLOOR         = 1e-4;
static const float LOG2_EDGE_LUMA_FLOOR    = -13.2877123795;

// Neutral passthrough: |delta log2| * |strength| below this cannot perturb an 8-bit output
static const float BCE_NEUTRAL_LOG2_EPS    = 1e-7;

// Linear conditioning for ICtCp chroma in bilateral accumulator
static const float BCE_CHROMA_CONDITIONING_ACC = 7.0710678; // 5*sqrt(2)
static const float BCE_CHROMA_EDGE_GAIN        = 12.0;
static const float BCE_CHROMA_CONDITIONING     = 100.0;

static const float SRGB_THRESHOLD_EOTF     = 0.04045;
static const float SRGB_THRESHOLD_OETF     = (0.04045 / 12.92);

static const float3 Luma709                = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020               = float3(0.2627, 0.6780, 0.0593);

// Standard Rec.709 to Rec.2020 Linear Transformation Matrix
static const float3x3 RGB709_to_2020 = float3x3(
    0.6274040, 0.3292830, 0.0433130,
    0.0690970, 0.9195440, 0.0113590,
    0.0163910, 0.0880130, 0.8955960
);

// ITU-R BT.2100 / BT.2124 Rec.2020 to HPE LMS Matrix
static const float3x3 RGB_to_LMS = float3x3(
    1688.0 / 4096.0, 2146.0 / 4096.0,  262.0 / 4096.0,
     683.0 / 4096.0, 2951.0 / 4096.0,  462.0 / 4096.0,
      99.0 / 4096.0,  309.0 / 4096.0, 3688.0 / 4096.0
);

// ITU-R BT.2100 / BT.2124 LMS' to ICtCp Matrix
static const float3x3 LMS_to_ICtCp = float3x3(
    0.5,            0.5,             0.0,
    1.61376953125, -3.323486328125,  1.709716796875,
    4.378173828125, -4.24560546875,  -0.132568359375
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

// pooled = true per REFERENCE.md: re-uses texture memory across effects
texture2D TexLinearData < pooled = true; > { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = PREPASS_FORMAT; };
sampler2D SamplerLinearData
{
    Texture   = TexLinearData;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

texture2D TexBilateralOut < pooled = true; > { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = PREPASS_FORMAT; };
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

#ifndef BUFFER_COLOR_BIT_DEPTH
    #define BUFFER_COLOR_BIT_DEPTH 8
#endif

// ==============================================================================
// 3. UI Parameters (Modern REFERENCE.md Annotations)
// ==============================================================================

uniform float fStrength <
    ui_type = "slider";
    ui_label = "Contrast Strength";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.001;
    ui_units = "";
    ui_category = "Core Settings";
> = 3.2;

uniform float fShadowProtection <
    ui_type = "slider";
    ui_label = "Shadow Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0.35;

uniform float fMidtoneProtection <
    ui_type = "slider";
    ui_label = "Midtone Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0.05;

uniform float fHighlightProtection <
    ui_type = "slider";
    ui_label = "Highlight Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0.22;

uniform float fZoneWhitePoint <
    ui_type = "slider";
    ui_label = "Zone White Point (Nits)";
    ui_min = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_units = "nits";
    ui_tooltip = "Only applies when using HDR/scRGB Color Space overrides.\nConforms to ITU-R BT.2408 reference paper white.";
    ui_category = "Protection Zones";
> = 203.0;

uniform float fNegativeProtection <
    ui_type = "slider";
    ui_label = "Negative Value Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_tooltip = "Protects out-of-gamut negative RGB values created or preserved by ratio scaling.\nWorks in all color spaces (SDR and HDR).";
    ui_category = "Protection Zones";
> = 0.25;

uniform bool bAdaptiveStrength <
    ui_label = "Enable Adaptive Strength";
    ui_category = "Adaptive Processing";
    ui_category_toggle = true;
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
> = 1.0;

uniform int iRadius <
    ui_type = "slider";
    ui_label = "Filter Radius";
    ui_min = 1; ui_max = 32;
    ui_units = "px";
    ui_category = "Filter Parameters";
> = 14;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_min = 0.1; ui_max = 32.0; ui_step = 0.01;
    ui_units = "px";
    ui_category = "Filter Parameters";
> = 4.70;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Range Sigma (Stops)";
    ui_min = 0.01; ui_max = 4.0; ui_step = 0.001;
    ui_units = "stops";
    ui_category = "Filter Parameters";
> = 0.35;

uniform float fSigmaChroma <
    ui_type = "slider";
    ui_label = "Chroma Sigma";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.001;
    ui_tooltip = "Controls filter sensitivity to ICtCp chromaticity differences.\n"
                 "Note: Setting PREPASS_USE_RGBA32F = 1 in the preprocessor is recommended\n"
                 "to resolve sub-pixel chroma banding in highly saturated gradients.";
    ui_category = "Filter Parameters";
> = 0.22;

uniform bool bChromaAwareBilateral <
    ui_label = "Chroma-Aware Filtering";
    ui_category = "Filter Parameters";
> = true;

uniform bool bNonRiemannianPerception <
    ui_label = "Enable Non-Riemannian Metric";
    ui_tooltip = "Applies a log-compressed perceptual metric ln(1 + d^g) / g to the\n"
                 "extracted contrast detail band and ICtCp chroma edge detection.\n"
                 "Inspired by Bujack et al., 'The non-Riemannian nature of perceptual color space',\n"
                 "PNAS 2022. Models the HVS's diminishing sensitivity to large differences,\n"
                 "preventing specular blowout, halos, and white needle-point fireflies.";
    ui_category = "Non-Riemannian Perception";
    ui_category_toggle = true;
> = true;

uniform float fDiminishingReturnsExponent <
    ui_type = "slider";
    ui_label = "Perceptual Saturation Exponent";
    ui_min = 0.05; ui_max = 2.00; ui_step = 0.01;
    ui_tooltip = "Stevens' Power Law exponent (gamma) inside the compressed metric.\n"
                 "< 1.0: strong diminishing returns (large differences compress harder).\n"
                 "1.0:   soft-log baseline, ln(1 + d).\n"
                 "> 1.0: transitions back toward linear scaling.";
    ui_category = "Non-Riemannian Perception";
> = 0.82;

uniform bool bAdaptiveRadius <
    ui_label = "Enable Adaptive Radius";
    ui_category = "Adaptive Radius";
    ui_category_toggle = true;
> = true;

uniform float fAdaptiveRadiusStrength <
    ui_type = "slider";
    ui_label = "Adaptive Radius Strength";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Adaptive Radius";
> = 0.75;

uniform float fChromaEdgeStrength <
    ui_type = "slider";
    ui_label = "Chroma Edge Influence";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Controls how strongly chroma edges reduce the filter radius.\n0.0 = Luma only. 1.0 = Max(Luma, ICtCp Chroma).";
    ui_category = "Adaptive Radius";
> = 0.50;

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
    ui_category_closed = true;
> = 175.0;

uniform float fVarianceWeight <
    ui_type = "slider";
    ui_label = "Variance Weight";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Advanced Tuning";
    ui_category_closed = true;
> = 0.65;

uniform int iColorSpaceOverride <
    ui_type = "combo";
    ui_label = "Color Space Override";
    ui_items = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0HLG (HDR)\0";
    ui_tooltip = "Selects the EOTF/OETF used for decoding.\n'Auto' uses BUFFER_COLOR_SPACE definition.\nscRGB assumes 1.0 = 80 nits.\n\n"
                 "HLG highlights above nominal 1000 nits require an FP16 (16-bit) backbuffer;\n"
                 "on 8/10-bit UNORM backbuffers they are clamped to signal 1.0.";
    ui_category = "System";
> = 0;

// Compiles out debug UI when ReShade is in Performance Mode
#if !defined(__RESHADE_PERFORMANCE_MODE__) || !__RESHADE_PERFORMANCE_MODE__
uniform int iDebugMode <
    ui_type = "combo";
    ui_label = "Debug Visualization";
    ui_items = "Off\0Weights\0Variance\0Dynamic Range\0Enhancement Map\0Adaptive Radius\0Edge Detection\0Black Pixels\0Chroma Edges\0Entropy\0Zone Map\0Negative Values\0Signed Luminance\0";
    ui_category = "Debug";
    ui_category_closed = true;
> = 0;
#endif

// ==============================================================================
// 4. True Math Utilities (Bit-Exact Safety)
// ==============================================================================

float TrueSqrt(float x)
{
    return sqrt(max(x, 0.0));
}

float PowSafe(float base, float exponent)
{
    float safe_base = max(abs(base), BCE_FLT_MIN);
    float result = pow(safe_base, exponent);
    return (exponent < 0.0) ? min(result, 1e38) : result;
}

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

float GetMinComponent(float3 lin)
{
    return min(min(lin.r, lin.g), lin.b);
}

float TrueSmoothstep(float edge0, float edge1, float x)
{
    float diff = edge1 - edge0;
    if (abs(diff) < BCE_FLT_MIN) return step(edge0, x);
    float t = saturate((x - edge0) / diff);
    return t * t * (3.0 - 2.0 * t);
}

bool3 IsNan3(float3 v) { return isnan(v); }
bool3 IsInf3(float3 v) { return isinf(v); }

// ==============================================================================
// 5. Color Science (Exact Standard Definitions - Sign Preserving)
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
    float3 abs_N = saturate(abs(N));
    float3 Np = PowNonNegPreserveZero3(abs_N, 1.0 / PQ_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, BCE_FLT_MIN);

    return sign(N) * PowNonNegPreserveZero3(num / den, 1.0 / PQ_M1) * PQ_PEAK_LUMINANCE;
}

float3 PQ_InverseEOTF(float3 L)
{
    float3 abs_L = clamp(abs(L), 0.0, PQ_PEAK_LUMINANCE);
    float3 Lp = PowNonNegPreserveZero3(abs_L / PQ_PEAK_LUMINANCE, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;

    return sign(L) * saturate(PowNonNegPreserveZero3(num / den, PQ_M2));
}

float3 HLG_EOTF(float3 x)
{
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.55991073;

    float3 abs_x = abs(x);
    float3 r;
    r.r = (abs_x.r <= 0.5) ? sign(x.r) * (x.r * x.r) / 3.0 : sign(x.r) * ((exp((abs_x.r - c) / a) + b) / 12.0);
    r.g = (abs_x.g <= 0.5) ? sign(x.g) * (x.g * x.g) / 3.0 : sign(x.g) * ((exp((abs_x.g - c) / a) + b) / 12.0);
    r.b = (abs_x.b <= 0.5) ? sign(x.b) * (x.b * x.b) / 3.0 : sign(x.b) * ((exp((abs_x.b - c) / a) + b) / 12.0);

    return r * 1000.0;
}

float3 HLG_OETF(float3 x)
{
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.55991073;

    float3 abs_x = abs(x);
    float3 E = abs_x / 1000.0;
    float3 r;
    r.r = (E.r <= 1.0 / 12.0) ? sqrt(3.0 * E.r) : a * log(max(12.0 * E.r - b, BCE_FLT_MIN)) + c;
    r.g = (E.g <= 1.0 / 12.0) ? sqrt(3.0 * E.g) : a * log(max(12.0 * E.g - b, BCE_FLT_MIN)) + c;
    r.b = (E.b <= 1.0 / 12.0) ? sqrt(3.0 * E.b) : a * log(max(12.0 * E.b - b, BCE_FLT_MIN)) + c;

#if BUFFER_COLOR_BIT_DEPTH <= 10
    r = min(r, 1.0.xxx);
#endif

    return sign(x) * r;
}

float3 DecodeToLinear(float3 encoded)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;

    [branch]
    if (space == 4)
    {
        return HLG_EOTF(encoded);
    }

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
    if (space == 4)
    {
        return HLG_OETF(lin);
    }

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

float2 GetICtCpChroma(float3 linearRGB, int activeSpace)
{
    float3 rgb_2020 = (activeSpace >= 3) ? linearRGB : mul(RGB709_to_2020, linearRGB);
    float3 lms = mul(RGB_to_LMS, rgb_2020);
    float3 lms_p = PQ_InverseEOTF(lms);
    float3 ictcp = mul(LMS_to_ICtCp, lms_p);

    float I = max(ictcp.x, BCE_FLT_MIN);
    return ictcp.yz / I;
}

float GetChromaReliability(float luma_nits, float inv_white)
{
    float nl = luma_nits * inv_white;
    float t = saturate((nl - BCE_CHROMA_REL_START) * BCE_INV_CHROMA_REL_SPAN);
    return t * t * (3.0 - 2.0 * t);
}

float ApplyNonRiemannianMetric(float dist_sq, float kappa)
{
    if (!bNonRiemannianPerception) return dist_sq;

    float gamma = clamp(fDiminishingReturnsExponent, 0.05, 2.0);
    float dp = PowNonNegPreserveZero(dist_sq * kappa, gamma);
    return log(max(1.0 + dp, 1.0)) / (gamma * kappa);
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
    if (shadowProt + midProt + highProt + negProt < BCE_FLT_MIN) return 1.0;

    float negW = 1.0 - TrueSmoothstep(-0.001, 0.0, minCompNorm);
    float s = log2(max(nl, BCE_FLT_MIN));

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

    float3 color_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, pos).rgb);
    bool is_invalid = any(IsNan3(color_lin)) || any(IsInf3(color_lin));
    color_lin = is_invalid ? 0.0.xxx : color_lin;

    float luma_lin = GetLuminanceCS(color_lin);

    float safe_luma = max(luma_lin, BCE_FLT_MIN);
    float log2_luma = log2(safe_luma);

    float white_pt = GetResolvedWhitePoint();

    float2 chroma = float2(0.0, 0.0);
    if (bChromaAwareBilateral && luma_lin > BCE_CHROMA_REL_START * white_pt)
    {
        chroma = GetICtCpChroma(color_lin, space);
    }

    outData = float4(log2_luma, chroma.x, chroma.y, luma_lin);
}

// ==============================================================================
// 8. Analysis & Edge Detection (Planar LDS Optimized)
// ==============================================================================

groupshared float gs_Log2Luma[LDS_TILE_SIZE * LDS_STRIDE];
groupshared float gs_ChromaA[LDS_TILE_SIZE * LDS_STRIDE];
groupshared float gs_ChromaB[LDS_TILE_SIZE * LDS_STRIDE];
groupshared float gs_LumaLin[LDS_TILE_SIZE * LDS_STRIDE];

#define GS_IDX(x, y) ((y) * LDS_STRIDE + (x))

int2 ClampGlobalToTile(int2 gp, int2 base_pos)
{
    int2 gc = clamp(gp, int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
    return gc - base_pos;
}

float FetchPerceptualLumaShared(int2 local_pos)
{
    float log2_luma = gs_Log2Luma[GS_IDX(local_pos.x, local_pos.y)];
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
    return response * response * 0.00390625; // Fixed normalization: 1/256
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
    float coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + BCE_FLT_MIN);

    return (lambda1 * (1.0 + coherence) * 0.5) * 0.08333333;
}

float ChromaEdgeShared(int2 local_center, float inv_white)
{
    int center_idx = GS_IDX(local_center.x, local_center.y);
    float2 center_chroma = float2(gs_ChromaA[center_idx], gs_ChromaB[center_idx]);
    float center_reliability = GetChromaReliability(gs_LumaLin[center_idx], inv_white);

    float max_chroma_diff = 0.0;

    [unroll]
    for (int y = -1; y <= 1; y++)
    {
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            if (x == 0 && y == 0) continue;

            int neighbor_idx = GS_IDX(local_center.x + x, local_center.y + y);
            float neighbor_reliability = GetChromaReliability(gs_LumaLin[neighbor_idx], inv_white);

            float2 d = center_chroma - float2(gs_ChromaA[neighbor_idx], gs_ChromaB[neighbor_idx]);
            float dist_sq = dot(d, d);

            float metric = ApplyNonRiemannianMetric(dist_sq, BCE_CHROMA_CONDITIONING);
            max_chroma_diff = max(max_chroma_diff, metric * max(center_reliability, neighbor_reliability));
        }
    }
    return max_chroma_diff * BCE_CHROMA_EDGE_GAIN;
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
// 9. Bilateral Processing (Compute Shader Hybrid LDS)
// ==============================================================================

float CalculateAdaptiveStrength(float sum_log, float sum_diff_sq, float sum_weight, float min_log, float max_log, float log2_center, float base_strength, int mode)
{
    if (sum_weight < BCE_FLT_MIN) return base_strength;

    float inv_weight = 1.0 / sum_weight;
    float range = max_log - min_log;
    float mean_diff = (sum_log * inv_weight) - log2_center;
    float var = max(0.0, sum_diff_sq * inv_weight - mean_diff * mean_diff);
    float metric;

    if (mode == 0)      metric = saturate(range * 0.166666667);
    else if (mode == 1) metric = saturate(var * 0.5);
    else if (mode == 2) metric = PowSafe(max(saturate(range * 0.166666667), BCE_FLT_MIN), 1.0 - fVarianceWeight) * PowSafe(max(saturate(var * 0.5), BCE_FLT_MIN), fVarianceWeight);
    else                metric = saturate((log2(1.0 + var) * (1.0 + range * 0.1)) * 0.25);

    return base_strength * lerp(1.0, PowSafe(metric, fAdaptiveCurve) * 2.0, fAdaptiveAmount);
}

#if !defined(__RESHADE_PERFORMANCE_MODE__) || !__RESHADE_PERFORMANCE_MODE__
void WriteDebugOut(int2 pos, float3 dbg, float alpha)
{
    int activeSpace = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float whitePt = GetResolvedWhitePoint();
    float3 encoded;

    [branch]
    if (activeSpace == 4)
    {
        encoded = HLG_OETF(dbg * whitePt);
    }
    else if (activeSpace == 3)
    {
        encoded = PQ_InverseEOTF(dbg * whitePt);
    }
    else if (activeSpace == 2)
    {
        encoded = dbg * (whitePt / SCRGB_WHITE_NITS);
    }
    else
    {
        encoded = sRGB_OETF(saturate(dbg));
    }

    tex2Dstore(StorageBilateralOut, pos, float4(encoded, alpha));
}
#endif

// Bilateral accumulator:
// Range kernel is strictly Gaussian exponential to isolate specular highlights and avoid
// Cauchy heavy-tailed light leakage (which causes dark edge halos and white point fireflies).
#define BCE_ACCUMULATE(n_data, x_coord)                                                                                        \
{                                                                                                                              \
    float _n_log = (n_data).r;                                                                                                 \
    float _n_luma = (n_data).a;                                                                                                \
    float _d_luma = log2_center - _n_log;                                                                                      \
    float _dist_sq = _d_luma * _d_luma * inv_2_sigma_r_sq;                                                                     \
    [branch]                                                                                                                   \
    if (bChromaAwareBilateral)                                                                                                 \
    {                                                                                                                          \
        float _dcx = center_chroma.x - (n_data).g;                                                                             \
        float _dcy = center_chroma.y - (n_data).b;                                                                             \
        float _d_chroma_sq = _dcx * _dcx + _dcy * _dcy;                                                                        \
        float _chroma_reliability = center_chroma_reliability * GetChromaReliability(_n_luma, inv_white);                       \
        _dist_sq += _d_chroma_sq * (BCE_CHROMA_CONDITIONING_ACC * BCE_CHROMA_CONDITIONING_ACC)                                  \
                    * _chroma_reliability * inv_2_sigma_c_sq;                                                                  \
    }                                                                                                                          \
    float _exponent = -(float((x_coord) * (x_coord)) * inv_2_sigma_s_sq + spatial_y) - _dist_sq;                                \
    if (_exponent > BCE_LN_FLT_MIN)                                                                                            \
    {                                                                                                                          \
        float _weight = exp(_exponent);                                                                                        \
        float _val = _n_log * _weight;                                                                                         \
        float _t = stats_log.x + _val;                                                                                         \
        stats_log.y += (abs(stats_log.x) >= abs(_val)) ? ((stats_log.x - _t) + _val) : ((_val - _t) + stats_log.x);             \
        stats_log.x = _t;                                                                                                      \
        float _d_center = _n_log - log2_center;                                                                                \
        float _val_sq = _d_center * _d_center * _weight;                                                                       \
        _t = stats_sq.x + _val_sq;                                                                                             \
        stats_sq.y += (abs(stats_sq.x) >= abs(_val_sq)) ? ((stats_sq.x - _t) + _val_sq) : ((_val_sq - _t) + stats_sq.x);        \
        stats_sq.x = _t;                                                                                                       \
        _t = stats_w.x + _weight;                                                                                              \
        stats_w.y += (abs(stats_w.x) >= abs(_weight)) ? ((stats_w.x - _t) + _weight) : ((_weight - _t) + stats_w.x);            \
        stats_w.x = _t;                                                                                                        \
        min_log = min(min_log, _n_log);                                                                                        \
        max_log = max(max_log, _n_log);                                                                                        \
    }                                                                                                                          \
}

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

            fetch_pos = max(int2(0, 0), min(int2(BUFFER_WIDTH, BUFFER_HEIGHT) - 1, fetch_pos));
            float4 val = tex2Dfetch(SamplerLinearData, fetch_pos);
            int idx = GS_IDX(lx, ly);
            gs_Log2Luma[idx] = val.r;
            gs_ChromaA[idx]  = val.g;
            gs_ChromaB[idx]  = val.b;
            gs_LumaLin[idx]  = val.a;
        }
    }
    barrier();

    // -------------------------------------------------------------
    // PHASE 2: SETUP & EARLY OUTS
    // -------------------------------------------------------------
    if (global_pos.x >= BUFFER_WIDTH || global_pos.y >= BUFFER_HEIGHT) return;

    float4 src = tex2Dfetch(SamplerBackBuffer, global_pos);

#if !defined(__RESHADE_PERFORMANCE_MODE__) || !__RESHADE_PERFORMANCE_MODE__
    if (fStrength <= 0.0 && iDebugMode == 0)
#else
    if (fStrength <= 0.0)
#endif
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

    int2 local_center = int2(tid.xy) + int2(LDS_HALO, LDS_HALO);
    int center_idx = GS_IDX(local_center.x, local_center.y);

    float log2_center = gs_Log2Luma[center_idx];
    float luma_lin    = gs_LumaLin[center_idx];
    float whitePt     = GetResolvedWhitePoint();
    float inv_white   = 1.0 / max(whitePt, BCE_FLT_MIN);

    float3 color_lin  = DecodeToLinear(src.rgb);
    bool is_invalid = any(IsNan3(color_lin)) || any(IsInf3(color_lin));
    color_lin = is_invalid ? 0.0.xxx : color_lin;

#if !defined(__RESHADE_PERFORMANCE_MODE__) || !__RESHADE_PERFORMANCE_MODE__
    if (iDebugMode == 0 && luma_lin <= BCE_FLT_MIN)
#else
    if (luma_lin <= BCE_FLT_MIN)
#endif
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

#if !defined(__RESHADE_PERFORMANCE_MODE__) || !__RESHADE_PERFORMANCE_MODE__
    // -------------------------------------------------------------
    // PHASE 2.5: NON-NEIGHBORHOOD DEBUG VIEWS (Early Outs)
    // -------------------------------------------------------------
    if (iDebugMode == 7)
    {
        float3 dbg = (luma_lin <= BCE_FLT_MIN) ? float3(1, 0, 1) : float3(0, 0, 0);
        WriteDebugOut(global_pos, dbg, src.a);
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
        float norm = luma_lin / max(whitePt, BCE_FLT_MIN);
        float stops = log2(max(abs(norm), BCE_FLT_MIN));
        float t = saturate((stops + 6.0) / 8.0);
        float3 dbg = (norm < 0.0) ? float3(0, t, 0) : float3(t, 0, 0);
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
#endif

    // -------------------------------------------------------------
    // PHASE 3: EDGE DETECTION (100% via LDS)
    // -------------------------------------------------------------
    int base_radius = iRadius;
    float sigma_s = fSigmaSpatial;
    int radius = base_radius;

    if (bAdaptiveRadius && base_radius > 2)
    {
        float edge = GetEdgeStrengthShared(local_center, iEdgeDetectionMethod);

        if (bChromaAwareBilateral && fChromaEdgeStrength > 0.0)
        {
            float chromaEdge = ChromaEdgeShared(local_center, inv_white);
            edge = lerp(edge, max(edge, chromaEdge), fChromaEdgeStrength);
        }

        float scale = TrueSmoothstep(0.0, 1.0, edge * (fGradientSensitivity * 0.01));
        float factor = lerp(1.0, lerp(1.0, 0.15, scale), fAdaptiveRadiusStrength);
        int sigma_max = (int)(sigma_s * 3.0 + 0.5);
        radius = clamp(min((int)(base_radius * factor + 0.5), sigma_max), 1, base_radius);
    }

#if !defined(__RESHADE_PERFORMANCE_MODE__) || !__RESHADE_PERFORMANCE_MODE__
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
        float c = ChromaEdgeShared(local_center, inv_white);
        WriteDebugOut(global_pos, float3(c, c, c) * 5.0, src.a);
        return;
    }
#endif

    // -------------------------------------------------------------
    // PHASE 4: THE BILATERAL LOOP (HYBRID: LDS + VRAM FALLBACK)
    // -------------------------------------------------------------
    float inv_2_sigma_s_sq = 0.5 / (sigma_s * sigma_s);
    float inv_2_sigma_r_sq = 0.5 / (fSigmaRange * fSigmaRange);
    float inv_2_sigma_c_sq = 0.5 / (fSigmaChroma * fSigmaChroma);

    int cutoff_int  = (int)TrueSqrt(NEG_LN_SPATIAL_CUTOFF / inv_2_sigma_s_sq);
    int safe_radius = min(cutoff_int + 1, radius);
    int max_r       = min(safe_radius, MAX_LOOP_RADIUS);
    float r_limit_sq = float(max_r * max_r);

    float2 center_chroma = float2(gs_ChromaA[center_idx], gs_ChromaB[center_idx]);
    float center_chroma_reliability = 0.0;

    if (bChromaAwareBilateral)
    {
        center_chroma_reliability = GetChromaReliability(luma_lin, inv_white);
    }

    float2 stats_log = 0.0;
    float2 stats_sq  = 0.0;
    float2 stats_w   = 0.0;
    float min_log = log2_center;
    float max_log = log2_center;

    int r_lds = min(max_r, LDS_RADIUS);

    // Phase 4A: LDS-only inner core
    [loop]
    for (int y = -r_lds; y <= r_lds; ++y)
    {
        float y_f = float(y);
        float spatial_y = y_f * y_f * inv_2_sigma_s_sq;

        int x_limit_circ = (int)TrueSqrt(max(0.0, r_limit_sq - y_f * y_f));
        int x_start = max(-x_limit_circ, -r_lds);
        int x_end   = min( x_limit_circ,  r_lds);

        [loop]
        for (int x = x_start; x <= x_end; ++x)
        {
            int2 local_xy = ClampGlobalToTile(global_pos + int2(x, y), base_pos);
            int local_idx = GS_IDX(local_xy.x, local_xy.y);
            float4 n_data = float4(
                gs_Log2Luma[local_idx],
                gs_ChromaA[local_idx],
                gs_ChromaB[local_idx],
                gs_LumaLin[local_idx]
            );
            BCE_ACCUMULATE(n_data, x);
        }
    }

    // Phase 4B: VRAM fallback for outer ring (radius > halo)
    [branch]
    if (max_r > LDS_RADIUS)
    {
        // 1. Top Outer Ring
        [loop]
        for (int y = -max_r; y <= -LDS_RADIUS - 1; ++y)
        {
            float y_f = float(y);
            float spatial_y = y_f * y_f * inv_2_sigma_s_sq;
            int x_limit_circ = (int)TrueSqrt(max(0.0, r_limit_sq - y_f * y_f));
            int x_start = -x_limit_circ;
            int x_end   =  x_limit_circ;

            [loop]
            for (int x = x_start; x <= x_end; ++x)
            {
                int2 fetch_pos = clamp(global_pos + int2(x, y), int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
                float4 n_data = tex2Dfetch(SamplerLinearData, fetch_pos);
                BCE_ACCUMULATE(n_data, x);
            }
        }

        // 2. Middle Ring (Left and Right wings)
        [loop]
        for (int y = -LDS_RADIUS; y <= LDS_RADIUS; ++y)
        {
            float y_f = float(y);
            float spatial_y = y_f * y_f * inv_2_sigma_s_sq;
            int x_limit_circ = (int)TrueSqrt(max(0.0, r_limit_sq - y_f * y_f));

            // Left Wing
            int left_end = min(x_limit_circ, -LDS_RADIUS - 1);
            [loop]
            for (int x = -x_limit_circ; x <= left_end; ++x)
            {
                int2 fetch_pos = clamp(global_pos + int2(x, y), int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
                float4 n_data = tex2Dfetch(SamplerLinearData, fetch_pos);
                BCE_ACCUMULATE(n_data, x);
            }

            // Right Wing
            int right_start = LDS_RADIUS + 1;
            [loop]
            for (int x = right_start; x <= x_limit_circ; ++x)
            {
                int2 fetch_pos = clamp(global_pos + int2(x, y), int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
                float4 n_data = tex2Dfetch(SamplerLinearData, fetch_pos);
                BCE_ACCUMULATE(n_data, x);
            }
        }

        // 3. Bottom Outer Ring
        [loop]
        for (int y = LDS_RADIUS + 1; y <= max_r; ++y)
        {
            float y_f = float(y);
            float spatial_y = y_f * y_f * inv_2_sigma_s_sq;
            int x_limit_circ = (int)TrueSqrt(max(0.0, r_limit_sq - y_f * y_f));
            int x_start = -x_limit_circ;
            int x_end   =  x_limit_circ;

            [loop]
            for (int x = x_start; x <= x_end; ++x)
            {
                int2 fetch_pos = clamp(global_pos + int2(x, y), int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
                float4 n_data = tex2Dfetch(SamplerLinearData, fetch_pos);
                BCE_ACCUMULATE(n_data, x);
            }
        }
    }

    // -------------------------------------------------------------
    // PHASE 5: FINAL EVALUATION & WRITE
    // -------------------------------------------------------------
    float total_w = stats_w.x + stats_w.y;

    if (total_w < BCE_FLT_MIN)
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

    float total_log = stats_log.x + stats_log.y;
    float total_sq  = stats_sq.x  + stats_sq.y;
    float blurred   = total_log / total_w;
    float diff      = log2_center - blurred;

    float strength = fStrength;

    [branch]
    if (bAdaptiveStrength)
    {
        strength = CalculateAdaptiveStrength(total_log, total_sq, total_w, min_log, max_log, log2_center, fStrength, iAdaptiveMode);
    }

    float norm_luma = luma_lin / whitePt;
    float minCompNorm = GetMinComponent(color_lin) / whitePt;

    strength *= GetZoneProtection(norm_luma, minCompNorm, fShadowProtection, fMidtoneProtection, fHighlightProtection, fNegativeProtection);

    if (abs(strength) < BCE_FLT_MIN)
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

#if !defined(__RESHADE_PERFORMANCE_MODE__) || !__RESHADE_PERFORMANCE_MODE__
    if (iDebugMode == 1)
    {
        float3 dbg = saturate(log2(total_w + 1.0) * 0.1).xxx;
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 2)
    {
        float mean_diff = blurred - log2_center;
        float v = max(0.0, (total_sq / total_w) - mean_diff * mean_diff);
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
        float mean_diff = (total_log / total_w) - log2_center;
        float v = max(0.0, (total_sq / total_w) - mean_diff * mean_diff);
        float r = max_log - min_log;
        float e = log2(1.0 + v) * (1.0 + r * 0.1);
        WriteDebugOut(global_pos, float3(e * 0.25, e * 0.125, 0.0), src.a);
        return;
    }
#endif

    // Enforced bit-exact neutrality: sub-half-ulp deltas bypass transcoding
    if (abs(diff) * abs(strength) < BCE_NEUTRAL_LOG2_EPS)
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

    // Bujack non-Riemannian perceptual saturation applied to the extracted detail band:
    // Models diminishing returns on large contrast steps (specular glints, sharp silhouettes)
    // while delivering full linear boost to subtle micro-textures (ln(1 + x) ~= x for small x).
    float compressed_diff = diff;
    [branch]
    if (bNonRiemannianPerception)
    {
        float g = clamp(fDiminishingReturnsExponent, 0.05, 2.0);
        compressed_diff = sign(diff) * log(1.0 + PowNonNegPreserveZero(abs(diff), g)) / g;
    }

    // Direct stop-domain delta scaling: cancels baseline luma quantization drift (FP16 safe),
    // guarantees exact 1.0 multiplier when delta is 0, and saves an exp2 and a division.
    float ratio = clamp(exp2(strength * compressed_diff), RATIO_MIN, RATIO_MAX);
    float3 final_color = color_lin * ratio;

    if (any(IsNan3(final_color)) || any(IsInf3(final_color)))
    {
        final_color = color_lin;
    }

    int activeSpace = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;

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

void PS_OutputToScreen(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 fragColor : SV_Target)
{
    fragColor = tex2Dfetch(SamplerBilateralOut, int2(vpos.xy));
}

technique BilateralContrast_Reference <
    ui_label = "Bilateral Contrast v8.7.9 (Non-Riemannian Perception)";
    ui_tooltip = "Manual Stop-Domain Spatial Tuning - Non-Riemannian Perceptual Space\n\n"
                 "Verified against ReShade 6.8.0 (August 2026)\n\n"
                 "V8.7.9 Changes:\n"
                 "- Explicit sign(x) * (x * x) / 3.0 in HLG EOTF\n"
                 "- Symmetric sign-preserving color transforms across sRGB, PQ, and HLG\n"
                 "- Modern UI annotations (ui_category_toggle) per REFERENCE.md\n"
                 "- Zero-cost __RESHADE_PERFORMANCE_MODE__ debug stripping\n"
                 "- Retained Gaussian bilateral range kernel (no Cauchy halos)\n"
                 "- Bujack perceptual saturation on detail band with NaN protection\n"
                 "- Direct stop-domain ratio scaling exp2(strength * compressed_diff)\n"
                 "- Corrected LoG normalization constant to 0.00390625\n\n"
                 "Requires: DirectX 11+, OpenGL 4.3+, or Vulkan";
>
{
    pass PreCompute
    {
        VertexShader      = PostProcessVS;
        PixelShader       = PS_PrePass;
        RenderTarget      = TexLinearData;
        VertexCount       = 3;
        PrimitiveTopology = TRIANGLELIST;
        GenerateMipMaps   = false;
    }

    pass BilateralCompute
    {
        ComputeShader     = CS_BilateralContrast<16, 16, 1>;
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