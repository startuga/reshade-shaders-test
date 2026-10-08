/**
 * Bilateral Contrast Enhancement - COMPUTE LDS EDITION
 *
 * Design Philosophy: PRECISION AND QUALITY OVER PERFORMANCE
 * - Exact IEC / SMPTE / ITU constants (sRGB, ST 2084 PQ, BT.2100 HLG, BT.2124 ICtCp)
 * - Bit-exact passthrough: every early-out path, and every pixel whose correction is
 *   below the neutrality threshold, is written back untouched
 * - float32 pre-pass and output storage by default (PREPASS_USE_RGBA32F = 1)
 * - Pre-computed spatial kernels (groupshared LUT), no approximated math
 * - Stop-domain (log2) processing, sign-preserving EOTF/OETF pipelines (sRGB, PQ, HLG)
 * - ICtCp (Rec. ITU-R BT.2100 / BT.2124) chromaticity as an additional edge-stopping term
 * - Diminishing-returns saturation per detail band, inspired by Bujack et al., PNAS 2022
 * - Multi-scale detail decomposition: three bands (Micro / Medium / Macro) built from
 *   edge-aware bilateral base layers at increasing spatial scales
 * - Compute shader with groupshared LDS tile cache
 *
 * Terminology note:
 *   The decomposition is a difference of bilateral-weighted means (a telescoping,
 *   multi-scale base/detail split). It is NOT the Local Laplacian Filter of
 *   Paris et al. (2011), which remaps every pyramid coefficient around each local
 *   intensity level. The bilateral approach can show halos / gradient reversal at
 *   large radii and high strengths; keep the Medium / Macro strengths moderate.
 *
 * Precision note:
 *   exp / log / pow / sqrt are evaluated at the driver's precision (HLSL and Vulkan
 *   both permit a few ULP of error). "Exact" applies to constants and to the
 *   passthrough paths, not to the transcendental functions.
 *
 * Micro band (since 9.2.0):
 *   The finest band is an RCAS-class robust sharpener (same limiter as AMD FidelityFX FSR1
 *   RCAS, FsrRcasF) applied to PERCEPTUAL (sRGB-curve) luminance instead of a bilateral
 *   residual in stops, with luma-only ratio scaling and quantization coring:
 *     P      = sRGB-curve(luminance / white)                    (perceptual luma, unclamped)
 *     lobe   = clamp(no-clip limits from the 4-neighbour min/max, 0.1875)
 *              * (1 - Noise Suppression * |P - mean4| / local range)
 *     gain   = Sharpness * 4 lobe / (1 - 4 lobe)               (Sharpness 1.0 = RCAS maximum)
 *     detail = coring(P - mean4, tau),  tau = Coring x one source code step
 *     delta  = log2( EOTF(P + gain * detail) / EOTF(P) )        (stops, applied as an RGB ratio)
 *   Since 9.3.0 the limiter has two more terms:
 *     - Edge Halo Control: lobe <= 0.25 * (ring min / ring max)^(1 + 1.4 * halo). halo = 0 is the
 *       perceptual-domain RCAS limit; halo = 1 approximates RCAS run on LINEAR luminance (as in
 *       Lilium's HDR RCAS, luminance mode), which limits sharpening at strong edges much harder.
 *     - HDR ceiling: in HDR the no-clip ceiling is max(paper white, 1.25 x local max) in the
 *       perceptual domain, so isolated bright pixels are not pushed far above paper white.
 *   Why: the 9.1 micro band used the 0.28-stop range kernel at 1-2 px. That kernel excludes
 *   real mid-tone texture (gain 1.03x at +-16 codes) while including noise and 1-code
 *   banding steps (2.5x). Measured on synthetic 8-bit images (indicative, not game captures):
 *                              noise gain   texture gain (+-6 / +-16)   8-bit ramp banding
 *     9.1 defaults               2.59x          2.04x / 1.03x            3-code jumps
 *     RCAS max sharpness         2.20x          2.15x / 2.14x            none
 *     9.2 defaults               1.87x          2.18x / 2.14x            none
 *   Edge overshoot and isolated-sparkle response equal RCAS (same limiter). At colored edges
 *   RCAS shifts hue by ~1 deg and raises chroma by up to 29% (blue|yellow); the luma-only
 *   ratio keeps hue and chroma (<= 0.23 deg / 0.7%, i.e. 8-bit rounding).
 *   Medium and Macro bands are unchanged from 9.1 (they measured 1.09x noise, no banding).
 *
 * Comparison with Lilium's HDR RCAS (lilium__rcas_hdr.fx, luminance mode, noise removal on).
 *   Synthetic 8-bit content; SDR = 8-bit sRGB output, HDR = content at 203-nit paper white,
 *   10-bit PQ output. Gains are in source-code units. Indicative, not game captures.
 *                               noise    texture +-6 / +-16   edge 40->200      sparkle    HDR dark-ramp
 *                               gain                         under / over      200 on 60  reversals
 *     Lilium amount 0.5 (dflt)  1.65x    1.60x / 1.41x         3 / 0 codes      228        36
 *     Lilium amount 1.0         2.19x    2.17x / 1.72x         4 / 1 codes      245        40
 *     9.2 (Micro 1.0)           1.80x    2.14x / 2.13x         8 / 8 codes      223 (HDR 284)  8
 *     9.3 dflt (0.8, halo 0.5)  1.65x    1.87x / 1.64x         2 / 2 codes      218         6
 *     9.3 (1.3, halo 1.0)       2.02x    2.25x / 1.64x         1 / 1 codes      223         9
 *   Strong texture (+-40 codes) gets about the same or slightly less than Lilium (1.18-1.23x vs
 *   1.24-1.31x). Lilium's luminance mode is luma-only too, so neither fringes at colored edges
 *   (the colored-edge advantage stated for 9.2 applies against AMD's per-channel RCAS only).
 *
 * Version: 9.3.0
 * Changes since 9.2.0:
 * - New: Edge Halo Control (default 0.5) - limits micro sharpening at strong edges (see above).
 * - Fixed: in HDR the micro sharpener had no ceiling and pushed isolated bright pixels above
 *   paper white (sparkle 200 on 60 -> ~284 code-equivalent). Now limited as described above.
 * - Changed: Micro Sharpness default 1.0 -> 0.8 (same noise gain as Lilium's default).
 *
 * Changes in 9.2.0 (since 9.1.0):
 * - Changed: Micro band replaced by the RCAS-class robust sharpener described above.
 *   New controls: Micro Sharpness (0-2, 1.0 = RCAS maximum), Micro Noise Suppression
 *   (RCAS denoise, 0.5 = RCAS default), Quantization Coring and Source Content Precision.
 *   The old fStrengthMicro uniform is removed, so saved 9.1 Micro values are not reused
 *   with the new meaning.
 * - Changed: Adaptive Strength now modulates only the Medium / Macro bands; the micro
 *   sharpener has its own contrast adaptivity (RCAS limiter).
 * - Changed: Debug "Micro Band" shows the signed micro correction.
 * - Everything else (Medium / Macro bands, legacy mode, protection zones, color pipelines,
 *   LDS layout) is identical to 9.1.0.
 *
 * Changes in 9.1.0 (since 9.0.1):
 * - Changed: PREPASS_USE_RGBA32F now defaults to 1. Half-float log2-luma has only
 *   ~0.002-0.004 stop spacing near mid-gray, which is visible at high Micro strength
 *   on 10-bit / HDR outputs.
 * - Changed: LDS row stride 33 -> 48. With a 16x16 group mapped row-major, a wave32 spans
 *   two 16-wide rows; a stride of 16 (mod 32) puts the second row on the other 16 banks
 *   (stride 33 only shifted it by one bank). LDS use is 24 KB + 1.5 KB LUT.
 *   BCE_COMPAT_VULKAN_MIN_LDS = 1 still gives stride 32 / exactly 16 KB, with no LUT.
 * - Changed: Spatial weights come from a groupshared LUT indexed by integer r^2, and the
 *   range / chroma exponential is evaluated once per tap and shared by all three bands
 *   (3 exp() per tap -> 1 exp() per tap in the LDS core).
 * - Fixed: Diminishing-returns exponent is clamped to [1, 2]. For exponents below 1 the
 *   curve has infinite slope at zero and amplified noise by an unbounded factor. Slope
 *   at zero is now exactly 1 (g = 1) or 0 (g > 1). Added a Saturation Knee (stops).
 *   Saved presets with an exponent below 1.0 load as 1.0.
 * - Fixed: ln(1 + x) uses a cancellation-free evaluation (BCE_Log1p).
 * - Fixed: Loop bounds use integer sqrt (BCE_ISqrt), so a sqrt() that is 1 ULP low can no
 *   longer drop a boundary tap.
 * - Fixed: Micro / Medium radii are clamped to the macro footprint (max_r).
 * - Fixed: Tooltips / labels no longer overstate the Macro scale or compare against RCAS.
 * - Removed: unused EDGE_LUMA_FLOOR. Moved the BUFFER_WIDTH/HEIGHT guard ahead of its use.
 *
 * Requires: DirectX 11+, OpenGL 4.3+, or Vulkan
 * Targets:  ReShade 4.8.0+ (compute shader support)
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

#if !defined(BUFFER_WIDTH) || !defined(BUFFER_HEIGHT)
    #error "Bilateral Contrast: Missing BUFFER_WIDTH/HEIGHT. ReShade.fxh injection failed."
#endif

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

#ifndef BUFFER_COLOR_BIT_DEPTH
    #define BUFFER_COLOR_BIT_DEPTH 8
#endif

// Set to 1: float32 intermediate and output storage (Mastering Standard - no precision loss)
// Set to 0: float16 (50% less VRAM, ~0.002-0.004 stop quantization of log2-luma near mid-gray)
// Note: Visible in ReShade UI Preprocessor Definitions dialogue (>= 8 characters).
#ifndef PREPASS_USE_RGBA32F
    #define PREPASS_USE_RGBA32F 1
#endif

#if PREPASS_USE_RGBA32F
    #define PREPASS_FORMAT RGBA32F
#else
    #define PREPASS_FORMAT RGBA16F
#endif

// Set to 1 on devices that only provide the Vulkan minimum of 16384 bytes of
// groupshared memory. Uses an unpadded stride of 32 (4 x 1024 x 4 B = 16384 B exactly)
// and disables the spatial-weight LUT (the exponentials are evaluated per tap instead).
#ifndef BCE_COMPAT_VULKAN_MIN_LDS
    #define BCE_COMPAT_VULKAN_MIN_LDS 0
#endif

#if BCE_COMPAT_VULKAN_MIN_LDS
    #define LDS_STRIDE          32
    #define BCE_USE_SPATIAL_LUT 0
#else
    // 16x16 group, row-major lane mapping: lane = ty*16 + tx. A stride that is 16 mod 32
    // places the second row of a wave32 on the complementary 16 banks (conflict-free).
    // 4 arrays x 32 x 48 x 4 B = 24576 B (+ 1548 B LUT) - within the 32 KB D3D11 limit.
    #define LDS_STRIDE          48
    #define BCE_USE_SPATIAL_LUT 1
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

// Squared-radius LUT range of the LDS core: |x|,|y| <= LDS_RADIUS  ->  r^2 in [0, 2*R^2]
static const int BCE_LUT_SIZE              = 2 * LDS_RADIUS * LDS_RADIUS + 1;

static const float RATIO_MIN               = 0.0001;
static const float RATIO_MAX               = 10000.0;

// Chroma reliability fade-in, expressed in NORMALIZED luma (fraction of active white point).
static const float BCE_CHROMA_REL_START    = 4.8828125e-4;      // 2^-11 of active white
static const float BCE_CHROMA_REL_FULL     = 1.953125e-3;       // 2^-9  of active white
static const float BCE_INV_CHROMA_REL_SPAN = 2048.0 / 3.0;      // 1 / (FULL - START), exact in binary
static const float LOG2_EDGE_LUMA_FLOOR    = -13.2877123795;

// Neutral passthrough: |delta log2| below this cannot perturb an 8-bit output
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

// pooled = true re-uses texture memory across effects
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

// ==============================================================================
// 3. UI Parameters (Multi-Scale Detail Pyramid)
// ==============================================================================

uniform bool bEnableMultiScale <
    ui_label = "Enable Multi-Scale Engine";
    ui_tooltip = "Splits stop-domain contrast into three bands built from edge-aware bilateral\n"
                 "base layers: Micro-Texture, Mid-Clarity, and Macro-Depth.";
    ui_category = "Multi-Scale Detail Pyramid";
    ui_category_toggle = true;
> = true;

uniform float fMicroSharpness <
    ui_type = "slider";
    ui_label = "Micro Sharpness (RCAS-class)";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_tooltip = "Robust 1-2 px sharpening of perceptual luminance (skin pores, fabric weave, hair, foliage).\n"
                 "1.0 = the gain of AMD RCAS at maximum sharpness; 0.8 has about the same noise gain\n"
                 "as Lilium's HDR RCAS at its default amount. Values above 1.0 sharpen more;\n"
                 "the no-clip and halo limiters still apply, so strong edges are not over-driven.\n"
                 "Luma only: hue and saturation are preserved at colored edges.";
    ui_category = "Multi-Scale Detail Pyramid";
> = 0.8;

uniform float fMicroHaloControl <
    ui_type = "slider";
    ui_label = "Micro Edge Halo Control";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Limits micro sharpening at strong edges (bright / dark rims, ringing).\n"
                 "0.0 = AMD RCAS limit (perceptual domain).\n"
                 "1.0 = about the limit of RCAS on linear luminance (Lilium's HDR RCAS):\n"
                 "      the smallest halos, but strong texture is sharpened less.";
    ui_category = "Multi-Scale Detail Pyramid";
> = 0.5;

uniform float fMicroNoiseSuppression <
    ui_type = "slider";
    ui_label = "Micro Noise Suppression";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Reduces micro sharpening on isolated pixels that stand out from all neighbours\n"
                 "(noise, dither, sparkle). 0.5 = RCAS denoise. 1.0 = strongest.";
    ui_category = "Multi-Scale Detail Pyramid";
> = 0.5;

uniform float fMicroCoring <
    ui_type = "slider";
    ui_label = "Quantization Coring (Source Steps)";
    ui_min = 0.0; ui_max = 4.0; ui_step = 0.01;
    ui_tooltip = "Micro detail smaller than this many source code steps is not sharpened, so banding\n"
                 "steps and +-1 code noise / dither are not amplified. 0 = off.";
    ui_category = "Multi-Scale Detail Pyramid";
> = 1.0;

uniform int iSourcePrecision <
    ui_type = "combo";
    ui_label = "Source Content Precision";
    ui_items = "Off (no coring)\0""8-bit (typical game, default)\0""6-bit (BC1/DXT1 green, RGB565)\0""5-bit (BC1/DXT1 red/blue)\0""10-bit\0";
    ui_tooltip = "Precision the GAME produced its image with (independent of the swapchain format).\n"
                 "Old games are 8-bit or less, also when shown on HDR / FP16 swapchains.\n"
                 "Use Off or 10-bit for native HDR content.";
    ui_category = "Multi-Scale Detail Pyramid";
> = 1;

uniform float fStrengthMedium <
    ui_type = "slider";
    ui_label = "Mid-Frequency Clarity";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
    ui_tooltip = "Enhances structural edges, bevels, depth contours, and object boundaries.\n"
                 "Band radius follows the filter radius (up to 8 px).";
    ui_category = "Multi-Scale Detail Pyramid";
> = 0.5;

uniform float fStrengthMacro <
    ui_type = "slider";
    ui_label = "Macro Depth (broad scale)";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
    ui_tooltip = "Enhances the broad band between the Medium base layer and the full-radius base layer:\n"
                 "local illumination gradients, shape curvature, 3D depth pop.\n"
                 "Its scale is set by Filter Base Radius and Spatial Sigma; raise both for broader depth.\n"
                 "High values at large radii can produce halos around strong edges.";
    ui_category = "Multi-Scale Detail Pyramid";
> = 0.25;

uniform float fStrengthMaster <
    ui_type = "slider";
    ui_label = "Master Scale Multiplier";
    ui_min = 0.0; ui_max = 3.0; ui_step = 0.01;
    ui_tooltip = "Global intensity multiplier scaling all three frequency bands simultaneously.";
    ui_category = "Multi-Scale Detail Pyramid";
> = 1.00;

uniform float fStrength <
    ui_type = "slider";
    ui_label = "Legacy Single-Scale Strength";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.001;
    ui_tooltip = "Only active when 'Enable Multi-Scale Engine' is disabled.";
    ui_category = "Legacy Single-Scale Settings";
> = 3.20;

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
    ui_units = "nits";
    ui_tooltip = "Only applies when using HDR/scRGB Color Space overrides.\nConforms to ITU-R BT.2408 reference paper white.";
    ui_category = "Protection Zones";
> = 203.0;

uniform float fNegativeProtection <
    ui_type = "slider";
    ui_label = "Negative Value Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_tooltip = "Protects out-of-gamut negative RGB values created or preserved by ratio scaling.";
    ui_category = "Protection Zones";
> = 0;

uniform bool bAdaptiveStrength <
    ui_label = "Enable Adaptive Strength";
    ui_category = "Adaptive Processing";
    ui_category_toggle = true;
> = true;

uniform int iAdaptiveMode <
    ui_type = "combo";
    ui_label = "Adaptive Mode";
    ui_tooltip = "Modulates the Medium / Macro bands (and the legacy mode) by local variance / range.\n"
                 "The micro sharpener has its own contrast adaptivity and is not affected.";
    ui_items = "Dynamic Range\0""Variance\0""Hybrid\0""Range-Variance Hybrid\0";
    ui_category = "Adaptive Processing";
> = 3;

uniform float fAdaptiveAmount <
    ui_type = "slider";
    ui_label = "Adaptive Amount";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Adaptive Processing";
> = 0.30;

uniform float fAdaptiveCurve <
    ui_type = "slider";
    ui_label = "Adaptive Curve";
    ui_min = 0.1; ui_max = 4.0; ui_step = 0.01;
    ui_category = "Adaptive Processing";
> = 1.15;

uniform int iRadius <
    ui_type = "slider";
    ui_label = "Filter Base Radius";
    ui_min = 1; ui_max = 32;
    ui_units = "px";
    ui_tooltip = "Governs the outer spatial extent (Macro scale). Micro and Medium radii scale proportionally.\n"
                 "Radii above 8 px additionally read the outer ring from VRAM.";
    ui_category = "Filter Parameters";
> = 8;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_min = 0.1; ui_max = 32.0; ui_step = 0.01;
    ui_units = "px";
    ui_category = "Filter Parameters";
> = 3;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Range Sigma (Stops)";
    ui_min = 0.01; ui_max = 4.0; ui_step = 0.001;
    ui_units = "stops";
    ui_tooltip = "Edge-stopping sensitivity. Smaller values strictly lock the filter to boundaries.";
    ui_category = "Filter Parameters";
> = 0.28;

uniform float fSigmaChroma <
    ui_type = "slider";
    ui_label = "Chroma Sigma";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.001;
    ui_tooltip = "Controls filter sensitivity to ICtCp chromaticity differences.";
    ui_category = "Filter Parameters";
> = 0.20;

uniform bool bChromaAwareBilateral <
    ui_label = "Chroma-Aware Filtering";
    ui_category = "Filter Parameters";
> = true;

uniform bool bNonRiemannianPerception <
    ui_label = "Enable Non-Riemannian Metric";
    ui_tooltip = "Applies a Bujack-inspired diminishing-returns saturation to each detail band:\n"
                 "    f(d) = (k / g) * ln(1 + (|d| / k)^g)\n"
                 "with k = Saturation Knee and g = Perceptual Saturation Exponent.\n"
                 "Limits specular blow-out and halo amplification from large detail magnitudes.";
    ui_category = "Non-Riemannian Perception";
    ui_category_toggle = true;
> = true;

uniform float fDiminishingReturnsExponent <
    ui_type = "slider";
    ui_label = "Perceptual Saturation Exponent";
    ui_min = 1.00; ui_max = 2.00; ui_step = 0.01;
    ui_tooltip = "Exponent g of the saturation curve (clamped to 1..2).\n"
                 "1.0: logarithmic compression, slope exactly 1 at zero (small details pass unchanged).\n"
                 "> 1.0: additionally suppresses very small details (soft dead-zone) and tames extremes.\n"
                 "Values below 1.0 are not allowed: the curve slope diverges at zero and amplifies noise.";
    ui_category = "Non-Riemannian Perception";
> = 1.00;

uniform float fSaturationKnee <
    ui_type = "slider";
    ui_label = "Saturation Knee (Stops)";
    ui_min = 0.10; ui_max = 8.00; ui_step = 0.01;
    ui_units = "stops";
    ui_tooltip = "Detail magnitude at which compression becomes significant.\n"
                 "Smaller: compress earlier (gentler on strong edges). Larger: more linear.\n"
                 "1.0 reproduces the original ln(1 + |d|) behaviour at exponent 1.0.";
    ui_category = "Non-Riemannian Perception";
> = 1.00;

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
> = 0.65;

uniform float fChromaEdgeStrength <
    ui_type = "slider";
    ui_label = "Chroma Edge Influence";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Controls how strongly chroma edges reduce the filter radius.\n0.0 = Luma only. 1.0 = Max(Luma, ICtCp Chroma).";
    ui_category = "Adaptive Radius";
> = 0.40;

uniform int iEdgeDetectionMethod <
    ui_type = "combo";
    ui_label = "Edge Detection Method";
    ui_items = "Sobel 3x3\0""Scharr 3x3\0""Prewitt 3x3\0""Sobel 5x5\0""Laplacian of Gaussian\0""Structure Tensor\0";
    ui_category = "Adaptive Radius";
> = 5;

uniform float fGradientSensitivity <
    ui_type = "slider";
    ui_label = "Gradient Sensitivity";
    ui_min = 10.0; ui_max = 500.0; ui_step = 1.0;
    ui_category = "Advanced Tuning";
    ui_category_closed = true;
> = 180.0;

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
    ui_items = "Auto (Default)\0""sRGB (SDR)\0""scRGB (HDR Linear)\0""HDR10 (PQ)\0""HLG (HDR)\0";
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
    ui_items = "Off\0"
                "Weights\0"
                "Variance\0"
                "Dynamic Range\0"
                "Enhancement Map\0"
                "Adaptive Radius\0"
                "Edge Detection\0"
                "Black Pixels\0"
                "Chroma Edges\0"
                "Entropy\0"
                "Zone Map\0"
                "Negative Values\0"
                "Signed Luminance\0"
                "Micro Sharpening (signed)\0"
                "Medium Band (Clarity)\0"
                "Macro Band (Depth)\0"
                "Multi-Scale Composite\0";
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

// Integer floor(sqrt(v)) for loop bounds. The +0.5 centres the truncation between
// consecutive perfect squares, so an sqrt() that is off by 1 ULP can never change the result.
int BCE_ISqrt(int v)
{
    return (int)sqrt(float(max(v, 0)) + 0.5);
}

// ln(1 + x) for x >= 0 without catastrophic cancellation near zero (Kahan's formulation).
float BCE_Log1p(float x)
{
    float u = 1.0 + x;
    return (u == 1.0) ? x : log(u) * (x / (u - 1.0));
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

// Odd-symmetric diminishing-returns curve applied to a signed detail value (in stops).
//   f(d) = sign(d) * (k / g) * ln(1 + (|d| / k)^g),   g in [1, 2],  k = Saturation Knee
// g = 1 : slope at zero is exactly 1 (small details pass unchanged).
// g > 1 : slope at zero is 0 (soft dead-zone); never amplifies noise.
float BCE_DetailSaturate(float d)
{
    float g = clamp(fDiminishingReturnsExponent, 1.0, 2.0);
    float k = max(fSaturationKnee, 1e-3);
    float a = abs(d) / k;
    return sign(d) * (k / g) * BCE_Log1p(PowNonNegPreserveZero(a, g));
}

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

// Diminishing-returns metric on a squared chroma distance. Exponent clamped to [1, 2]
// so the slope at zero stays finite (see BCE_DetailSaturate).
float ApplyNonRiemannianMetric(float dist_sq, float kappa)
{
    if (!bNonRiemannianPerception) return dist_sq;

    float gamma = clamp(fDiminishingReturnsExponent, 1.0, 2.0);
    float dp = PowNonNegPreserveZero(dist_sq * kappa, gamma);
    return BCE_Log1p(dp) / (gamma * kappa);
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

#if BCE_USE_SPATIAL_LUT
// exp(-r^2 / (2 sigma^2)) for integer r^2, one table per octave band
groupshared float gs_SpatialMacro[BCE_LUT_SIZE];
groupshared float gs_SpatialMed[BCE_LUT_SIZE];
groupshared float gs_SpatialMicro[BCE_LUT_SIZE];
#endif

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
    return response * response * 0.00390625;
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

// ------------------------------------------------------------------------------
// RCAS-class robust micro sharpener on perceptual luminance (see header).
// ------------------------------------------------------------------------------

// HDR ceiling headroom over the local maximum (perceptual units; 1.25 ~ +0.77 stops)
static const float BCE_MICRO_HDR_HEADROOM = 1.25;

// Perceptual luma: sRGB curve relative to the active white, unclamped above white (HDR)
float BCE_PerceptualLuma(float luma_nits, float inv_white)
{
    return sRGB_OETF(max(luma_nits, 0.0).xxx * inv_white).x;
}

int BCE_SourceBits()
{
    if (iSourcePrecision == 1) return 8;
    if (iSourcePrecision == 2) return 6;
    if (iSourcePrecision == 3) return 5;
    if (iSourcePrecision == 4) return 10;
    return 0;
}

// Returns the micro correction in stops (0 exactly when Sharpness is 0 or nothing changes).
float BCE_RobustMicroStops(int2 local_center, float inv_white, bool sdr_ceiling)
{
    if (fMicroSharpness <= 0.0) return 0.0;

    float e = BCE_PerceptualLuma(gs_LumaLin[GS_IDX(local_center.x,     local_center.y    )], inv_white);
    float b = BCE_PerceptualLuma(gs_LumaLin[GS_IDX(local_center.x,     local_center.y - 1)], inv_white);
    float d = BCE_PerceptualLuma(gs_LumaLin[GS_IDX(local_center.x - 1, local_center.y    )], inv_white);
    float f = BCE_PerceptualLuma(gs_LumaLin[GS_IDX(local_center.x + 1, local_center.y    )], inv_white);
    float h = BCE_PerceptualLuma(gs_LumaLin[GS_IDX(local_center.x,     local_center.y + 1)], inv_white);

    float mn4 = min(min(b, d), min(f, h));
    float mx4 = max(max(b, d), max(f, h));
    float mxa = max(mx4, e);

    // Output ceiling: SDR white, or in HDR max(paper white, 1.25 x local max) (perceptual domain)
    float ceil_p = sdr_ceiling ? 1.0 : max(1.0, BCE_MICRO_HDR_HEADROOM * mxa);

    // RCAS no-clip limits: no undershoot below 0, no overshoot above the ceiling.
    // Denominator kept strictly negative: a ring at the ceiling would otherwise give 0 / 0
    float hit_min  = min(mn4, e) / max(4.0 * mx4, BCE_FLT_MIN);
    float hit_max  = (ceil_p - mxa) / min(4.0 * mn4 - 4.0 * ceil_p, -1e-6);
    float lobe_hit = -min(max(-hit_min, hit_max), 0.0);     // >= 0, bound on |lobe|

    // Edge halo limit: ring contrast ratio raised to 1 (perceptual RCAS) .. 2.4 (~linear-light RCAS)
    float ring_ratio = saturate(mn4 / max(mx4, BCE_FLT_MIN));
    float lobe_halo  = 0.25 * PowNonNegPreserveZero(ring_ratio, 1.0 + 1.4 * saturate(fMicroHaloControl));

    float lobe_lim = min(lobe_hit, lobe_halo);
    float lobe     = min(lobe_lim, 0.1875);

    // RCAS denoise: isolated pixels (center far from the ring mean, relative to the local range)
    float detail = e - 0.25 * (b + d + f + h);
    float range  = max(mx4, e) - min(mn4, e);
    float nz     = saturate(abs(detail) / max(range, BCE_FLT_MIN));
    float dn     = 1.0 - saturate(fMicroNoiseSuppression) * nz;

    float l   = lobe * dn;
    float gain = fMicroSharpness * (4.0 * l) / (1.0 - 4.0 * l);

    // Above 1.0 the strength may not exceed the no-clip / halo bound
    float l_hit = lobe_lim * dn;
    if (l_hit < 0.25)
    {
        gain = min(gain, (4.0 * l_hit) / (1.0 - 4.0 * l_hit));
    }

    // Quantization coring: detail * detail^2 / (detail^2 + tau^2), tau = N source code steps
    int bits = BCE_SourceBits();
    if (bits > 0 && fMicroCoring > 0.0)
    {
        float tau = fMicroCoring / (exp2(float(bits)) - 1.0);
        float d2  = detail * detail;
        detail = detail * d2 / (d2 + tau * tau);
    }

    float dp = gain * detail;
    if (dp == 0.0) return 0.0;

    // Same EOTF for numerator and denominator: no round-trip error when dp is tiny
    float l_old = sRGB_EOTF(e.xxx).x;
    float l_new = sRGB_EOTF(max(e + dp, 0.0).xxx).x;
    return log2(max(l_new, BCE_FLT_MIN) / max(l_old, BCE_FLT_MIN));
}

// ==============================================================================
// 9. Multi-Scale Bilateral Processing (Compute Shader Hybrid LDS)
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

// Spatial weight factor for an integer squared radius.
// LUT mode: groupshared table filled once per group. Compat mode: evaluated per tap.
#if BCE_USE_SPATIAL_LUT
    #define BCE_SPATIAL_W_MACRO(r_sq_i) gs_SpatialMacro[(r_sq_i)]
    #define BCE_SPATIAL_W_MED(r_sq_i)   gs_SpatialMed[(r_sq_i)]
    #define BCE_SPATIAL_W_MICRO(r_sq_i) gs_SpatialMicro[(r_sq_i)]
#else
    #define BCE_SPATIAL_W_MACRO(r_sq_i) exp(-float(r_sq_i) * inv_2_sigma_s_sq)
    #define BCE_SPATIAL_W_MED(r_sq_i)   exp(-float(r_sq_i) * inv_2_sigma_s_med_sq)
    #define BCE_SPATIAL_W_MICRO(r_sq_i) exp(-float(r_sq_i) * inv_2_sigma_s_micro_sq)
#endif

// Micro & Medium & Macro LDS Accumulator Macro (Phase 4A)
// The range/chroma exponential is evaluated once and shared by all three bands:
//   w_band = exp(-dist_sq) * spatial_band[r^2]
#define BCE_ACCUMULATE_MULTISCALE(n_data, x_coord, y_coord)                                                                    \
{                                                                                                                              \
    int   _r_sq_i = (x_coord) * (x_coord) + (y_coord) * (y_coord);                                                             \
    float _n_log  = (n_data).r;                                                                                                \
    float _n_luma = (n_data).a;                                                                                                \
    float _d_luma = log2_center - _n_log;                                                                                      \
    float _dist_sq = _d_luma * _d_luma * inv_2_sigma_r_sq;                                                                     \
    [branch]                                                                                                                   \
    if (bChromaAwareBilateral)                                                                                                 \
    {                                                                                                                          \
        float _dcx = center_chroma.x - (n_data).g;                                                                             \
        float _dcy = center_chroma.y - (n_data).b;                                                                             \
        float _d_chroma_sq = _dcx * _dcx + _dcy * _dcy;                                                                        \
        float _chroma_reliability = center_chroma_reliability * GetChromaReliability(_n_luma, inv_white);                      \
        _dist_sq += _d_chroma_sq * (BCE_CHROMA_CONDITIONING_ACC * BCE_CHROMA_CONDITIONING_ACC)                                 \
                    * _chroma_reliability * inv_2_sigma_c_sq;                                                                  \
    }                                                                                                                          \
    float _w_range = exp(-_dist_sq);                                                                                           \
    /* Macro Band Accumulation (full radius) */                                                                                \
    float _w_macro = _w_range * BCE_SPATIAL_W_MACRO(_r_sq_i);                                                                  \
    if (_w_macro > BCE_FLT_MIN)                                                                                                \
    {                                                                                                                          \
        stats_log_macro += _n_log * _w_macro;                                                                                  \
        stats_w_macro   += _w_macro;                                                                                           \
        float _d_center = _n_log - log2_center;                                                                                \
        stats_sq_macro  += _d_center * _d_center * _w_macro;                                                                   \
        min_log = min(min_log, _n_log);                                                                                        \
        max_log = max(max_log, _n_log);                                                                                        \
    }                                                                                                                          \
    [branch]                                                                                                                   \
    if (bEnableMultiScale)                                                                                                     \
    {                                                                                                                          \
        /* Medium Band Accumulation (octave 2) */                                                                              \
        if (_r_sq_i <= r_med_sq_i)                                                                                             \
        {                                                                                                                      \
            float _w_med = _w_range * BCE_SPATIAL_W_MED(_r_sq_i);                                                              \
            if (_w_med > BCE_FLT_MIN)                                                                                          \
            {                                                                                                                  \
                stats_log_med += _n_log * _w_med;                                                                              \
                stats_w_med   += _w_med;                                                                                       \
            }                                                                                                                  \
        }                                                                                                                      \
        /* Micro Band Accumulation (octave 1: micro-texture) */                                                                \
        if (_r_sq_i <= r_micro_sq_i)                                                                                           \
        {                                                                                                                      \
            float _w_micro = _w_range * BCE_SPATIAL_W_MICRO(_r_sq_i);                                                          \
            if (_w_micro > BCE_FLT_MIN)                                                                                        \
            {                                                                                                                  \
                stats_log_micro += _n_log * _w_micro;                                                                          \
                stats_w_micro   += _w_micro;                                                                                   \
            }                                                                                                                  \
        }                                                                                                                      \
    }                                                                                                                          \
}

// Dedicated Macro Outer-Ring Accumulator Macro (Phase 4B: VRAM Fallback)
// Avoids dead micro/medium branches and hoists spatial_y out of the loop
#define BCE_ACCUMULATE_MACRO(n_data, x_coord, spatial_y)                                                                       \
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
        float _chroma_reliability = center_chroma_reliability * GetChromaReliability(_n_luma, inv_white);                      \
        _dist_sq += _d_chroma_sq * (BCE_CHROMA_CONDITIONING_ACC * BCE_CHROMA_CONDITIONING_ACC)                                 \
                    * _chroma_reliability * inv_2_sigma_c_sq;                                                                  \
    }                                                                                                                          \
    float _exp_macro = -(float((x_coord) * (x_coord)) * inv_2_sigma_s_sq + (spatial_y)) - _dist_sq;                            \
    if (_exp_macro > BCE_LN_FLT_MIN)                                                                                           \
    {                                                                                                                          \
        float _w_macro = exp(_exp_macro);                                                                                      \
        stats_log_macro += _n_log * _w_macro;                                                                                  \
        stats_w_macro   += _w_macro;                                                                                           \
        float _d_center = _n_log - log2_center;                                                                                \
        stats_sq_macro  += _d_center * _d_center * _w_macro;                                                                   \
        min_log = min(min_log, _n_log);                                                                                        \
        max_log = max(max_log, _n_log);                                                                                        \
    }                                                                                                                          \
}

void CS_BilateralContrast(uint3 id : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    int2 global_pos = int2(id.xy);

    // -------------------------------------------------------------
    // Dispatch-uniform spatial / range constants (needed before the barrier for the LUT)
    // -------------------------------------------------------------
    float sigma_s     = fSigmaSpatial;
    float sigma_med   = max(sigma_s * 0.55, 1.60);
    float sigma_micro = max(sigma_s * 0.22, 0.75);

    float inv_2_sigma_s_sq       = 0.5 / (sigma_s * sigma_s);
    float inv_2_sigma_s_med_sq   = 0.5 / (sigma_med * sigma_med);
    float inv_2_sigma_s_micro_sq = 0.5 / (sigma_micro * sigma_micro);

    float inv_2_sigma_r_sq = 0.5 / (fSigmaRange * fSigmaRange);
    float inv_2_sigma_c_sq = 0.5 / (fSigmaChroma * fSigmaChroma);

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

#if BCE_USE_SPATIAL_LUT
    {
        // 256 threads fill the 129-entry tables (3 exp() per entry, once per group)
        uint lut_i = tid.y * 16 + tid.x;
        if (lut_i < (uint)BCE_LUT_SIZE)
        {
            float lut_r_sq = float(lut_i);
            gs_SpatialMacro[lut_i] = exp(-lut_r_sq * inv_2_sigma_s_sq);
            gs_SpatialMed[lut_i]   = exp(-lut_r_sq * inv_2_sigma_s_med_sq);
            gs_SpatialMicro[lut_i] = exp(-lut_r_sq * inv_2_sigma_s_micro_sq);
        }
    }
#endif
    barrier();

    // -------------------------------------------------------------
    // PHASE 2: SETUP & EARLY OUTS
    // -------------------------------------------------------------
    if (global_pos.x >= BUFFER_WIDTH || global_pos.y >= BUFFER_HEIGHT) return;

    float4 src = tex2Dfetch(SamplerBackBuffer, global_pos);

    bool is_active = bEnableMultiScale ?
        (fStrengthMaster > 0.0 && (fMicroSharpness > 0.0 || fStrengthMedium > 0.0 || fStrengthMacro > 0.0)) :
        (fStrength > 0.0);

#if !defined(__RESHADE_PERFORMANCE_MODE__) || !__RESHADE_PERFORMANCE_MODE__
    if (!is_active && iDebugMode == 0)
#else
    if (!is_active)
#endif
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

    int2 local_center = int2(tid.xy) + int2(LDS_HALO, LDS_HALO);
    int center_idx = GS_IDX(local_center.x, local_center.y);

    float log2_center = gs_Log2Luma[center_idx];
    float luma_lin    = gs_LumaLin[center_idx];
    int   activeSpaceEarly = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
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
    // PHASE 4: MULTI-SCALE ACCUMULATION (HYBRID: LDS + VRAM)
    // -------------------------------------------------------------
    // Macro octave footprint (Gaussian truncated at exp(-9.21) = 1e-4)
    int cutoff_int   = (int)TrueSqrt(NEG_LN_SPATIAL_CUTOFF / inv_2_sigma_s_sq);
    int safe_radius  = min(cutoff_int + 1, radius);
    int max_r        = min(safe_radius, MAX_LOOP_RADIUS);
    int r_limit_sq_i = max_r * max_r;

    // Micro octave (1-3 px) and Medium octave (up to 8 px), clamped to the macro footprint
    int r_micro = min(clamp((int)round(float(radius) * 0.22), 1, 3), max_r);
    int r_med   = max(r_micro, min(clamp((int)round(float(radius) * 0.55), r_micro + 1, 8), max_r));

    int r_micro_sq_i = r_micro * r_micro;
    int r_med_sq_i   = r_med * r_med;

    float2 center_chroma = float2(gs_ChromaA[center_idx], gs_ChromaB[center_idx]);
    float center_chroma_reliability = bChromaAwareBilateral ? GetChromaReliability(luma_lin, inv_white) : 0.0;

    // Multi-scale accumulation state (scalar floats keep VGPR pressure low)
    float stats_log_macro = 0.0, stats_w_macro = 0.0, stats_sq_macro = 0.0;
    float stats_log_med   = 0.0, stats_w_med   = 0.0;
    float stats_log_micro = 0.0, stats_w_micro = 0.0;

    float min_log = log2_center;
    float max_log = log2_center;

    int r_lds = min(max_r, LDS_RADIUS);

    // Phase 4A: LDS-only inner core
    // All three bands execute here with zero VRAM cost
    [loop]
    for (int y = -r_lds; y <= r_lds; ++y)
    {
        int x_limit_circ = BCE_ISqrt(r_limit_sq_i - y * y);
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
            BCE_ACCUMULATE_MULTISCALE(n_data, x, y);
        }
    }

    // Phase 4B: VRAM fallback for Macro outer ring (radius > LDS_RADIUS)
    [branch]
    if (max_r > LDS_RADIUS)
    {
        // 1. Top Outer Ring
        [loop]
        for (int y = -max_r; y <= -LDS_RADIUS - 1; ++y)
        {
            float spatial_y = float(y * y) * inv_2_sigma_s_sq;
            int x_limit_circ = BCE_ISqrt(r_limit_sq_i - y * y);
            int x_start = -x_limit_circ;
            int x_end   =  x_limit_circ;

            [loop]
            for (int x = x_start; x <= x_end; ++x)
            {
                int2 fetch_pos = clamp(global_pos + int2(x, y), int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
                float4 n_data = tex2Dfetch(SamplerLinearData, fetch_pos);
                BCE_ACCUMULATE_MACRO(n_data, x, spatial_y);
            }
        }

        // 2. Middle Ring (Left and Right wings)
        [loop]
        for (int y = -LDS_RADIUS; y <= LDS_RADIUS; ++y)
        {
            float spatial_y = float(y * y) * inv_2_sigma_s_sq;
            int x_limit_circ = BCE_ISqrt(r_limit_sq_i - y * y);

            // Left Wing
            int left_end = min(x_limit_circ, -LDS_RADIUS - 1);
            [loop]
            for (int x = -x_limit_circ; x <= left_end; ++x)
            {
                int2 fetch_pos = clamp(global_pos + int2(x, y), int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
                float4 n_data = tex2Dfetch(SamplerLinearData, fetch_pos);
                BCE_ACCUMULATE_MACRO(n_data, x, spatial_y);
            }

            // Right Wing
            int right_start = LDS_RADIUS + 1;
            [loop]
            for (int x = right_start; x <= x_limit_circ; ++x)
            {
                int2 fetch_pos = clamp(global_pos + int2(x, y), int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
                float4 n_data = tex2Dfetch(SamplerLinearData, fetch_pos);
                BCE_ACCUMULATE_MACRO(n_data, x, spatial_y);
            }
        }

        // 3. Bottom Outer Ring
        [loop]
        for (int y = LDS_RADIUS + 1; y <= max_r; ++y)
        {
            float spatial_y = float(y * y) * inv_2_sigma_s_sq;
            int x_limit_circ = BCE_ISqrt(r_limit_sq_i - y * y);
            int x_start = -x_limit_circ;
            int x_end   =  x_limit_circ;

            [loop]
            for (int x = x_start; x <= x_end; ++x)
            {
                int2 fetch_pos = clamp(global_pos + int2(x, y), int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
                float4 n_data = tex2Dfetch(SamplerLinearData, fetch_pos);
                BCE_ACCUMULATE_MACRO(n_data, x, spatial_y);
            }
        }
    }

    // -------------------------------------------------------------
    // PHASE 5: MULTI-SCALE RECONSTRUCTION
    // -------------------------------------------------------------
    if (stats_w_macro < BCE_FLT_MIN)
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

    // Edge-preserving octave base layers
    float base_micro  = (stats_w_micro > BCE_FLT_MIN) ? (stats_log_micro / stats_w_micro) : log2_center;
    float base_medium = (stats_w_med   > BCE_FLT_MIN) ? (stats_log_med   / stats_w_med)   : base_micro;
    float base_macro  = (stats_log_macro / stats_w_macro);

    // Exact telescoping detail decomposition:
    // (center - base_micro) + (base_micro - base_medium) + (base_medium - base_macro) = center - base_macro
    float diff_micro  = log2_center - base_micro;
    float diff_medium = base_micro  - base_medium;
    float diff_macro  = base_medium - base_macro;

    // Per-band diminishing-returns saturation (odd-symmetric, slope <= 1 at zero)
    float comp_medium = diff_medium;
    float comp_macro  = diff_macro;

    [branch]
    if (bNonRiemannianPerception)
    {
        comp_medium = BCE_DetailSaturate(diff_medium);
        comp_macro  = BCE_DetailSaturate(diff_macro);
    }

    float adaptive_mod = 1.0;
    [branch]
    if (bAdaptiveStrength)
    {
        adaptive_mod = CalculateAdaptiveStrength(stats_log_macro, stats_sq_macro, stats_w_macro, min_log, max_log, log2_center, 1.0, iAdaptiveMode);
    }

    float norm_luma = luma_lin / whitePt;
    float minCompNorm = GetMinComponent(color_lin) / whitePt;
    float zone_mod = GetZoneProtection(norm_luma, minCompNorm, fShadowProtection, fMidtoneProtection, fHighlightProtection, fNegativeProtection);

    float micro_part = 0.0;
    float total_diff = 0.0;
    [branch]
    if (bEnableMultiScale)
    {
        // Micro: RCAS-class sharpener (own contrast adaptivity, not modulated by Adaptive Strength)
        micro_part = BCE_RobustMicroStops(local_center, inv_white, activeSpaceEarly <= 1) * fStrengthMaster;

        total_diff = micro_part +
                     (comp_medium * fStrengthMedium +
                      comp_macro  * fStrengthMacro) * fStrengthMaster * adaptive_mod;
    }
    else
    {
        float diff_legacy = log2_center - base_macro;
        float comp_legacy = diff_legacy;
        if (bNonRiemannianPerception)
        {
            comp_legacy = BCE_DetailSaturate(diff_legacy);
        }
        total_diff = comp_legacy * fStrength * adaptive_mod;
    }

    float effective_diff = total_diff * zone_mod;

#if !defined(__RESHADE_PERFORMANCE_MODE__) || !__RESHADE_PERFORMANCE_MODE__
    if (iDebugMode == 1)
    {
        float3 dbg = saturate(log2(stats_w_macro + 1.0) * 0.1).xxx;
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 2)
    {
        float mean_diff = base_macro - log2_center;
        float v = max(0.0, (stats_sq_macro / stats_w_macro) - mean_diff * mean_diff);
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
        float3 dbg = lerp(float3(0, 0, 1), float3(1, 0, 0), saturate(abs(effective_diff) * 2.0));
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 9)
    {
        float mean_diff = base_macro - log2_center;
        float v = max(0.0, (stats_sq_macro / stats_w_macro) - mean_diff * mean_diff);
        float r = max_log - min_log;
        float e = log2(1.0 + v) * (1.0 + r * 0.1);
        WriteDebugOut(global_pos, float3(e * 0.25, e * 0.125, 0.0), src.a);
        return;
    }
    if (iDebugMode == 13) // Micro Sharpening (signed): orange = brighten, blue = darken
    {
        float v = micro_part * zone_mod;
        float3 dbg = (v >= 0.0) ? float3(saturate(v * 8.0), saturate(v * 4.0), 0.0) : float3(0.0, saturate(-v * 4.0), saturate(-v * 8.0));
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
    if (iDebugMode == 14) // Medium Band (Clarity & Edges)
    {
        float v = saturate(abs(diff_medium) * 5.0);
        WriteDebugOut(global_pos, float3(0, v, v * 0.5), src.a);
        return;
    }
    if (iDebugMode == 15) // Macro Band (Broad-Scale Depth)
    {
        float v = saturate(abs(diff_macro) * 3.0);
        WriteDebugOut(global_pos, float3(v * 0.5, 0, v), src.a);
        return;
    }
    if (iDebugMode == 16) // Multi-Scale Composite Delta
    {
        float3 dbg = (effective_diff >= 0.0) ?
            lerp(float3(0, 0, 0), float3(1, 0.5, 0), saturate(effective_diff * 2.0)) :
            lerp(float3(0, 0, 0), float3(0, 0.5, 1), saturate(-effective_diff * 2.0));
        WriteDebugOut(global_pos, dbg, src.a);
        return;
    }
#endif

    // Enforced bit-exact neutrality: sub-threshold deltas bypass transcoding
    if (abs(effective_diff) < BCE_NEUTRAL_LOG2_EPS)
    {
        tex2Dstore(StorageBilateralOut, global_pos, src);
        return;
    }

    // Direct stop-domain ratio scaling across combined frequency bands
    float ratio = clamp(exp2(effective_diff), RATIO_MIN, RATIO_MAX);
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

technique BilateralContrast_v93 <
    ui_label = "Bilateral Contrast v9.3.0 (RCAS-Class Micro + Bilateral Bands)";
    ui_tooltip = "Multi-scale stop-domain detail enhancement with edge-aware (bilateral) base layers\n"
                 "and an RCAS-class robust micro sharpener on perceptual luminance.\n\n"
                 "V9.3.0 Changes:\n"
                 "- Micro Edge Halo Control (0 = AMD RCAS limit, 1 = linear-light RCAS limit)\n"
                 "- HDR ceiling for the micro sharpener (no sparkles pushed above paper white)\n"
                 "- Micro Sharpness default 0.8 (noise gain of Lilium's HDR RCAS default)\n\n"
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
