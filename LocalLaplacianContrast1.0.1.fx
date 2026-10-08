/**
 * Local Laplacian Contrast Enhancement - FAST LOCAL LAPLACIAN EDITION
 *
 * Design Philosophy: PRECISION AND QUALITY OVER PERFORMANCE
 *
 * Algorithm
 *   Local Laplacian Filter (Paris et al., 2011) in its fast formulation
 *   (Aubry et al., 2014), operating on stop-domain (log2) luminance:
 *     1. N reference intensities g_k span a configurable range of stops.
 *     2. For every g_k the image is remapped around g_k (detail bump, see below)
 *        and a Gaussian pyramid of the result is built.
 *     3. At every pyramid level each output Laplacian coefficient is linearly
 *        interpolated between the two reference levels that bracket the local
 *        Gaussian-pyramid intensity of the INPUT at that level.
 *     4. The interpolated coefficients are collapsed into the final image.
 *   Because every coefficient is taken from a remap anchored at the local intensity,
 *   detail is boosted relative to its own surroundings, not relative to an
 *   edge-filtered base layer. Large edges are left alone by construction
 *   (no bilateral-style halos or gradient reversal from a wrong base layer).
 *
 * Implementation notes
 *   - Delta formulation: the shader builds the pyramid of the remap *difference*
 *     B_k = bump(I - g_k) instead of the remapped image itself. By linearity,
 *         I_out = I + collapse( a_l * sum_k w_k * Laplacian(B_k)[l] )
 *     is mathematically identical to the classic algorithm, and with every band
 *     amount at 0 the output delta is exactly 0 (bit-exact passthrough, no
 *     pyramid reconstruction round-off).
 *   - Remap ("detail bump"): identity + a * d * exp(-0.5 * (d / sigma)^2).
 *     Finite slope 1 + a at d = 0 (the classic power-law remap has infinite slope
 *     there and amplifies noise without bound), returns smoothly to identity for
 *     |d| >> sigma (edges are preserved), and is strictly monotone for
 *     -1 < a < exp(1.5) / 2 = 2.2408 (the sliders are limited to that range).
 *   - Band amounts: Micro = pyramid level 0, Medium = level 1, Macro = levels 2..7.
 *     A per-level multiplier on the interpolated coefficients is exactly equivalent
 *     to running the filter with the bump amplitude scaled for that level.
 *   - The atlas layout stores the N reference slices of a pyramid level in one
 *     texture (GRID_X x GRID_Y cells), so the reconstruction pass reads exactly the
 *     two slices it needs with tex2Dfetch (no dynamic indexing of samplers).
 *   - Pyramid operators: Burt-Adelson 5-tap binomial reduce and the matching
 *     polyphase expand, clamp-to-edge borders. Level 0 of the bump pyramid is
 *     evaluated on the fly (never stored). Level sizes are ceil(size / 2).
 *   - Pass chain (23 passes): pre-pass, input pyramid (7), bump pyramid (7),
 *     reconstruction (7), final composite + encode (1).
 *   - ReShade refuses to compile a pass whose shader can sample its own render target and
 *     checks this statically. Every pyramid level therefore has dedicated fetch functions and
 *     the reduce / reconstruct bodies are instantiated per level by macros, so a pass only
 *     references the samplers it actually reads.
 *
 * Differences from Bilateral Contrast
 *   - Luma only. Chromaticity is untouched (RGB is scaled by the luma ratio), but there
 *     is no ICtCp chroma edge-stopping, adaptive radius, edge-detector selection or
 *     adaptive-strength mode; the pyramid scale replaces the filter radius.
 *   - Reference range: pixels more than one detail threshold outside
 *     [Range Min, Range Max] (stops relative to white) are not modified.
 *   - The reference spacing must be fine enough for linear interpolation. The effective
 *     threshold is raised to 2x the reference spacing if the grid is too coarse
 *     (at that limit the interpolation changes the local gain by <= ~9%).
 *   - Non-Riemannian saturation is applied to the per-level CHANGE (delta) as a limiter
 *     (odd, slope exactly 1 at zero). Bujack et al., PNAS 2022 inspired; not a
 *     reimplementation of their metric.
 *
 * Cost (FP32, GRID 4x4 = 16 reference levels)
 *   VRAM  : ~16 B per pixel for the bump pyramid atlases (+33%) = ~45 MB at 1080p,
 *           ~180 MB at 4K, plus ~12 B per pixel for the small scalar pyramids.
 *   Time  : the first reduce pass evaluates 25 taps x 16 slices per half-resolution
 *           texel. Precision and quality are preferred over speed throughout.
 *
 * Precision note
 *   exp / log / pow / sqrt run at the driver's precision (HLSL / Vulkan permit a few
 *   ULP). Constants and the passthrough paths are exact.
 *
 * Version: 1.0.1
 * Changes since 1.0.0:
 * - Fixed: ReShade error X3020 ("cannot sample from texture that is also used as render target").
 *   The level-switch fetch helpers referenced every level sampler in every pass. Each pyramid level
 *   now has dedicated fetch functions and the reduce / reconstruct bodies are macro-instantiated per
 *   level, so no pass can reach the sampler of its own render target.
 * - Fixed: kernel array name collided with the pyramid-width macro BCE_W5.
 *
 * Requires: ReShade 4.0.0+, a renderer with R32F / RGBA32F render targets,
 *           texture dimension limit >= GRID_X * ceil(width / 2) and GRID_Y * ceil(height / 2)
 *           (16384 satisfies up to 8K with the default 4x4 grid).
 *
 * Author: startuga
 * Formatter: Strict Opinionated Style (Allman, 4-space, Aligned Macros)
 */

#include "ReShade.fxh"

// ==============================================================================
// 0. Compilation Guard & Pre-Processor Configuration
// ==============================================================================

#if __RESHADE__ < 40000
    #error "Local Laplacian Contrast requires ReShade 4.0.0 or newer."
#endif

#if !defined(BUFFER_WIDTH) || !defined(BUFFER_HEIGHT)
    #error "Local Laplacian Contrast: Missing BUFFER_WIDTH/HEIGHT. ReShade.fxh injection failed."
#endif

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

#ifndef BUFFER_COLOR_BIT_DEPTH
    #define BUFFER_COLOR_BIT_DEPTH 8
#endif

// Reference-level atlas layout: N = GRID_X * GRID_Y reference intensities.
// More levels = finer interpolation (4x4 = 16 default, 4x6 = 24, 4x8 = 32).
// Keep GRID_Y * ceil(height / 2) and GRID_X * ceil(width / 2) <= 16384.
#ifndef BCE_LLF_GRID_X
    #define BCE_LLF_GRID_X 4
#endif

#ifndef BCE_LLF_GRID_Y
    #define BCE_LLF_GRID_Y 4
#endif

// 1: float32 bump-pyramid atlases (Mastering Standard)
// 0: float16 atlases (half the VRAM; ~0.002-0.004 stop quantization of the detail layers)
#ifndef BCE_LLF_USE_FP32
    #define BCE_LLF_USE_FP32 1
#endif

#if BCE_LLF_USE_FP32
    #define BCE_LLF_FORMAT R32F
#else
    #define BCE_LLF_FORMAT R16F
#endif

#if !defined(__RESHADE_PERFORMANCE_MODE__) || !__RESHADE_PERFORMANCE_MODE__
    #define BCE_DBG iDebugMode
#else
    #define BCE_DBG 0
#endif

// Pyramid level sizes: level l+1 = ceil(level l / 2). Identical integer arithmetic
// is used by BCE_LevelDims() in the shaders.
#define BCE_W0 BUFFER_WIDTH
#define BCE_H0 BUFFER_HEIGHT
#define BCE_W1 ((BCE_W0 + 1) / 2)
#define BCE_H1 ((BCE_H0 + 1) / 2)
#define BCE_W2 ((BCE_W1 + 1) / 2)
#define BCE_H2 ((BCE_H1 + 1) / 2)
#define BCE_W3 ((BCE_W2 + 1) / 2)
#define BCE_H3 ((BCE_H2 + 1) / 2)
#define BCE_W4 ((BCE_W3 + 1) / 2)
#define BCE_H4 ((BCE_H3 + 1) / 2)
#define BCE_W5 ((BCE_W4 + 1) / 2)
#define BCE_H5 ((BCE_H4 + 1) / 2)
#define BCE_W6 ((BCE_W5 + 1) / 2)
#define BCE_H6 ((BCE_H5 + 1) / 2)
#define BCE_W7 ((BCE_W6 + 1) / 2)
#define BCE_H7 ((BCE_H6 + 1) / 2)

// ==============================================================================
// 1. High-Precision Constants & Color Science Definitions
// ==============================================================================

static const float BCE_FLT_MIN             = 1.175494351e-38;

static const int   BCE_LLF_LEVELS          = 8;
static const int   BCE_N                   = BCE_LLF_GRID_X * BCE_LLF_GRID_Y;

// Minimum ratio of the effective detail threshold to the reference-level spacing
static const float BCE_SIGMA_OVER_STEP     = 2.0;

// Remap monotonicity limits: slope = 1 + a * exp(-u^2/2) * (1 - u^2) stays > 0 for
// -1 < a < exp(1.5) / 2 = 2.2408
static const float BCE_AMOUNT_MAX          = 2.2;
static const float BCE_AMOUNT_MIN          = -0.95;

static const float RATIO_MIN               = 0.0001;
static const float RATIO_MAX               = 10000.0;

// Neutral passthrough: |delta log2| below this cannot perturb an 8-bit output
static const float BCE_NEUTRAL_LOG2_EPS    = 1e-7;

static const float SRGB_THRESHOLD_EOTF     = 0.04045;
static const float SRGB_THRESHOLD_OETF     = (0.04045 / 12.92);

static const float3 Luma709                = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020               = float3(0.2627, 0.6780, 0.0593);

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

// Burt-Adelson 5-tap binomial kernel [1 4 6 4 1] / 16
static const float BCE_KERNEL5[5] = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };

// ==============================================================================
// 2. Texture & System Config
// ==============================================================================

#define BCE_POINT_CLAMP MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP;

texture2D TextureBackBuffer : COLOR;
sampler2D SamplerBackBuffer { Texture = TextureBackBuffer; BCE_POINT_CLAMP };

// Full-resolution log2 luminance (absolute, nits domain)
texture2D TexLogLuma { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R32F; };
sampler2D SamplerLogLuma { Texture = TexLogLuma; BCE_POINT_CLAMP };

// Gaussian pyramid of the (range-clamped) input log2 luminance, levels 1..7
texture2D TexGI1 { Width = BCE_W1; Height = BCE_H1; Format = R32F; };
texture2D TexGI2 { Width = BCE_W2; Height = BCE_H2; Format = R32F; };
texture2D TexGI3 { Width = BCE_W3; Height = BCE_H3; Format = R32F; };
texture2D TexGI4 { Width = BCE_W4; Height = BCE_H4; Format = R32F; };
texture2D TexGI5 { Width = BCE_W5; Height = BCE_H5; Format = R32F; };
texture2D TexGI6 { Width = BCE_W6; Height = BCE_H6; Format = R32F; };
texture2D TexGI7 { Width = BCE_W7; Height = BCE_H7; Format = R32F; };
sampler2D SamplerGI1 { Texture = TexGI1; BCE_POINT_CLAMP };
sampler2D SamplerGI2 { Texture = TexGI2; BCE_POINT_CLAMP };
sampler2D SamplerGI3 { Texture = TexGI3; BCE_POINT_CLAMP };
sampler2D SamplerGI4 { Texture = TexGI4; BCE_POINT_CLAMP };
sampler2D SamplerGI5 { Texture = TexGI5; BCE_POINT_CLAMP };
sampler2D SamplerGI6 { Texture = TexGI6; BCE_POINT_CLAMP };
sampler2D SamplerGI7 { Texture = TexGI7; BCE_POINT_CLAMP };

// Gaussian pyramids of the bump images B_k, one atlas per level (GRID_X x GRID_Y cells), levels 1..7
texture2D TexGB1 { Width = BCE_LLF_GRID_X * BCE_W1; Height = BCE_LLF_GRID_Y * BCE_H1; Format = BCE_LLF_FORMAT; };
texture2D TexGB2 { Width = BCE_LLF_GRID_X * BCE_W2; Height = BCE_LLF_GRID_Y * BCE_H2; Format = BCE_LLF_FORMAT; };
texture2D TexGB3 { Width = BCE_LLF_GRID_X * BCE_W3; Height = BCE_LLF_GRID_Y * BCE_H3; Format = BCE_LLF_FORMAT; };
texture2D TexGB4 { Width = BCE_LLF_GRID_X * BCE_W4; Height = BCE_LLF_GRID_Y * BCE_H4; Format = BCE_LLF_FORMAT; };
texture2D TexGB5 { Width = BCE_LLF_GRID_X * BCE_W5; Height = BCE_LLF_GRID_Y * BCE_H5; Format = BCE_LLF_FORMAT; };
texture2D TexGB6 { Width = BCE_LLF_GRID_X * BCE_W6; Height = BCE_LLF_GRID_Y * BCE_H6; Format = BCE_LLF_FORMAT; };
texture2D TexGB7 { Width = BCE_LLF_GRID_X * BCE_W7; Height = BCE_LLF_GRID_Y * BCE_H7; Format = BCE_LLF_FORMAT; };
sampler2D SamplerGB1 { Texture = TexGB1; BCE_POINT_CLAMP };
sampler2D SamplerGB2 { Texture = TexGB2; BCE_POINT_CLAMP };
sampler2D SamplerGB3 { Texture = TexGB3; BCE_POINT_CLAMP };
sampler2D SamplerGB4 { Texture = TexGB4; BCE_POINT_CLAMP };
sampler2D SamplerGB5 { Texture = TexGB5; BCE_POINT_CLAMP };
sampler2D SamplerGB6 { Texture = TexGB6; BCE_POINT_CLAMP };
sampler2D SamplerGB7 { Texture = TexGB7; BCE_POINT_CLAMP };

// Reconstructed delta pyramid (output of each level, scalar), levels 1..7
texture2D TexOut1 { Width = BCE_W1; Height = BCE_H1; Format = R32F; };
texture2D TexOut2 { Width = BCE_W2; Height = BCE_H2; Format = R32F; };
texture2D TexOut3 { Width = BCE_W3; Height = BCE_H3; Format = R32F; };
texture2D TexOut4 { Width = BCE_W4; Height = BCE_H4; Format = R32F; };
texture2D TexOut5 { Width = BCE_W5; Height = BCE_H5; Format = R32F; };
texture2D TexOut6 { Width = BCE_W6; Height = BCE_H6; Format = R32F; };
texture2D TexOut7 { Width = BCE_W7; Height = BCE_H7; Format = R32F; };
sampler2D SamplerOut1 { Texture = TexOut1; BCE_POINT_CLAMP };
sampler2D SamplerOut2 { Texture = TexOut2; BCE_POINT_CLAMP };
sampler2D SamplerOut3 { Texture = TexOut3; BCE_POINT_CLAMP };
sampler2D SamplerOut4 { Texture = TexOut4; BCE_POINT_CLAMP };
sampler2D SamplerOut5 { Texture = TexOut5; BCE_POINT_CLAMP };
sampler2D SamplerOut6 { Texture = TexOut6; BCE_POINT_CLAMP };
sampler2D SamplerOut7 { Texture = TexOut7; BCE_POINT_CLAMP };

// ==============================================================================
// 3. UI Parameters
// ==============================================================================

uniform float fStrengthMicro <
    ui_type = "slider";
    ui_label = "Micro-Texture (Level 0)";
    ui_min = -1.0; ui_max = 2.2; ui_step = 0.01;
    ui_tooltip = "Detail amount for the finest pyramid level (roughly 2-4 px features).\n"
                 "The local gain on small details is about 1 + amount (2.0 = ~3x). Negative values smooth.\n"
                 "Limited to 2.2: above exp(1.5)/2 = 2.24 the remap would stop being monotone.\n"
                 "Very high values also amplify 8-bit quantization steps in smooth gradients.";
    ui_category = "Local Laplacian Bands";
> = 2.0;

uniform float fStrengthMedium <
    ui_type = "slider";
    ui_label = "Mid-Frequency Clarity (Level 1)";
    ui_min = -1.0; ui_max = 2.2; ui_step = 0.01;
    ui_tooltip = "Detail amount for pyramid level 1 (roughly 4-8 px features): structural edges, bevels, contours.";
    ui_category = "Local Laplacian Bands";
> = 0.5;

uniform float fStrengthMacro <
    ui_type = "slider";
    ui_label = "Macro Depth (Levels 2-7)";
    ui_min = -1.0; ui_max = 2.2; ui_step = 0.01;
    ui_tooltip = "Detail amount for all coarser levels (8 px and above): local illumination gradients,\n"
                 "shape curvature, 3D depth pop. Halo-free by construction; large edges (above the\n"
                 "Detail Threshold) are preserved.";
    ui_category = "Local Laplacian Bands";
> = 0.25;

uniform float fStrengthMaster <
    ui_type = "slider";
    ui_label = "Master Scale Multiplier";
    ui_min = 0.0; ui_max = 3.0; ui_step = 0.01;
    ui_tooltip = "Multiplies all band amounts (result is limited to the monotone range -0.95 .. 2.2).";
    ui_category = "Local Laplacian Bands";
> = 1.00;

uniform float fDetailThreshold <
    ui_type = "slider";
    ui_label = "Detail Threshold (Stops)";
    ui_min = 0.25; ui_max = 4.0; ui_step = 0.01;
    ui_units = "stops";
    ui_tooltip = "Local contrast below about this many stops is treated as detail and enhanced;\n"
                 "contrast far above it is treated as an edge and left alone.\n"
                 "1.3 stops (2.5x) is the classic detail-enhancement setting.\n"
                 "Raised automatically to 2x the reference spacing if the reference grid is too coarse.";
    ui_category = "Reference Levels";
> = 1.5;

uniform float fRangeMin <
    ui_type = "slider";
    ui_label = "Reference Range Min (Stops vs White)";
    ui_min = -16.0; ui_max = -1.0; ui_step = 0.1;
    ui_units = "stops";
    ui_tooltip = "Darkest luminance that is processed, in stops relative to the white point.\n"
                 "Pixels darker than this (by more than the Detail Threshold) are left untouched.";
    ui_category = "Reference Levels";
> = -9.0;

uniform float fRangeMax <
    ui_type = "slider";
    ui_label = "Reference Range Max (Stops vs White)";
    ui_min = 0.0; ui_max = 8.0; ui_step = 0.1;
    ui_units = "stops";
    ui_tooltip = "Brightest luminance that is processed, in stops relative to the white point.\n"
                 "SDR content never exceeds 0; raise it for HDR / scRGB highlights.\n"
                 "Spacing = (Max - Min) / (N - 1). A narrower range gives a finer grid for the same N.";
    ui_category = "Reference Levels";
> = 1.0;

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

uniform bool bNonRiemannianPerception <
    ui_label = "Enable Non-Riemannian Limiter";
    ui_tooltip = "Applies a Bujack-inspired diminishing-returns saturation to the change produced by each\n"
                 "pyramid level:\n"
                 "    f(d) = (k / g) * ln(1 + (|d| / k)^g)\n"
                 "with k = Saturation Knee and g = Perceptual Saturation Exponent.\n"
                 "Odd, slope exactly 1 at zero: small changes pass unchanged, extreme local boosts are limited.";
    ui_category = "Non-Riemannian Perception";
    ui_category_toggle = true;
> = true;

uniform float fDiminishingReturnsExponent <
    ui_type = "slider";
    ui_label = "Perceptual Saturation Exponent";
    ui_min = 1.00; ui_max = 2.00; ui_step = 0.01;
    ui_tooltip = "Exponent g of the saturation curve (clamped to 1..2).\n"
                 "1.0: logarithmic compression, slope exactly 1 at zero.\n"
                 "> 1.0: additionally suppresses very small changes (soft dead-zone).\n"
                 "Values below 1.0 are not allowed: the curve slope diverges at zero and amplifies noise.";
    ui_category = "Non-Riemannian Perception";
> = 1.00;

uniform float fSaturationKnee <
    ui_type = "slider";
    ui_label = "Saturation Knee (Stops)";
    ui_min = 0.10; ui_max = 8.00; ui_step = 0.01;
    ui_units = "stops";
    ui_tooltip = "Magnitude of the per-level change (in stops) at which compression becomes significant.\n"
                 "Smaller: limit earlier. Larger: more linear.";
    ui_category = "Non-Riemannian Perception";
> = 2.00;

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
    ui_items = "Off\0Enhancement Map\0Micro Band (Level 0)\0Coarse Bands (Level 1+)\0Reference Position\0Black Pixels\0Zone Map\0Negative Values\0Signed Luminance\0";
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

// ln(1 + x) for x >= 0 without catastrophic cancellation near zero (Kahan's formulation).
float BCE_Log1p(float x)
{
    float u = 1.0 + x;
    return (u == 1.0) ? x : log(u) * (x / (u - 1.0));
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

// Odd-symmetric diminishing-returns limiter applied to a signed value in stops.
//   f(d) = sign(d) * (k / g) * ln(1 + (|d| / k)^g),   g in [1, 2],  k = Saturation Knee
// g = 1 : slope at zero is exactly 1.   g > 1 : slope at zero is 0 (soft dead-zone).
float BCE_DetailSaturate(float d)
{
    float g = clamp(fDiminishingReturnsExponent, 1.0, 2.0);
    float k = max(fSaturationKnee, 1e-3);
    float a = abs(d) / k;
    return sign(d) * (k / g) * BCE_Log1p(PowNonNegPreserveZero(a, g));
}

float BCE_ApplyLimiter(float d)
{
    return bNonRiemannianPerception ? BCE_DetailSaturate(d) : d;
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

float3 BCE_DebugEncode(float3 dbg)
{
    int activeSpace = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float whitePt = GetResolvedWhitePoint();

    [branch]
    if (activeSpace == 4)
    {
        return HLG_OETF(dbg * whitePt);
    }
    else if (activeSpace == 3)
    {
        return PQ_InverseEOTF(dbg * whitePt);
    }
    else if (activeSpace == 2)
    {
        return dbg * (whitePt / SCRGB_WHITE_NITS);
    }

    return sRGB_OETF(saturate(dbg));
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
// 7. Local Laplacian Core
// ==============================================================================

// Reference-level grid and detail-remap parameters, all derived from uniforms.
struct BCEParams
{
    float g0;       // absolute log2 luminance of reference level 0
    float step;     // spacing between reference levels (stops)
    float sigma;    // effective detail threshold (stops)
    float lo;       // working log2-luma clamp floor
    float hi;       // working log2-luma clamp ceiling
};

BCEParams BCE_GetParams()
{
    BCEParams P;
    float range = max(fRangeMax - fRangeMin, 1.0);

    P.g0    = log2(max(GetResolvedWhitePoint(), BCE_FLT_MIN)) + fRangeMin;
    P.step  = range / float(BCE_N - 1);
    P.sigma = max(fDetailThreshold, BCE_SIGMA_OVER_STEP * P.step);
    P.lo    = P.g0 - 2.0 * P.sigma;
    P.hi    = P.g0 + range + 2.0 * P.sigma;
    return P;
}

// Unit-amplitude detail bump of the remap r(d) = d + a * bump(d):
//   bump(d) = d * exp(-0.5 * (d / sigma)^2)      (odd, slope 1 at 0, -> 0 for |d| >> sigma)
float BCE_Bump(float d, float sigma)
{
    float u = d / sigma;
    return d * exp(-0.5 * u * u);
}

// Per-band remap amplitude (pyramid level 0 = Micro, 1 = Medium, 2..7 = Macro),
// limited to the strictly monotone range of the remap.
float BCE_BandAmount(int l)
{
    float a = (l == 0) ? fStrengthMicro : ((l == 1) ? fStrengthMedium : fStrengthMacro);
    return clamp(a * fStrengthMaster, BCE_AMOUNT_MIN, BCE_AMOUNT_MAX);
}

// Size of pyramid level l (level 0 = full resolution, ceil(size / 2) per level)
int2 BCE_LevelDims(int l)
{
    int2 d = int2(BUFFER_WIDTH, BUFFER_HEIGHT);

    [unroll]
    for (int i = 0; i < BCE_LLF_LEVELS - 1; ++i)
    {
        if (i < l)
        {
            d = (d + 1) / 2;
        }
    }
    return d;
}

// Working log2 luminance at full resolution (clamped coordinates and value range)
float BCE_I0(int2 p, BCEParams P)
{
    int2 q = clamp(p, int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
    return clamp(tex2Dfetch(SamplerLogLuma, q).r, P.lo, P.hi);
}

// ------------------------------------------------------------------------------
// Level-specific fetch primitives
//
// ReShade rejects a pass whose shader can sample the texture it renders to, and it
// checks this statically: a runtime "if (level == n)" switch over all level samplers
// would count as sampling every level in every pass. Every pyramid level therefore gets
// its own tiny fetch function, and the generic reduce / reconstruct bodies below are
// instantiated once per level through macros that receive the fetch function names.
// Each pass then references only the samplers it really reads.
//
//   BCE_FGI_n   input Gaussian pyramid, level n   (n = 0: working log-luma, on the fly)
//   BCE_FGB_n   bump-pyramid atlas, level n       (n = 0: bump evaluated on the fly)
//   BCE_FOUT_n  reconstructed delta pyramid, level n
//   BCE_FGB_8 / BCE_FOUT_8 are zero placeholders for "no coarser level" at level 7
// ------------------------------------------------------------------------------

int2 BCE_AtlasPos(int2 dims, int2 p, int k)
{
    return int2((k % BCE_LLF_GRID_X) * dims.x + p.x, (k / BCE_LLF_GRID_X) * dims.y + p.y);
}

float BCE_FGI_0(int2 p, BCEParams P) { return BCE_I0(p, P); }
float BCE_FGI_1(int2 p, BCEParams P) { return tex2Dfetch(SamplerGI1, p).r; }
float BCE_FGI_2(int2 p, BCEParams P) { return tex2Dfetch(SamplerGI2, p).r; }
float BCE_FGI_3(int2 p, BCEParams P) { return tex2Dfetch(SamplerGI3, p).r; }
float BCE_FGI_4(int2 p, BCEParams P) { return tex2Dfetch(SamplerGI4, p).r; }
float BCE_FGI_5(int2 p, BCEParams P) { return tex2Dfetch(SamplerGI5, p).r; }
float BCE_FGI_6(int2 p, BCEParams P) { return tex2Dfetch(SamplerGI6, p).r; }
float BCE_FGI_7(int2 p, BCEParams P) { return tex2Dfetch(SamplerGI7, p).r; }

float BCE_FGB_0(int2 dims, int2 p, int k, BCEParams P)
{
    return BCE_Bump(BCE_I0(p, P) - (P.g0 + float(k) * P.step), P.sigma);
}
float BCE_FGB_1(int2 dims, int2 p, int k, BCEParams P) { return tex2Dfetch(SamplerGB1, BCE_AtlasPos(dims, p, k)).r; }
float BCE_FGB_2(int2 dims, int2 p, int k, BCEParams P) { return tex2Dfetch(SamplerGB2, BCE_AtlasPos(dims, p, k)).r; }
float BCE_FGB_3(int2 dims, int2 p, int k, BCEParams P) { return tex2Dfetch(SamplerGB3, BCE_AtlasPos(dims, p, k)).r; }
float BCE_FGB_4(int2 dims, int2 p, int k, BCEParams P) { return tex2Dfetch(SamplerGB4, BCE_AtlasPos(dims, p, k)).r; }
float BCE_FGB_5(int2 dims, int2 p, int k, BCEParams P) { return tex2Dfetch(SamplerGB5, BCE_AtlasPos(dims, p, k)).r; }
float BCE_FGB_6(int2 dims, int2 p, int k, BCEParams P) { return tex2Dfetch(SamplerGB6, BCE_AtlasPos(dims, p, k)).r; }
float BCE_FGB_7(int2 dims, int2 p, int k, BCEParams P) { return tex2Dfetch(SamplerGB7, BCE_AtlasPos(dims, p, k)).r; }
float BCE_FGB_8(int2 dims, int2 p, int k, BCEParams P) { return 0.0; }

float BCE_FOUT_1(int2 p) { return tex2Dfetch(SamplerOut1, p).r; }
float BCE_FOUT_2(int2 p) { return tex2Dfetch(SamplerOut2, p).r; }
float BCE_FOUT_3(int2 p) { return tex2Dfetch(SamplerOut3, p).r; }
float BCE_FOUT_4(int2 p) { return tex2Dfetch(SamplerOut4, p).r; }
float BCE_FOUT_5(int2 p) { return tex2Dfetch(SamplerOut5, p).r; }
float BCE_FOUT_6(int2 p) { return tex2Dfetch(SamplerOut6, p).r; }
float BCE_FOUT_7(int2 p) { return tex2Dfetch(SamplerOut7, p).r; }
float BCE_FOUT_8(int2 p) { return 0.0; }

// Expand (polyphase of the Burt-Adelson up-sampler), one axis.
//   even fine pixel 2i   : coarse i-1, i, i+1 with weights 1/8, 6/8, 1/8
//   odd  fine pixel 2i+1 : coarse i, i+1       with weights 1/2, 1/2
void BCE_ExpandSetup(int p, out int i0, out float3 w)
{
    int i = p / 2;
    bool odd = (p - 2 * i) != 0;
    i0 = i - 1;
    w  = odd ? float3(0.0, 0.5, 0.5) : float3(0.125, 0.75, 0.125);
}

// Reduce: one texel of the next level of the INPUT log-luma pyramid.
//   SRC = fetch function of the finer level (BCE_FGI_{LEVEL-1})
#define BCE_DEFINE_GI_DOWN(NAME, LEVEL, SRC)                                                                                   \
float NAME(int2 c)                                                                                                             \
{                                                                                                                              \
    BCEParams P = BCE_GetParams();                                                                                             \
    int2 sdims = BCE_LevelDims((LEVEL) - 1);                                                                                   \
    float sum = 0.0;                                                                                                           \
    [unroll]                                                                                                                   \
    for (int n = -2; n <= 2; ++n)                                                                                              \
    {                                                                                                                          \
        [unroll]                                                                                                               \
        for (int m = -2; m <= 2; ++m)                                                                                          \
        {                                                                                                                      \
            int2 q = clamp(2 * c + int2(m, n), int2(0, 0), sdims - 1);                                                         \
            sum += BCE_KERNEL5[m + 2] * BCE_KERNEL5[n + 2] * SRC(q, P);                                                        \
        }                                                                                                                      \
    }                                                                                                                          \
    return sum;                                                                                                                \
}

// Reduce: one atlas texel of the next level of the BUMP pyramid.
//   SRC = atlas fetch function of the finer level (BCE_FGB_{LEVEL-1})
#define BCE_DEFINE_GB_DOWN(NAME, LEVEL, SRC)                                                                                   \
float NAME(int2 apos)                                                                                                          \
{                                                                                                                              \
    BCEParams P = BCE_GetParams();                                                                                             \
    int2 dims  = BCE_LevelDims(LEVEL);                                                                                         \
    int2 sdims = BCE_LevelDims((LEVEL) - 1);                                                                                   \
    int cell_x = apos.x / dims.x;                                                                                              \
    int cell_y = apos.y / dims.y;                                                                                              \
    int k      = cell_x + BCE_LLF_GRID_X * cell_y;                                                                             \
    int2 c     = int2(apos.x - cell_x * dims.x, apos.y - cell_y * dims.y);                                                     \
    float sum = 0.0;                                                                                                           \
    [unroll]                                                                                                                   \
    for (int n = -2; n <= 2; ++n)                                                                                              \
    {                                                                                                                          \
        [unroll]                                                                                                               \
        for (int m = -2; m <= 2; ++m)                                                                                          \
        {                                                                                                                      \
            int2 q = clamp(2 * c + int2(m, n), int2(0, 0), sdims - 1);                                                         \
            sum += BCE_KERNEL5[m + 2] * BCE_KERNEL5[n + 2] * SRC(sdims, q, k, P);                                              \
        }                                                                                                                      \
    }                                                                                                                          \
    return sum;                                                                                                                \
}

// Reconstruction of pyramid level LEVEL at pixel p:
//   own    = limiter( a_l * interp_k( Laplacian(B_k)[l](p) ) ),  k bracketing the input intensity at level l
//   coarse = expand( out[l + 1] )
//   out[l] = own + coarse
//   GI / GB / GBC / OUTC = fetch functions for this level's input pyramid, this level's bump atlas,
//   the next-coarser bump atlas and the next-coarser reconstructed delta.
#define BCE_DEFINE_RECON(NAME, LEVEL, GI, GB, GBC, OUTC)                                                                       \
void NAME(int2 p, BCEParams P, out float own, out float coarse)                                                                \
{                                                                                                                              \
    float gi = GI(p, P);                                                                                                       \
    float t  = clamp((gi - P.g0) / P.step, 0.0, float(BCE_N - 1));                                                             \
    int   k0 = min((int)floor(t), BCE_N - 1);                                                                                  \
    int   k1 = min(k0 + 1, BCE_N - 1);                                                                                         \
    float f  = t - float(k0);                                                                                                  \
    int2 dims = BCE_LevelDims(LEVEL);                                                                                          \
    float lb0 = GB(dims, p, k0, P);                                                                                            \
    float lb1 = GB(dims, p, k1, P);                                                                                            \
    coarse = 0.0;                                                                                                              \
    if ((LEVEL) < BCE_LLF_LEVELS - 1)                                                                                          \
    {                                                                                                                          \
        int2 dimsc = BCE_LevelDims((LEVEL) + 1);                                                                               \
        int bx, by;                                                                                                            \
        float3 wx, wy;                                                                                                         \
        BCE_ExpandSetup(p.x, bx, wx);                                                                                          \
        BCE_ExpandSetup(p.y, by, wy);                                                                                          \
        float e0 = 0.0, e1 = 0.0, eo = 0.0;                                                                                    \
        [unroll]                                                                                                               \
        for (int b = 0; b < 3; ++b)                                                                                            \
        {                                                                                                                      \
            [unroll]                                                                                                           \
            for (int a = 0; a < 3; ++a)                                                                                        \
            {                                                                                                                  \
                int2 q = clamp(int2(bx + a, by + b), int2(0, 0), dimsc - 1);                                                   \
                float w = wx[a] * wy[b];                                                                                       \
                e0 += w * GBC(dimsc, q, k0, P);                                                                                \
                e1 += w * GBC(dimsc, q, k1, P);                                                                                \
                eo += w * OUTC(q);                                                                                             \
            }                                                                                                                  \
        }                                                                                                                      \
        lb0   -= e0;                                                                                                           \
        lb1   -= e1;                                                                                                           \
        coarse = eo;                                                                                                           \
    }                                                                                                                          \
    own = BCE_ApplyLimiter(BCE_BandAmount(LEVEL) * lerp(lb0, lb1, f));                                                         \
}

BCE_DEFINE_GI_DOWN(BCE_GIDown1, 1, BCE_FGI_0)
BCE_DEFINE_GI_DOWN(BCE_GIDown2, 2, BCE_FGI_1)
BCE_DEFINE_GI_DOWN(BCE_GIDown3, 3, BCE_FGI_2)
BCE_DEFINE_GI_DOWN(BCE_GIDown4, 4, BCE_FGI_3)
BCE_DEFINE_GI_DOWN(BCE_GIDown5, 5, BCE_FGI_4)
BCE_DEFINE_GI_DOWN(BCE_GIDown6, 6, BCE_FGI_5)
BCE_DEFINE_GI_DOWN(BCE_GIDown7, 7, BCE_FGI_6)

BCE_DEFINE_GB_DOWN(BCE_GBDown1, 1, BCE_FGB_0)
BCE_DEFINE_GB_DOWN(BCE_GBDown2, 2, BCE_FGB_1)
BCE_DEFINE_GB_DOWN(BCE_GBDown3, 3, BCE_FGB_2)
BCE_DEFINE_GB_DOWN(BCE_GBDown4, 4, BCE_FGB_3)
BCE_DEFINE_GB_DOWN(BCE_GBDown5, 5, BCE_FGB_4)
BCE_DEFINE_GB_DOWN(BCE_GBDown6, 6, BCE_FGB_5)
BCE_DEFINE_GB_DOWN(BCE_GBDown7, 7, BCE_FGB_6)

BCE_DEFINE_RECON(BCE_Recon0, 0, BCE_FGI_0, BCE_FGB_0, BCE_FGB_1, BCE_FOUT_1)
BCE_DEFINE_RECON(BCE_Recon1, 1, BCE_FGI_1, BCE_FGB_1, BCE_FGB_2, BCE_FOUT_2)
BCE_DEFINE_RECON(BCE_Recon2, 2, BCE_FGI_2, BCE_FGB_2, BCE_FGB_3, BCE_FOUT_3)
BCE_DEFINE_RECON(BCE_Recon3, 3, BCE_FGI_3, BCE_FGB_3, BCE_FGB_4, BCE_FOUT_4)
BCE_DEFINE_RECON(BCE_Recon4, 4, BCE_FGI_4, BCE_FGB_4, BCE_FGB_5, BCE_FOUT_5)
BCE_DEFINE_RECON(BCE_Recon5, 5, BCE_FGI_5, BCE_FGB_5, BCE_FGB_6, BCE_FOUT_6)
BCE_DEFINE_RECON(BCE_Recon6, 6, BCE_FGI_6, BCE_FGB_6, BCE_FGB_7, BCE_FOUT_7)
BCE_DEFINE_RECON(BCE_Recon7, 7, BCE_FGI_7, BCE_FGB_7, BCE_FGB_8, BCE_FOUT_8)

// ==============================================================================
// 8. Pass Entry Points
// ==============================================================================

void PS_PrePass(float4 vpos : SV_Position, out float4 outData : SV_Target)
{
    int2 pos = int2(vpos.xy);

    float3 color_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, pos).rgb);
    bool is_invalid = any(IsNan3(color_lin)) || any(IsInf3(color_lin));
    color_lin = is_invalid ? 0.0.xxx : color_lin;

    float luma_lin = GetLuminanceCS(color_lin);
    outData = float4(log2(max(luma_lin, BCE_FLT_MIN)), 0.0, 0.0, 1.0);
}

void PS_GI1(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GIDown1(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GI2(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GIDown2(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GI3(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GIDown3(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GI4(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GIDown4(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GI5(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GIDown5(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GI6(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GIDown6(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GI7(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GIDown7(int2(vpos.xy)), 0.0, 0.0, 1.0); }

void PS_GB1(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GBDown1(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GB2(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GBDown2(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GB3(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GBDown3(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GB4(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GBDown4(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GB5(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GBDown5(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GB6(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GBDown6(int2(vpos.xy)), 0.0, 0.0, 1.0); }
void PS_GB7(float4 vpos : SV_Position, out float4 o : SV_Target0) { o = float4(BCE_GBDown7(int2(vpos.xy)), 0.0, 0.0, 1.0); }

// Reconstruction, coarse to fine
#define BCE_DEFINE_RECON_PS(NAME, FN)                                                                                          \
void NAME(float4 vpos : SV_Position, out float4 o : SV_Target0)                                                                \
{                                                                                                                              \
    float own, coarse;                                                                                                         \
    FN(int2(vpos.xy), BCE_GetParams(), own, coarse);                                                                           \
    o = float4(own + coarse, 0.0, 0.0, 1.0);                                                                                   \
}

BCE_DEFINE_RECON_PS(PS_Recon7, BCE_Recon7)
BCE_DEFINE_RECON_PS(PS_Recon6, BCE_Recon6)
BCE_DEFINE_RECON_PS(PS_Recon5, BCE_Recon5)
BCE_DEFINE_RECON_PS(PS_Recon4, BCE_Recon4)
BCE_DEFINE_RECON_PS(PS_Recon3, BCE_Recon3)
BCE_DEFINE_RECON_PS(PS_Recon2, BCE_Recon2)
BCE_DEFINE_RECON_PS(PS_Recon1, BCE_Recon1)

// Level 0 reconstruction fused with protection, ratio scaling and output encoding
void PS_Final(float4 vpos : SV_Position, out float4 fragColor : SV_Target)
{
    int2 pos = int2(vpos.xy);
    float4 src = tex2Dfetch(SamplerBackBuffer, pos);
    fragColor = src;

    bool is_active = (fStrengthMaster > 0.0) &&
                     (fStrengthMicro != 0.0 || fStrengthMedium != 0.0 || fStrengthMacro != 0.0);

    if (!is_active && BCE_DBG == 0)
    {
        return;
    }

    float whitePt = GetResolvedWhitePoint();

    float3 color_lin = DecodeToLinear(src.rgb);
    bool is_invalid = any(IsNan3(color_lin)) || any(IsInf3(color_lin));
    color_lin = is_invalid ? 0.0.xxx : color_lin;

    float luma_lin = GetLuminanceCS(color_lin);

    if (BCE_DBG == 0 && luma_lin <= BCE_FLT_MIN)
    {
        return;
    }

    // Non-neighborhood debug views
    if (BCE_DBG == 5)
    {
        float3 dbg = (luma_lin <= BCE_FLT_MIN) ? float3(1, 0, 1) : float3(0, 0, 0);
        fragColor = float4(BCE_DebugEncode(dbg), src.a);
        return;
    }
    if (BCE_DBG == 6)
    {
        float3 dbg = GetZoneColor(GetZone(luma_lin / whitePt));
        fragColor = float4(BCE_DebugEncode(dbg), src.a);
        return;
    }
    if (BCE_DBG == 7)
    {
        float3 dbg = (GetMinComponent(color_lin) < 0.0) ? float3(1, 0, 1) : float3(0, 0.1, 0);
        fragColor = float4(BCE_DebugEncode(dbg), src.a);
        return;
    }
    if (BCE_DBG == 8)
    {
        float norm = luma_lin / max(whitePt, BCE_FLT_MIN);
        float stops = log2(max(abs(norm), BCE_FLT_MIN));
        float t = saturate((stops + 6.0) / 8.0);
        float3 dbg = (norm < 0.0) ? float3(0, t, 0) : float3(t, 0, 0);
        fragColor = float4(BCE_DebugEncode(dbg), src.a);
        return;
    }

    BCEParams P = BCE_GetParams();

    if (BCE_DBG == 4)
    {
        // Fractional position between reference levels (red) and reference index (green)
        float t = clamp((BCE_I0(pos, P) - P.g0) / P.step, 0.0, float(BCE_N - 1));
        float3 dbg = float3(frac(t), t / float(BCE_N - 1), 0.0);
        fragColor = float4(BCE_DebugEncode(dbg), src.a);
        return;
    }

    float own0, coarse0;
    BCE_Recon0(pos, P, own0, coarse0);
    float total_diff = own0 + coarse0;

    float norm_luma   = luma_lin / whitePt;
    float minCompNorm = GetMinComponent(color_lin) / whitePt;
    float strength_mod = GetZoneProtection(norm_luma, minCompNorm, fShadowProtection, fMidtoneProtection, fHighlightProtection, fNegativeProtection);

    float effective_diff = total_diff * strength_mod;

    if (BCE_DBG == 1)
    {
        float3 dbg = (effective_diff >= 0.0) ?
            lerp(float3(0, 0, 0), float3(1, 0.5, 0), saturate(effective_diff * 2.0)) :
            lerp(float3(0, 0, 0), float3(0, 0.5, 1), saturate(-effective_diff * 2.0));
        fragColor = float4(BCE_DebugEncode(dbg), src.a);
        return;
    }
    if (BCE_DBG == 2)
    {
        float v = own0 * strength_mod;
        float3 dbg = (v >= 0.0) ? float3(saturate(v * 4.0), saturate(v * 2.0), 0) : float3(0, saturate(-v * 2.0), saturate(-v * 4.0));
        fragColor = float4(BCE_DebugEncode(dbg), src.a);
        return;
    }
    if (BCE_DBG == 3)
    {
        float v = coarse0 * strength_mod;
        float3 dbg = (v >= 0.0) ? float3(saturate(v * 4.0), saturate(v * 2.0), 0) : float3(0, saturate(-v * 2.0), saturate(-v * 4.0));
        fragColor = float4(BCE_DebugEncode(dbg), src.a);
        return;
    }

    // Enforced bit-exact neutrality: sub-threshold deltas bypass transcoding
    if (abs(effective_diff) < BCE_NEUTRAL_LOG2_EPS)
    {
        return;
    }

    // Direct stop-domain ratio scaling: chromaticity is preserved
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

    fragColor = float4(encoded, src.a);
}

// ==============================================================================
// 9. Technique
// ==============================================================================

technique LocalLaplacianContrast <
    ui_label = "Local Laplacian Contrast v1.0.1 (Fast Local Laplacian Filter)";
    ui_tooltip = "Fast Local Laplacian Filter (Paris et al. 2011 / Aubry et al. 2014) on stop-domain luminance.\n"
                 "Detail is boosted relative to its own local intensity, so large edges are preserved without\n"
                 "bilateral-style halos.\n\n"
                 "- 8-level Gaussian / Laplacian pyramid, 16 reference intensities (4x4 atlas by default)\n"
                 "- Smooth, monotone detail remap with finite slope at zero\n"
                 "- Per-band amounts: Micro (level 0), Medium (level 1), Macro (levels 2-7)\n"
                 "- Exact passthrough when all amounts are 0\n"
                 "- Luma only; chromaticity is preserved by ratio scaling\n\n"
                 "Memory-hungry: roughly 45 MB at 1080p and 180 MB at 4K with the default FP32 4x4 atlas.";
>
{
    pass PreCompute
    {
        VertexShader      = PostProcessVS;
        PixelShader       = PS_PrePass;
        RenderTarget      = TexLogLuma;
        VertexCount       = 3;
        PrimitiveTopology = TRIANGLELIST;
    }

    pass InputPyramid1 { VertexShader = PostProcessVS; PixelShader = PS_GI1; RenderTarget = TexGI1; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass InputPyramid2 { VertexShader = PostProcessVS; PixelShader = PS_GI2; RenderTarget = TexGI2; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass InputPyramid3 { VertexShader = PostProcessVS; PixelShader = PS_GI3; RenderTarget = TexGI3; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass InputPyramid4 { VertexShader = PostProcessVS; PixelShader = PS_GI4; RenderTarget = TexGI4; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass InputPyramid5 { VertexShader = PostProcessVS; PixelShader = PS_GI5; RenderTarget = TexGI5; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass InputPyramid6 { VertexShader = PostProcessVS; PixelShader = PS_GI6; RenderTarget = TexGI6; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass InputPyramid7 { VertexShader = PostProcessVS; PixelShader = PS_GI7; RenderTarget = TexGI7; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }

    pass BumpPyramid1 { VertexShader = PostProcessVS; PixelShader = PS_GB1; RenderTarget = TexGB1; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass BumpPyramid2 { VertexShader = PostProcessVS; PixelShader = PS_GB2; RenderTarget = TexGB2; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass BumpPyramid3 { VertexShader = PostProcessVS; PixelShader = PS_GB3; RenderTarget = TexGB3; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass BumpPyramid4 { VertexShader = PostProcessVS; PixelShader = PS_GB4; RenderTarget = TexGB4; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass BumpPyramid5 { VertexShader = PostProcessVS; PixelShader = PS_GB5; RenderTarget = TexGB5; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass BumpPyramid6 { VertexShader = PostProcessVS; PixelShader = PS_GB6; RenderTarget = TexGB6; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass BumpPyramid7 { VertexShader = PostProcessVS; PixelShader = PS_GB7; RenderTarget = TexGB7; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }

    pass Reconstruct7 { VertexShader = PostProcessVS; PixelShader = PS_Recon7; RenderTarget = TexOut7; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass Reconstruct6 { VertexShader = PostProcessVS; PixelShader = PS_Recon6; RenderTarget = TexOut6; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass Reconstruct5 { VertexShader = PostProcessVS; PixelShader = PS_Recon5; RenderTarget = TexOut5; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass Reconstruct4 { VertexShader = PostProcessVS; PixelShader = PS_Recon4; RenderTarget = TexOut4; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass Reconstruct3 { VertexShader = PostProcessVS; PixelShader = PS_Recon3; RenderTarget = TexOut3; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass Reconstruct2 { VertexShader = PostProcessVS; PixelShader = PS_Recon2; RenderTarget = TexOut2; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }
    pass Reconstruct1 { VertexShader = PostProcessVS; PixelShader = PS_Recon1; RenderTarget = TexOut1; VertexCount = 3; PrimitiveTopology = TRIANGLELIST; }

    pass Final
    {
        VertexShader      = PostProcessVS;
        PixelShader       = PS_Final;
        VertexCount       = 3;
        PrimitiveTopology = TRIANGLELIST;
    }
}
