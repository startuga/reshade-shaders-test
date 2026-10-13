// =================================================================================================
// Photoreal HDR Scene Grader (V9.2.2 - Reference Laboratory Edition)
// =================================================================================================
//
// Design Philosophy: PRECISION AND QUALITY OVER PERFORMANCE
// - Scene-referred grading in linear light, chroma decisions in the projection-based Iapbp space of
//   Huang et al. (Optics Express 32(17), 2024), fed absolute luminance / 10,000 cd/m2.
// - Numerically stable SA-PQ and PQ (ST 2084): the ratio raised to n ~ 78 is evaluated as
//   exp(n * log1p(-d)) with cancellation-free series, 3-100x more accurate in float32 than the
//   direct form; the Iapbp neutral-axis noise floor drops from ~1e-3 to <= 2.5e-5.
// - Exact achromatic axis: the paper's M1 / M2 re-fitted to satisfy its own Eq. 18 constraints.
// - Tone-dependent chroma correction (Mantiuk et al. 2009) driven by the local log-log slope of the
//   whole tone chain.
// - Bounded contrast (slope g at the pivot, returns to 1 away from it; shift <= (g - 1) W stops).
// - Display rendering: path to white keyed to the display peak, ACES 2.0-style gamut mapping along
//   lines toward a focus lightness between the per-hue cusp and mid grey, optional reach-gamut
//   compression, per-pixel boundary searches to float32 precision (no LUT approximation of the
//   boundary), run only where a color can actually change.
// - Shared configuration with BilateralContrast v9.4 through global preprocessor definitions.
// - Bit-exact passthrough at factory defaults; untouched pixels never round-trip.
// - Work that depends only on settings runs once, not per pixel or per frame (see V9.0.2 below);
//   ReShade performance mode turns every setting into a compile-time constant on top of that.
//
// Recommended chain (top to bottom in ReShade):
//   Photoreal HDR Deband (optional) -> BilateralContrast v9.4 -> Photoreal HDR Scene Grade ->
//   (optional display mapping / analysis shaders)
//   The grader is self-contained: it does not model any display mapper placed after it. Path to
//   white and the cusp table are keyed to PHDR_DISPLAY_PEAK_NITS.
//
// Global preprocessor definitions (ReShade: "Edit global preprocessor definitions"), shared with
// BilateralContrast v9.4:
//   PHDR_REF_WHITE_NITS        reference (paper / SDR) white in HDR, default 203 (ITU-R BT.2408)
//   PHDR_HLG_PEAK_NITS         HLG nominal display peak Lw, default 1000 (BT.2100 system gamma)
//   PHDR_DISPLAY_PEAK_NITS     display peak used for path to white and the cusp table, default 1000
//   PHDR_DISPLAY_PRIMARIES     0 Rec.709, 1 Display P3 (default), 2 Rec.2020, 3 custom (below)
//   PHDR_DISPLAY_RX ... _BY    custom display primaries (CIE 1931 xy), D65 white
//   PHDR_SDR_DECODE            0 piecewise sRGB (default, as Windows), 1 gamma 2.2 per channel,
//                              2 gamma 2.2 luminance only, 3 gamma 2.4 per channel (BT.1886-like)
//   PHDR_BLUE_NOISE            1 = blue-noise dither from Textures/PhotorealHDR_BlueNoise.png
//   PHDR_BOUNDARY_SOLVER       gamut-boundary search: 1 = Brent's method (default), 0 = bisection
//
// References:
// - Y. Huang et al., "Towards perceptual uniformity and HDR-WCG image processing: a projection-based
//   color space," Optics Express 32(17), 30742-30755 (2024). https://doi.org/10.1364/OE.530213
// - R. Mantiuk, R. Mantiuk, A. Tomaszewska, W. Heidrich, "Color correction for tone mapping,"
//   Computer Graphics Forum 28(2), 193-202 (2009).
// - ACES 2.0 Output Transforms (Academy Color Encoding System): chroma compression, gamut
//   compression with cusp / focus lightness, Reference Gamut Compression power curve.
// - C. Li et al., "Comprehensive color appearance model ...: CAM16," Color Res. Appl. 42(6) (2017);
//   C. Li et al., "Symmetric and transitive CATs," Color Res. Appl. 43(5), 633-642 (2018).
// - ITU-R BT.2100-2 (PQ, HLG, ICtCp), BT.2408 (203 cd/m2 reference white,
//   26 cd/m2 18% grey), BT.2124 (Delta E ITP).
// - SMPTE ST 2084:2014, IEC 61966-2-1:1999 (sRGB).
// - ReShade FX reference: https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md
//
// V9.2.2 changes (since V9.2.1):
// - Fixed (present since V9.0.0, SDR): pale colors pushed just above SDR white turned pure white
//   in a ~0.1% brightness band (a thin white contour in gradients crossing that brightness; 17 of
//   162,816 pixels in a test with exposure +0.6). Where no displayable chroma exists at a pixel's
//   lightness the boundary search correctly returns 0, and the purity form p * S * max_t then
//   collapsed the color. Purity is undefined there: saturation now applies as the plain radial
//   scale C1 * S, as everywhere else. HDR output is unchanged (0 of 162,816 pixels).
// - New: Brent's method (Brent 1973) for the gamut-boundary searches, default. It interpolates the
//   signed gamut margin (smallest target channel / luminance; ceiling headroom) instead of
//   halving, stops at the same bracket width and returns the inside end, like the bisection.
//   Measured by running the shader's code on the CPU on 5 photos (162,816 pixels, scRGB), with GPU
//   lockstep groups modeled as 8x4 / 8x8 tiles: 2.0-2.1x fewer boundary tests per pixel in
//   32-wide groups for vibrance presets, 1.7-2.0x for others; counting Brent's extra arithmetic
//   per step (upper bound), the searches are 1.2-1.46x faster (same in HDR10). In SDR the gain is
//   smaller, 1.1-1.2x, because the outward march under the SDR white ceiling is kept. Distance from
//   a float64 reference: median 3.4e-4 Delta E ITP (bisection 3.5e-4); Brent vs bisection at most
//   0.12 Delta E ITP. The cusp table matches the float64 reference as before.
//   PHDR_BOUNDARY_SOLVER = 0 restores the bisection (bit-identical to V9.2.1 in HDR; in SDR only
//   the fix above differs).
//
// V9.2.1 changes (since V9.2.0):
// - Fixed (present since V9.0.0): wrong per-hue cusp table. Every cusp landed at the display peak
//   with too much chroma (e.g. hue 0: I 0.314 / C 0.572 instead of 0.167 / 0.421), in scRGB and,
//   at the default 1000-nit peak, in HDR10 / HLG. The table's top sample is the neutral at the
//   peak; the container and target matrices differ in their D65 white by 1.3e-4, so that neutral
//   tested as above the brightness ceiling and the ceiling was dropped for it. The ceiling is now
//   dropped only for neutrals more than 0.1% above it. The table now matches the float64 reference.
//   Affects Lightness Focus (where compressed colors are aimed) and Path to White (cusp chroma).
//
// V9.2.0 changes (since V9.1.0) - performance regression vs V8 fixed:
// - Cause: with typical settings V9.0-9.1 ran 2.3x as many exact gamut-boundary tests per pixel as
//   V8 (121 vs 52). Gamut mapping ran a full search, preceded by a 48-step outward march, for every
//   pixel whose chroma grew (any saturation / vibrance boost, and tone chroma correction at
//   contrast > 1), and saturation alone ran a boundary search whose result cancels out.
// - Gamut mapping: one test decides whether the compression applies at all. In HDR every mapping
//   line leaves the target exactly once (measured: 0 re-entries in 12,960 lines, P3 / Rec.709 /
//   Rec.2020), so "the point at chroma C3 / t is inside" is exactly "q <= t" (identity). The full
//   search runs only past the knee, bracketed by that test and by the color itself, and needs no
//   outward march. With the SDR white ceiling a line can re-enter, so SDR keeps the march; it now
//   starts at chroma 2^-14 (same step size), not 1e-9.
// - Saturation alone: no boundary search (purity form p0 S max_t equals C1 S).
// - Constant-lightness searches start from the pixel itself when it is inside the volume.
// - The color engine (including the Iapbp transform) is skipped when no chroma operation is
//   active and the color is inside the target: the result was already the unchanged color.
// - Bisection tolerance 1.3e-6 -> 1.04e-5 relative chroma: the float32 inside test places the
//   boundary only to 2.6e-5 of the float64 truth (median), so the last steps refined rounding.
// - expm1 in the stable SA-PQ / PQ inverses: 7-term polynomial, same accuracy (1.41 ulp).
// - Measured by running the shader's own generated code on the CPU, 3,000 colors per preset:
//   boundary tests per pixel 2.5-10.8x fewer than V9.1 in HDR (typical 121 -> 47), 1.6x in SDR;
//   largest output difference 0.01 Delta E ITP (1 = just noticeable). Versus V8.0.0 with the same
//   settings (SDR-origin content): typical 52 -> 48, saturation only 26 -> 8.
//
// V9.1.0 changes (since V9.0.2):
// - Removed: BT.2390 EETF prediction (PHDR_EETF_MODE and the five PHDR_EETF_* definitions). The
//   grader no longer models a display mapper placed after it. With the former default
//   (PHDR_EETF_MODE 0) the output is bit-identical to V9.0.2; the Reference Levels debug view no
//   longer marks an EETF knee.
//
// V9.0.2 changes (since V9.0.1) - performance, output bit-identical:
// - Cusp table cached: rebuilt only when its inputs change (active color space, gamut target,
//   display peak / primaries), tracked by a key texture. Before, every frame recomputed 1024 hues
//   x 107 exact boundary searches (~3 million Iapbp inverse transforms). In ReShade performance
//   mode the inputs are compile-time constants, so the table is built once per compile.
// - Passes that are not needed are skipped in the vertex shader (no pixel shader invocations):
//   the scene grade when the settings are bit-transparent and no debug / split view is shown
//   (the back buffer is left untouched, identical to the pass-through), and the cusp table when
//   nothing reads it (Bypass target, or neither path to white nor gamut focus in use).
// - White balance gains and the four Iapbp reference lightness levels: computed per vertex in
//   normal mode (25 exp2 and 12 log2 fewer per pixel), folded to literals in performance mode.
// - Performance mode: debug / QA views, scope, split view and the mouse uniform are compiled
//   out (a preset saved with a debug view grades normally).
//
// V9.0.1 changes (since V9.0.0):
// - Fixed: ReShade FX compile errors X3090 "nested struct members are not supported". The five
//   gamut volumes of the grading context are stored as flat members and rebuilt on use
//   (TargetVolume, ReachVolume, VibranceVolume, CuspVolume, PhysVolume). Same values, no
//   behavior change.
// - Verified with ReShade's own effect compiler (reshadefx) over 363 preprocessor configurations,
//   and the generated code with DXC (HLSL), glslang (OpenGL GLSL) and spirv-val (Vulkan SPIR-V).
//
// V9.0.0 changes (since V8.0.0):
// - New: tone-dependent chroma correction. Every tone operation changes luminance at fixed
//   chromaticity, which over-saturates compressed highlights and under-saturates expanded midtones.
//   The local log-log slope c of the full tone chain (central difference, 1/32 stop) drives the
//   Mantiuk 2009 luminance-preserving model s(c) = (1 + k1) c^k2 / (1 + k1 c^k2), k1 = 2.3892,
//   k2 = 0.8552 (s(0.3) = 0.65, s(1.5) = 1.09), applied as a radial Iapbp chroma factor. With EETF
//   prediction the EETF slope is included, so the grader pre-compensates the EETF's compression.
// - New: path to white keyed to the display peak (Iapbp lightness between "start" and the peak),
//   reduced for colors already near the cusp (ACES 2.0 chroma-compression behavior).
// - New: bounded contrast y = x + (g - 1) W tanh(x / W) with separate shadow / highlight roll-off W;
//   legacy unbounded power available. The old 80x soft ratio cap is no longer needed.
//   Contrast pivot default 0.1768 -> 0.128 of reference white (BT.2408 18% grey, 26 of 203 nits).
// - New: gamut mapping along a line toward a focus lightness (lerp of pixel lightness and
//   0.5 (cusp + mid grey), slider "Lightness Focus"), per-hue cusp table (1024 hues, exact
//   64-point scan + 40 golden-section steps per hue) rebuilt every frame, exact per-pixel boundary
//   search along the line. Optional reach-gamut compression (ACES 2.0 style).
// - New (removed in V9.1.0): Lilium BT.2390 EETF prediction (YRGB / maxCLL, all Lilium parameters mirrored): colors are
//   mapped in the predicted display space and the result is passed through the exact EETF inverse.
//   Measured (float64 reference, 300 boosted Rec.2020 HDR colors, P3 x 1000 nits): 0 outside the
//   display volume after the EETF, versus 30 without prediction.
// - New: display primaries (default P3-D65) as the HDR "Auto" gamut target; custom primaries.
// - New: SDR decode options (gamma 2.2 per channel / luminance, gamma 2.4) for SDR-origin games
//   decoded by Windows with piecewise sRGB (raised blacks); blended to identity above SDR white.
// - New: stable SA-PQ / PQ transfer functions (see above); chroma noise gate lowered 5x
//   (5e-4..2e-3 -> 1e-4..4e-4).
// - New: two-step CAT16 white balance with degree of adaptation D (Li et al. 2018); D = 1 equals V8.
// - New: shared configuration (global preprocessor definitions) for reference white, HLG peak,
//   display peak / primaries and SDR decode; the Reference White and HLG Peak sliders are removed.
// - New: PQ-step-sized dither for scRGB output (10-bit PQ, the downstream link), optional blue noise.
// - New: quantization-aware deband technique (place first in the chain).
// - New QA views: gamut compression amount, chroma correction, BT.2408 reference levels,
//   Delta E ITP vs input (BT.2124), hue-slice scope under the cursor (boundary, cusp, focus,
//   input -> output), split-screen comparison.
// =================================================================================================

#include "ReShade.fxh"

// =================================================================================================
// 0. Shared Configuration (global preprocessor definitions)
// =================================================================================================

#if defined(__RESHADE__) && __RESHADE__ < 40800
#error "Photoreal HDR requires ReShade 4.8.0 or newer."
#endif

#ifndef BUFFER_COLOR_SPACE
#define BUFFER_COLOR_SPACE 1
#endif

#ifndef BUFFER_COLOR_BIT_DEPTH
#define BUFFER_COLOR_BIT_DEPTH 8
#endif

#ifndef PHDR_REF_WHITE_NITS
#define PHDR_REF_WHITE_NITS 203.0
#endif

#ifndef PHDR_HLG_PEAK_NITS
#define PHDR_HLG_PEAK_NITS 1000.0
#endif

#ifndef PHDR_DISPLAY_PEAK_NITS
#define PHDR_DISPLAY_PEAK_NITS 1000.0
#endif

#ifndef PHDR_DISPLAY_PRIMARIES
#define PHDR_DISPLAY_PRIMARIES 1
#endif

#ifndef PHDR_DISPLAY_RX
#define PHDR_DISPLAY_RX 0.680
#endif
#ifndef PHDR_DISPLAY_RY
#define PHDR_DISPLAY_RY 0.320
#endif
#ifndef PHDR_DISPLAY_GX
#define PHDR_DISPLAY_GX 0.265
#endif
#ifndef PHDR_DISPLAY_GY
#define PHDR_DISPLAY_GY 0.690
#endif
#ifndef PHDR_DISPLAY_BX
#define PHDR_DISPLAY_BX 0.150
#endif
#ifndef PHDR_DISPLAY_BY
#define PHDR_DISPLAY_BY 0.060
#endif

#ifndef PHDR_SDR_DECODE
#define PHDR_SDR_DECODE 0
#endif

#ifndef PHDR_BLUE_NOISE
#define PHDR_BLUE_NOISE 0
#endif

// Gamut-boundary search: 1 = Brent's method (default), 0 = bisection (V9.2.1)
#ifndef PHDR_BOUNDARY_SOLVER
#define PHDR_BOUNDARY_SOLVER 1
#endif

// ReShade performance mode: every UI uniform becomes a compile-time constant taken from the preset
#if defined(__RESHADE_PERFORMANCE_MODE__) && __RESHADE_PERFORMANCE_MODE__
#define PHDR_PERF_MODE 1
#else
#define PHDR_PERF_MODE 0
#endif

// =================================================================================================
// 1. Constants & Definitions
// =================================================================================================

static const float FLT_MIN              = 1.175494351e-38;
static const float PROJ_EPS             = 1e-7;
static const float SCRGB_WHITE_NITS     = 80.0;
static const float NEUTRAL_EPS          = 1e-6;
static const float PI                   = 3.14159265358979323846;
static const float LN2                  = 0.69314718055994531;
static const float INV_LN2              = 1.44269504088896341;

// Iapbp absolute luminance normalization (PQ convention: 1.0 = 10,000 cd/m2)
static const float IAPBP_ABS_NORM       = 10000.0;

// float32 noise floor of the Iapbp forward transform with the stable SA-PQ (measured <= 2.5e-5 with a
// GPU transcendental error model up to 10,000 cd/m2). Chroma below LO is not boosted.
static const float CHROMA_NOISE_LO      = 1e-4;
static const float CHROMA_NOISE_HI      = 4e-4;

// Gamut boundary search: log2-domain bisection from this chroma; outward march where needed
static const float BOUNDARY_T_MIN       = 1e-9;
static const int   BOUNDARY_MARCH_STEPS = 48;
static const int   BOUNDARY_BISECT_STEPS = 24;
static const int   BRENT_MAX_STEPS       = 48;   // safety cap; measured worst case far below
// Bisection stops at this log2(chroma) bracket width: relative chroma precision 1.04e-5 (21
// halvings of [1e-9, 3]). The float32 inside test itself places the boundary only to 2.6e-5
// (median; 95th percentile 3e-4) of the float64 truth, so finer steps refine rounding noise.
// A tighter starting bracket reaches the width in fewer steps.
static const float BOUNDARY_LOG2_TOL    = 1.502e-5;
// Outward march (first-exit bracketing): the step of the original 48-step march over [1e-9, 3]
// (0.656 stops), starting at chroma 2^-14 instead of 1e-9. Points closer to the neutral axis are
// not marched; a first exit below 2^-14 is still bracketed (the march start itself is tested).
static const float BOUNDARY_MARCH_STEP_LOG2  = 0.65588157;
static const float BOUNDARY_MARCH_START_LOG2 = -14.0;

// Per-hue cusp table
static const int   CUSP_LUT_SIZE        = 1024;
static const int   CUSP_SCAN_STEPS      = 64;
static const int   CUSP_GOLDEN_STEPS    = 40;

// White balance reference (D65: 6504 K, Duv +0.0032 on the Krystek Planckian locus)
static const float WB_REF_KELVIN        = 6504.0;
static const float WB_REF_DUV           = 0.0032;

// ACES Reference Gamut Compression power (ACES 1.3 default)
static const float RGC_POWER            = 1.2;

// Mantiuk et al. 2009, luminance-preserving color correction fit
static const float MANTIUK_K1           = 2.3892;
static const float MANTIUK_K2           = 0.8552;

// Tone-slope estimate: central difference of +-1/32 stop
static const float TONE_SLOPE_H         = 0.03125;

// ITU-R BT.2408 18% grey: 26 cd/m2 at 203 cd/m2 reference white
static const float BT2408_GREY_FRACTION = 0.128;

// -------------------------------------------------------------------------------------------------
// sRGB Constants (IEC 61966-2-1:1999) - Analytically Solved C0-Intersection Roots
// -------------------------------------------------------------------------------------------------
static const float SRGB_THRESHOLD_EOTF  = 0.040448236277123205;
static const float SRGB_THRESHOLD_OETF  = 0.003130668442501796;
static const float SRGB_GAMMA           = 2.4;
static const float SRGB_INV_GAMMA       = 0.41666666666666667;

// -------------------------------------------------------------------------------------------------
// ST.2084 (PQ) Constants (SMPTE ST 2084:2014). PQ_A = 1 - c1 = c2 - c3 exactly.
// -------------------------------------------------------------------------------------------------
static const float PQ_M1                = 0.1593017578125;
static const float PQ_M2                = 78.84375;
static const float PQ_C1                = 0.8359375;
static const float PQ_C2                = 18.8515625;
static const float PQ_C3                = 18.6875;
static const float PQ_A                 = 0.1640625;
static const float PQ_INV_M1            = 6.2773946360153257;
static const float PQ_LN_C1             = -0.17920142945771100;   // ln(c1): PQ code of 0 cd/m2
static const float PQ_PEAK_LUMINANCE    = 10000.0;

// -------------------------------------------------------------------------------------------------
// Color Space CIE Tristimulus Matrices (D65 White Point Reference)
// -------------------------------------------------------------------------------------------------
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

// ITU-R BT.2100 ICtCp: Rec.2020 RGB -> LMS (x 4096)
static const float3x3 RGB2020_to_LMS_ICtCp = float3x3(
    1688.0 / 4096.0, 2146.0 / 4096.0,  262.0 / 4096.0,
     683.0 / 4096.0, 2951.0 / 4096.0,  462.0 / 4096.0,
      99.0 / 4096.0,  309.0 / 4096.0, 3688.0 / 4096.0
);

// -------------------------------------------------------------------------------------------------
// CIE CIECAM16:2022 Chromatic Adaptation Matrices (each row sums to 1: equal-energy E -> 1, 1, 1)
// -------------------------------------------------------------------------------------------------
static const float3x3 XYZ_to_CAT16 = float3x3(
     0.401288,  0.650173, -0.051461,
    -0.250268,  1.204414,  0.045854,
    -0.002079,  0.048952,  0.953127
);
static const float3x3 CAT16_to_XYZ = float3x3(
     1.86206786, -1.01125463,  0.14918677,
     0.38752654,  0.62144744, -0.00897399,
    -0.01584150, -0.03412294,  1.04996444
);

// ACES AP0 primaries (encloses the whole spectral locus), D65-normalized: the physical-realizability
// limit used by Bypass / Unclamped mode (together with positive luminance: AP0's blue primary
// lies below y = 0). Only the sign of the components is tested.
static const float3x3 XYZ_to_AP0 = float3x3(
     1.052238598,  0.000000000, -0.000097710,
    -0.491495220,  1.361106438,  0.097366836,
     0.000000000,  0.000000000,  0.918224951
);

// Retained specifically for calibrated 3D Fitzpatrick Melanin-Hemoglobin skin locus gating
static const float3x3 XYZ_to_CAT02 = float3x3(
     0.7328,  0.4296, -0.1624,
    -0.7036,  1.6975,  0.0061,
     0.0030,  0.0136,  0.9834
);

// -------------------------------------------------------------------------------------------------
// Optics Express 2024: Iapbp Projection Transformation Matrices & Inverses (Constraint-Exact)
// -------------------------------------------------------------------------------------------------
// Published M1 / M2 (Huang et al. 2024, Eqs. 20-21), minimally corrected so the paper's white-point
// constraints (Eq. 18) hold exactly:
//   M1: rows L, M, S rescaled so each row sum = a41 + a42 + a43 + 1   (factors 0.99899 .. 1.00091)
//   M2: row I scaled so its sum = b41 + b42 + b43 + 1                  (factor 1.001367)
//       rows ap, bp shifted by -(row sum) / 3 so each sums to 0        (shifts 3.50e-4, 1.17e-4)
// Result: normalized white -> L = M = S = 1 and every neutral -> ap = bp = 0 exactly.
// This correction is this shader's own engineering fix (no published erratum exists).
// Inverses computed in float64 and normalized so element [3][3] = 1.
static const float4x4 M1_XYZ_to_LMS = float4x4(
     0.490483972,  1.043949507,  0.481995522,  0.000000000,
     0.518030448,  1.257690024,  0.240708528,  0.000000000,
     1.587261882,  0.553434286, -0.124267169,  0.000000000,
     1.967421000, -0.807475000, -0.143517000,  1.000000000
);
static const float4x4 M1_LMS_to_XYZ = float4x4(
     0.579070859, -0.793043792,  0.709899147,  0.000000000,
    -0.892974277,  1.652176677, -0.263276186,  0.000000000,
     3.419523642, -2.771423589, -0.152173839,  0.000000000,
    -1.369570799,  2.496595974, -1.631098960,  1.000000000
);
static const float4x4 M2_LMSprime_to_Iapbp = float4x4(
    -0.011839158,  0.249166069, -0.106174911,  0.000000000,
     5.055613667, -5.937384333,  0.881770667,  0.000000000,
     1.247305667, -3.372872333,  2.125566667,  0.000000000,
    -0.741126000,  1.127297000, -1.255019000,  1.000000000
);
static const float4x4 M2_Iapbp_to_LMSprime = float4x4(
     7.624740759,  0.135564036,  0.324628622,  0.000000000,
     7.624740759, -0.084788566,  0.416039756,  0.000000000,
     7.624740759, -0.214093872,  0.940143584,  0.000000000,
     6.624740759, -0.072639950,  0.951488404,  1.000000000
);

// SA-PQ Constants (Huang et al. 2024 Eq. 6). SA_A = 1 - c1, SA_B = c2 - c3 (exact decimals).
static const float SA_PQ_C1     = 0.7707;
static const float SA_PQ_C3     = 44.2269;
static const float SA_PQ_M      = 0.2926;
static const float SA_PQ_N      = 78.2171;
static const float SA_PQ_INV_M  = 3.417634996582365;
static const float SA_A         = 0.2293;
static const float SA_B         = 0.2292;
static const float SA_LN_C1     = -0.26045608619818940;   // ln(c1): SA-PQ code of black
static const float SA_E_MAX     = 0.0051723146;          // (c2 / c3)(1 - 1e-5) - 1: asymptote guard

// SA-PQ inverse validity: L'M'S' at or below c1^n decodes to black (clamped), at or above
// K_MAX^n hits the asymptote clamp. Samples there are outside any gamut.
static const float SA_PQ_BLACK  = 1.4206949e-9 * 1.001;
static const float SA_PQ_V_MAX  = 1.4970863;

// Chroma Reliability Thresholds
static const float CHROMA_RELIABILITY_START     = 5e-4;
static const float CHROMA_STABILITY_THRESH      = 2e-3;
static const float INV_CHROMA_RELIABILITY_SPAN  = 1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);

// Zone System Constants (1/2-stop intervals pivoted at Zone V = 0.1767767)
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
// 2. Textures & Samplers
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

// Per-hue cusp table of the display volume: x = Iapbp lightness, y = chroma. Built only when its
// inputs change (see CuspTableKey): once per compile in performance mode.
texture2D PHDR_CuspTex { Width = CUSP_LUT_SIZE; Height = 1; Format = RG32F; };
sampler2D PHDR_CuspSampler
{
    Texture   = PHDR_CuspTex;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
    AddressU  = WRAP;
    AddressV  = CLAMP;
};

// Key of the inputs the cusp table was built for (0 = no valid table). ReShade creates render
// targets cleared to zero, so a new or recompiled effect always rebuilds the table once.
texture2D PHDR_CuspKeyTex { Width = 1; Height = 1; Format = RGBA32F; };
sampler2D PHDR_CuspKeySampler
{
    Texture   = PHDR_CuspKeyTex;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

#if PHDR_BLUE_NOISE
texture2D PHDR_BlueNoiseTex < source = "PhotorealHDR_BlueNoise.png"; > { Width = 64; Height = 64; Format = RGBA8; };
sampler2D PHDR_BlueNoiseSampler
{
    Texture   = PHDR_BlueNoiseTex;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
    AddressU  = WRAP;
    AddressV  = WRAP;
};
#endif

#if !PHDR_PERF_MODE
uniform float2 MousePoint < source = "mousepoint"; >;
#endif

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

uniform float fWhiteBalanceK <
    ui_type     = "slider";
    ui_min      = 2000.0; ui_max = 12000.0; ui_step = 1.0;
    ui_label    = "White Balance (Kelvin)";
    ui_tooltip  = "Color temperature that is rendered as neutral (CAT16 von Kries adaptation).\n"
                  "6504 K = D65 = no change. Higher = warmer image, lower = cooler image\n"
                  "(same convention as raw converters). Neutral luminance is preserved.";
    ui_category = "1. Scene Grade";
> = 6504.0;

uniform float fWhiteBalanceDuv <
    ui_type     = "slider";
    ui_min      = -0.0200; ui_max = 0.0200; ui_step = 0.0001;
    ui_label    = "White Balance Tint (Duv)";
    ui_tooltip  = "Distance of the neutral white from the Planckian locus (CIE 1960 uv).\n"
                  "+0.0032 = D65 = no change. Higher = more magenta image, lower = greener image.";
    ui_category = "1. Scene Grade";
> = 0.0032;

uniform float fAdaptationDegree <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "White Balance Adaptation (D)";
    ui_tooltip  = "Degree of chromatic adaptation (CIECAM16 D). 1.0 = complete correction (colorist default).\n"
                  "Below 1 the white balance is only partially applied, like incomplete visual adaptation.\n"
                  "Uses the two-step CAT16 (via equal-energy white), which stays symmetric for D < 1\n"
                  "(Li et al. 2018).";
    ui_category = "1. Scene Grade";
> = 1.00;

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
    ui_label    = "Contrast";
    ui_tooltip  = "Log-log slope at the pivot (1.0 = no change).\n"
                  "Bounded mode: the slope returns to 1 away from the pivot, so no highlight or shadow\n"
                  "moves by more than (Contrast - 1) x Roll-off stops (it cannot push highlights\n"
                  "into the display mapper's knee without limit).";
    ui_category = "1. Scene Grade";
> = 1.00;

uniform int iContrastMode <
    ui_type     = "combo";
    ui_label    = "Contrast Curve";
    ui_items    = "Bounded (tanh roll-off, recommended)\0"
                  "Unbounded power (legacy)\0";
    ui_category = "1. Scene Grade";
> = 0;

uniform float fContrastPivot <
    ui_type     = "slider";
    ui_min      = 0.01; ui_max = 1.00; ui_step = 0.001;
    ui_label    = "Contrast Pivot (x Reference White)";
    ui_tooltip  = "Luminance that is unchanged by contrast, as a fraction of reference white.\n"
                  "0.128 = ITU-R BT.2408 18% grey (26 nits at 203 nits reference white).";
    ui_category = "1. Scene Grade";
> = 0.128;

uniform float fContrastRolloffLo <
    ui_type     = "slider";
    ui_min      = 0.50; ui_max = 12.00; ui_step = 0.01;
    ui_label    = "Contrast Roll-off Shadows (Stops)";
    ui_tooltip  = "Bounded mode: width W of the contrast region below the pivot.\n"
                  "Shadows move by at most (Contrast - 1) x W stops.";
    ui_category = "1. Scene Grade";
> = 2.50;

uniform float fContrastRolloffHi <
    ui_type     = "slider";
    ui_min      = 0.50; ui_max = 12.00; ui_step = 0.01;
    ui_label    = "Contrast Roll-off Highlights (Stops)";
    ui_tooltip  = "Bounded mode: width W of the contrast region above the pivot.\n"
                  "Highlights move by at most (Contrast - 1) x W stops.";
    ui_category = "1. Scene Grade";
> = 2.50;

uniform float fShadows <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Shadows (Log Recovery)";
    ui_tooltip  = "Lifts (+) or deepens (-) shadows in the stop domain, by up to 3 stops in the deepest shadows.\n"
                  "Blends into Highlights with a C1-continuous (smoothstep) transition around the pivot.";
    ui_category = "1. Scene Grade";
> = 0.0;

uniform float fHighlights <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Highlights (Log Recovery)";
    ui_tooltip  = "Darkens (-) or brightens (+) highlights in the stop domain, by up to 3 stops at the top.\n"
                  "This is a tonal shift that grows away from the pivot, not a compression of highlight range.";
    ui_category = "1. Scene Grade";
> = 0.0;

uniform float fHighlightRecovery <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Highlight Recovery (Shoulder)";
    ui_tooltip  = "Compresses highlights above the Recovery Pivot in the stop domain (a soft shoulder).\n"
                  "Above the blend, the tone slope becomes 1 - 0.9 x Amount (Amount 1.0 = 10:1 compression).\n"
                  "Monotone and smooth. 0 = off (bit-exact).";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fRecoveryPivot <
    ui_type     = "slider";
    ui_min      = -6.00; ui_max = 6.00; ui_step = 0.01;
    ui_label    = "Recovery Pivot (Stops)";
    ui_tooltip  = "Center of the shoulder, in stops relative to the Recovery Pivot Reference.\n"
                  "Half of the compression is reached here.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform int iRecoveryReference <
    ui_type     = "combo";
    ui_label    = "Recovery Pivot Reference";
    ui_items    = "Reference White\0"
                  "Display Peak\0";
    ui_tooltip  = "Display Peak: the pivot is relative to PHDR_DISPLAY_PEAK_NITS (SDR: white), so the\n"
                  "shoulder stays below the display mapper's knee on any display.";
    ui_category = "1. Scene Grade";
> = 0;

uniform float fRecoveryBlend <
    ui_type     = "slider";
    ui_min      = 0.10; ui_max = 6.00; ui_step = 0.01;
    ui_label    = "Recovery Blend Width (Stops)";
    ui_tooltip  = "Width of the smoothstep transition into the shoulder. Small = firm knee,\n"
                  "large = gradual roll-off starting well below the pivot.";
    ui_category = "1. Scene Grade";
> = 1.00;

uniform float fToneChromaCorrection <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Tone Chroma Correction";
    ui_tooltip  = "Corrects colorfulness for tone changes (Mantiuk et al. 2009): luminance compression\n"
                  "(highlight recovery, contrast < 1) desaturates, expansion (contrast > 1)\n"
                  "saturates, as needed to keep the color appearance. Driven by the local log-log slope\n"
                  "of the whole tone chain. 1.0 = model strength, 0 = chromaticity strictly preserved.\n"
                  "Only acts where tone operations change contrast.";
    ui_category = "2. Color";
> = 1.00;

uniform float fSaturation <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 2.00; ui_step = 0.01;
    ui_label    = "Purity / Saturation (Iapbp)";
    ui_tooltip  = "Strictly radial chromaticity scaling in the projection-based Iapbp color space.\n"
                  "Set to 1.0 for neutral identity pass.\n"
                  "Set to 0.0 for exact bit-level R=G=B monochrome collapse.";
    ui_category = "2. Color";
> = 1.00;

uniform float fVibrance <
    ui_type     = "slider";
    ui_min      = -1.00; ui_max = 2.00; ui_step = 0.01;
    ui_label    = "Smart Saturation (Vibrance)";
    ui_tooltip  = "Boosts muted colors and leaves already-saturated ones alone.\n"
                  "Demand is measured against the Vibrance Reference Gamut (the content's own gamut),\n"
                  "and the expansion is bounded by the gamut target boundary: it never crosses it.\n"
                  "Negative values mute pastels first.";
    ui_category = "2. Color";
> = 0.00;

uniform int iVibranceReference <
    ui_type     = "combo";
    ui_label    = "Vibrance Reference Gamut";
    ui_items    = "Rec. 709 (SDR-origin content)\0Display P3 (D65)\0Rec. 2020 (native WCG content)\0";
    ui_tooltip  = "Gamut the content was made in. A color at this gamut's edge counts as fully saturated\n"
                  "and gets no vibrance boost. Use Rec. 709 for SDR games (also in HDR / scRGB output).";
    ui_category = "2. Color";
> = 0;

uniform float fSkinProtection <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Skin Tone Protection";
    ui_tooltip  = "Reduces saturation boosts on skin tones (Fitzpatrick I-VI) using a heuristic\n"
                  "melanin-hemoglobin cone-ratio locus (not fitted data).";
    ui_category = "2. Color";
> = 0.85;

uniform float fAbneyCorrection <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Abney Hue Compensation";
    ui_tooltip  = "Counteracts perceived hue shifts caused by CHANGING saturation (Abney effect).\n"
                  "The rotation follows the purity change, so it has no effect at Saturation 1.00 / Vibrance 0.\n"
                  "The hue profile is a heuristic model, not fitted data. Keep it subtle.";
    ui_category = "2. Color";
> = 0.00;

uniform float fPathToWhite <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Path to White";
    ui_tooltip  = "Desaturates colors as their lightness approaches the display peak (SDR: white), as\n"
                  "film and ACES 2.0 renderings do; colors already near the gamut cusp are desaturated\n"
                  "less. Keyed to the display peak (PHDR_DISPLAY_PEAK_NITS).\n"
                  "0 = off.";
    ui_category = "3. Display Rendering & Gamut";
> = 0.00;

uniform float fPathToWhiteStart <
    ui_type     = "slider";
    ui_min      = 0.25; ui_max = 6.00; ui_step = 0.01;
    ui_label    = "Path to White Start (Stops Below Peak)";
    ui_tooltip  = "Lightness (neutral equivalent, in stops below the display peak) where the path to\n"
                  "white begins. It reaches full strength at the peak.";
    ui_category = "3. Display Rendering & Gamut";
> = 1.50;

uniform int iGamutTarget <
    ui_type     = "combo";
    ui_label    = "Gamut Target";
    ui_items    = "Auto (Display Primaries in HDR, Rec. 709 in SDR)\0"
                  "Rec. 709 (SDR Standard)\0"
                  "Display P3 (D65)\0"
                  "Rec. 2020 (UHD Display)\0"
                  "Bypass / Unclamped (Physical Limit, AP0)\0";
    ui_tooltip  = "Gamut that colors are mapped into.\n"
                  "- Auto: PHDR_DISPLAY_PRIMARIES in HDR (default Display P3: most HDR panels), Rec.709 in SDR.\n"
                  "- Bypass: no display-gamut limit; colors only kept physically realizable (inside AP0).\n"
                  "In SDR the container channels are also kept at or below white without per-channel\n"
                  "clipping.";
    ui_category = "3. Display Rendering & Gamut";
> = 0;

uniform float fGamutGuardKnee <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label    = "Gamut Compression Knee";
    ui_tooltip  = "Width of the soft compression zone below the gamut boundary (fraction of the boundary).\n"
                  "ACES Reference Gamut Compression power curve: colors below 1 - Knee are untouched,\n"
                  "colors pushed beyond are rolled off smoothly instead of clipped. 0 = hard clip.";
    ui_category = "3. Display Rendering & Gamut";
> = 0.20;

uniform float fGamutFocus <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Gamut Mapping Lightness Focus";
    ui_tooltip  = "Direction of gamut compression in each hue plane (ACES 2.0 style).\n"
                  "0 = constant lightness (bright saturated colors bleach toward white).\n"
                  "1 = toward a focus lightness halfway between the hue's cusp and mid grey (bright\n"
                  "saturated colors keep more color at slightly lower lightness).\n"
                  "Only affects colors that are compressed.";
    ui_category = "3. Display Rendering & Gamut";
> = 0.50;

uniform int iOutOfGamutMode <
    ui_type     = "combo";
    ui_label    = "Compression Reach";
    ui_items    = "Target Only (identity inside the target)\0"
                  "Reach Gamut (ACES 2.0: Rec.2020 compressed into the target)\0";
    ui_tooltip  = "Target Only: colors inside the target are never changed by the gamut stage; colors\n"
                  "outside are placed on the boundary (or rolled off when they were expanded).\n"
                  "Reach Gamut: the whole reach gamut (Rec.2020 in HDR) is compressed smoothly into the\n"
                  "target, so out-of-target content keeps gradation. Colors in the knee zone near the\n"
                  "target edge are then slightly desaturated (Rec.709 blue in a P3 target: about -2.6%).";
    ui_category = "3. Display Rendering & Gamut";
> = 0;

uniform int iColorSpaceOverride <
    ui_type     = "combo";
    ui_label    = "Color Space Override";
    ui_items    = "Auto (Default via ReShade)\0"
                  "sRGB (SDR)\0"
                  "scRGB (HDR Linear)\0"
                  "HDR10 (PQ)\0"
                  "HLG (HDR)\0";
    ui_tooltip  = "Container format detection override.\n"
                  "Reference white, HLG peak, display peak / primaries and SDR decode are global\n"
                  "preprocessor definitions (PHDR_*), shared with BilateralContrast v9.4.";
    ui_category = "4. System";
> = 0;

uniform bool bDither <
    ui_label    = "Dither Graded Output (8/10-bit)";
    ui_tooltip  = "Adds triangular-PDF dither (+-1 code value) to graded pixels on 8/10-bit outputs\n"
                  "(sRGB, HDR10, HLG). Bypassed pixels stay bit-exact.";
    ui_category = "4. System";
> = true;

uniform bool bFloatDither <
    ui_label    = "Dither Graded Output (scRGB, 10-bit PQ link)";
    ui_tooltip  = "scRGB is converted to 10-bit PQ downstream (Windows / driver). Adds triangular-PDF\n"
                  "dither of +-1 10-bit PQ code value (Rec.2020) to graded pixels so smooth gradients\n"
                  "do not band in that conversion. Bypassed pixels stay bit-exact.";
    ui_category = "4. System";
> = true;

#if !PHDR_PERF_MODE
uniform int iDebugMode <
    ui_type     = "combo";
    ui_label    = "Debug Visualization";
    ui_items    = "Off\0"
                  "Luminance (False Color Stops)\0"
                  "Zone Map (Ansel Adams)\0"
                  "CAT16 Cone Response\0"
                  "Iapbp Chroma / Saturation\0"
                  "Iapbp Hue Wheel\0"
                  "Negative / WCG Out-of-Gamut\0"
                  "Skin Protection Mask\0"
                  "Gamut Compression Amount\0"
                  "Chroma Correction (Tone + Path to White)\0"
                  "Reference Levels (BT.2408 / Display)\0"
                  "Delta E ITP vs Input (BT.2124)\0"
                  "Hue Slice Scope (Cursor)\0";
    ui_tooltip  = "Gamut Compression Amount: black = untouched, blue..red = 0..50% chroma removed,\n"
                  "  magenta = color arrived outside the target.\n"
                  "Chroma Correction: blue = desaturated, orange = saturated by tone correction / path to white.\n"
                  "Reference Levels: blue < 1 nit, green = 18% grey (26 nits at 203) +-1/4 stop,\n"
                  "  white = reference white +-1/4 stop, yellow = above white, red = above the display peak.\n"
                  "Delta E ITP: 1 = one just-noticeable difference (black 0, blue 1, green 3, yellow 6, red >= 10).\n"
                  "Hue Slice Scope: Iapbp lightness (up) vs chroma (right) for the hue under the mouse:\n"
                  "  grey = target volume, dark grey = reach gamut, yellow = cusp, cyan = focus,\n"
                  "  white = input, green = output; lines: 18% grey, reference white, display peak.";
    ui_category = "5. Debug / QA";
> = 0;

uniform bool bSplitView <
    ui_label    = "Split View (left = ungraded)";
    ui_category = "5. Debug / QA";
> = false;

uniform float fSplitPosition <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.001;
    ui_label    = "Split Position";
    ui_category = "5. Debug / QA";
> = 0.50;
#else
// Performance mode: debug / QA views and the split view are compiled out. A preset saved with a
// debug view active still grades normally.
static const int   iDebugMode     = 0;
static const bool  bSplitView     = false;
static const float fSplitPosition = 0.5;
#endif

uniform float fDebandRadius <
    ui_type     = "slider";
    ui_min      = 2.0; ui_max = 48.0; ui_step = 0.5;
    ui_label    = "Deband Radius (px)";
    ui_tooltip  = "Base sampling radius of the Photoreal HDR Deband technique (4 iterations up to 4x).\n"
                  "Should exceed the width of the visible bands.";
    ui_category = "6. Deband (separate technique, place first)";
> = 16.0;

uniform float fDebandThreshold <
    ui_type     = "slider";
    ui_min      = 0.25; ui_max = 4.00; ui_step = 0.01;
    ui_label    = "Deband Threshold (Source Code Values)";
    ui_tooltip  = "Maximum local difference, in source code values, that is treated as banding.\n"
                  "1 code value = one quantization step of the source. Larger = smooths more,\n"
                  "including faint real texture.";
    ui_category = "6. Deband (separate technique, place first)";
> = 1.50;

uniform int iDebandSourceBits <
    ui_type     = "combo";
    ui_label    = "Deband Source Precision";
    ui_items    = "8-bit (typical game)\0"
                  "10-bit\0"
                  "6-bit (BC1/DXT1 green, RGB565)\0"
                  "5-bit (BC1/DXT1 red/blue)\0";
    ui_category = "6. Deband (separate technique, place first)";
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

float Max3(float3 v) { return max(max(v.r, v.g), v.b); }
float Min3(float3 v) { return min(min(v.r, v.g), v.b); }

bool3 IsNan3(float3 v) { return (asuint(v) & 0x7FFFFFFFu) > 0x7F800000u; }
bool3 IsInf3(float3 v) { return (asuint(v) & 0x7FFFFFFFu) == 0x7F800000u; }

// Per-channel sanitation: NaN -> 0, +/-Inf -> +/-10000 cd/m2 (PQ peak); finite values untouched
float3 SanitizeLinear(float3 v)
{
    bool3 nan_c = IsNan3(v);
    bool3 inf_c = IsInf3(v);
    float3 r;
    r.r = nan_c.r ? 0.0 : (inf_c.r ? ((v.r > 0.0) ? PQ_PEAK_LUMINANCE : -PQ_PEAK_LUMINANCE) : v.r);
    r.g = nan_c.g ? 0.0 : (inf_c.g ? ((v.g > 0.0) ? PQ_PEAK_LUMINANCE : -PQ_PEAK_LUMINANCE) : v.g);
    r.b = nan_c.b ? 0.0 : (inf_c.b ? ((v.b > 0.0) ? PQ_PEAK_LUMINANCE : -PQ_PEAK_LUMINANCE) : v.b);
    return r;
}

// "Hash without sine" (D. Hoskins), uniform in [0, 1)
float Hash12(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

// Triangular-PDF noise in (-1, 1): two uniform samples (blue noise texture or hash)
float TPDF(int2 pos)
{
#if PHDR_BLUE_NOISE
    float2 n = tex2Dfetch(PHDR_BlueNoiseSampler, int2(pos.x & 63, pos.y & 63)).rg;
    return n.x + n.y - 1.0 + (1.0 / 256.0);
#else
    float2 p = float2(pos) + 0.5;
    return Hash12(p) + Hash12(p + float2(127.31, 311.77)) - 1.0;
#endif
}

// ln(1 + u) for |u| <= ~0.25 without transcendental error: 2 atanh(z), z = u / (2 + u), |z| <= 0.13,
// 6 odd terms (truncation < 2e-11)
float Log1pSeries(float u)
{
    float z  = u / (2.0 + u);
    float z2 = z * z;
    float s  = 1.0 / 11.0;
    s = 1.0 / 9.0 + z2 * s;
    s = 1.0 / 7.0 + z2 * s;
    s = 1.0 / 5.0 + z2 * s;
    s = 1.0 / 3.0 + z2 * s;
    s = 1.0 + z2 * s;
    return 2.0 * z * s;
}

// exp(y) - 1 for |y| <= ~0.26 without cancellation (Taylor polynomial, Horner form)
float Expm1Series(float y)
{
    // Degree-7 Taylor polynomial: for |y| <= 0.261 the truncation (< 6e-10) is far below float32
    // rounding (measured max error 1.41 ulp, identical to the 11-term series of V9.0-9.1)
    float p = 1.0 / 5040.0;
    p = p * y + 1.0 / 720.0;
    p = p * y + 1.0 / 120.0;
    p = p * y + 1.0 / 24.0;
    p = p * y + 1.0 / 6.0;
    p = p * y + 0.5;
    p = p * y + 1.0;
    return y * p;
}

float3x3 Inverse3x3(float3x3 m)
{
    float3 a = m[0];
    float3 b = m[1];
    float3 c = m[2];
    float3 r0 = cross(b, c);
    float3 r1 = cross(c, a);
    float3 r2 = cross(a, b);
    float inv_det = 1.0 / dot(a, r0);
    return float3x3(r0.x, r1.x, r2.x,
                    r0.y, r1.y, r2.y,
                    r0.z, r1.z, r2.z) * inv_det;
}

// RGB -> XYZ matrix from CIE 1931 xy primaries and white
float3x3 PrimariesToXYZ(float2 r, float2 g, float2 b, float2 w)
{
    float3 R = float3(r.x / r.y, 1.0, (1.0 - r.x - r.y) / r.y);
    float3 G = float3(g.x / g.y, 1.0, (1.0 - g.x - g.y) / g.y);
    float3 B = float3(b.x / b.y, 1.0, (1.0 - b.x - b.y) / b.y);
    float3 W = float3(w.x / w.y, 1.0, (1.0 - w.x - w.y) / w.y);
    float3x3 P = float3x3(R.x, G.x, B.x,
                          R.y, G.y, B.y,
                          R.z, G.z, B.z);
    float3 S = mul(Inverse3x3(P), W);
    return float3x3(R.x * S.x, G.x * S.y, B.x * S.z,
                    R.y * S.x, G.y * S.y, B.y * S.z,
                    R.z * S.x, G.z * S.y, B.z * S.z);
}

float3x3 DisplayRGB_to_XYZ()
{
#if PHDR_DISPLAY_PRIMARIES == 0
    return RGB709_to_XYZ;
#elif PHDR_DISPLAY_PRIMARIES == 1
    return P3D65_to_XYZ;
#elif PHDR_DISPLAY_PRIMARIES == 2
    return RGB2020_to_XYZ;
#else
    return PrimariesToXYZ(float2(PHDR_DISPLAY_RX, PHDR_DISPLAY_RY), float2(PHDR_DISPLAY_GX, PHDR_DISPLAY_GY),
                          float2(PHDR_DISPLAY_BX, PHDR_DISPLAY_BY), float2(0.3127, 0.3290));
#endif
}

float3x3 XYZ_to_DisplayRGB()
{
#if PHDR_DISPLAY_PRIMARIES == 0
    return XYZ_to_RGB709;
#elif PHDR_DISPLAY_PRIMARIES == 1
    return XYZ_to_P3D65;
#elif PHDR_DISPLAY_PRIMARIES == 2
    return XYZ_to_RGB2020;
#else
    return Inverse3x3(DisplayRGB_to_XYZ());
#endif
}

// =================================================================================================
// 5. Projective Transformations & Stable SA-PQ Non-linear Transfer
// =================================================================================================

float3 ProjectiveTransform(float4x4 M, float3 v)
{
    float4 h = mul(M, float4(v, 1.0));
    if (any(IsNan3(h.xyz)) || (h.w != h.w))
        return float3(0.0, 0.0, 0.0);

    float w = h.w;
    if (abs(w) < PROJ_EPS)
        w = (w < 0.0) ? -PROJ_EPS : PROJ_EPS;

    return h.xyz / w;
}

// SA-PQ forward (Huang et al. Eq. 6), stable form:
//   ((c1 + c2 x) / (1 + c3 x))^n = exp(n ln(1 - d)),  d = ((1 - c1) - (c2 - c3) x) / (1 + c3 x)
float SA_PQ_Forward1(float Y)
{
    float x = PowNonNegPreserveZero(max(Y, 0.0), SA_PQ_M);
    float d = (SA_A - SA_B * x) / (1.0 + SA_PQ_C3 * x);
    return exp2(SA_PQ_N * Log1pSeries(-d) * INV_LN2);
}

// SA-PQ inverse, stable form: k = V^(1/n) = 1 + e, e = expm1(ln V / n)
//   Y = ((k - c1) / (c2 - c3 k))^(1/m) = ((1 - c1 + e) / ((c2 - c3) - c3 e))^(1/m)
float SA_PQ_Inverse1(float V)
{
    if (!(V > 0.0)) return 0.0;
    float y = log2(V) * (LN2 / SA_PQ_N);
    if (y <= SA_LN_C1) return 0.0;
    float e   = min(Expm1Series(y), SA_E_MAX);
    float num = max(SA_A + e, 0.0);
    float den = max(SA_B - SA_PQ_C3 * e, PROJ_EPS);
    return PowNonNegPreserveZero(num / den, SA_PQ_INV_M);
}

float3 SA_PQ_Forward(float3 Y)
{
    return float3(SA_PQ_Forward1(Y.x), SA_PQ_Forward1(Y.y), SA_PQ_Forward1(Y.z));
}

float3 SA_PQ_Inverse(float3 V)
{
    return float3(SA_PQ_Inverse1(V.x), SA_PQ_Inverse1(V.y), SA_PQ_Inverse1(V.z));
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

// Stable ST 2084 inverse EOTF for one value: Y (nits / 10000, clamped to [0, 1]) -> N.
// ((c1 + c2 x) / (1 + c3 x))^m2 = exp(m2 ln(1 - d)), d = (1 - c1)(1 - x) / (1 + c3 x) (exact factorization)
float PQ_EncodeNorm(float Y)
{
    Y = saturate(Y);
    float x = PowNonNegPreserveZero(Y, PQ_M1);
    float d = PQ_A * (1.0 - x) / (1.0 + PQ_C3 * x);
    return exp2(PQ_M2 * Log1pSeries(-d) * INV_LN2);
}

// Stable ST 2084 EOTF for one value: N -> Y (nits / 10000). With k = N^(1/m2) = 1 + e:
//   (k - c1) / (c2 - c3 k) = ((1 - c1) + e) / ((1 - c1) - c3 e)
float PQ_DecodeNorm(float N)
{
    if (!(N > 0.0)) return 0.0;
    N = min(N, 1.0);
    float y = log2(N) * (LN2 / PQ_M2);
    if (y <= PQ_LN_C1) return 0.0;
    float e   = Expm1Series(y);
    float num = max(PQ_A + e, 0.0);
    float den = max(PQ_A - PQ_C3 * e, FLT_MIN);
    return PowNonNegPreserveZero(num / den, PQ_INV_M1);
}

float3 PQ_EOTF(float3 N)
{
    N = saturate(N);
    return float3(PQ_DecodeNorm(N.x), PQ_DecodeNorm(N.y), PQ_DecodeNorm(N.z)) * PQ_PEAK_LUMINANCE;
}

float3 PQ_InverseEOTF(float3 L)
{
    float3 Y = clamp(L, 0.0, PQ_PEAK_LUMINANCE) / PQ_PEAK_LUMINANCE;
    return float3(PQ_EncodeNorm(Y.x), PQ_EncodeNorm(Y.y), PQ_EncodeNorm(Y.z));
}

float PQ_FromNits(float L) { return PQ_EncodeNorm(L / PQ_PEAK_LUMINANCE); }
float PQ_ToNits(float N)   { return PQ_DecodeNorm(N) * PQ_PEAK_LUMINANCE; }

// ITU-R BT.2100 HLG inverse OETF (signal -> normalized scene light E in [0, 1])
float3 HLG_InverseOETF(float3 x)
{
    x = max(x, 0.0);
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.55991073;
    float3 r;
    r.r = (x.r <= 0.5) ? (x.r * x.r) / 3.0 : (exp((x.r - c) / a) + b) / 12.0;
    r.g = (x.g <= 0.5) ? (x.g * x.g) / 3.0 : (exp((x.g - c) / a) + b) / 12.0;
    r.b = (x.b <= 0.5) ? (x.b * x.b) / 3.0 : (exp((x.b - c) / a) + b) / 12.0;
    return r;
}

// ITU-R BT.2100 HLG OETF (normalized scene light E in [0, 1] -> signal)
float3 HLG_OETF_Scene(float3 E)
{
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.55991073;
    E = max(E, 0.0);
    float3 r;
    r.r = (E.r <= (1.0 / 12.0)) ? sqrt(3.0 * E.r) : a * log(max(12.0 * E.r - b, PROJ_EPS)) + c;
    r.g = (E.g <= (1.0 / 12.0)) ? sqrt(3.0 * E.g) : a * log(max(12.0 * E.g - b, PROJ_EPS)) + c;
    r.b = (E.b <= (1.0 / 12.0)) ? sqrt(3.0 * E.b) : a * log(max(12.0 * E.b - b, PROJ_EPS)) + c;
    return r;
}

// ITU-R BT.2100 HLG reference display: nominal peak Lw (PHDR_HLG_PEAK_NITS), black 0,
// system gamma 1.2 + 0.42 log10(Lw / 1000) (BT.2100 Table 5, valid for Lw 400-2000 cd/m2)
float HLG_Lw()          { return clamp(PHDR_HLG_PEAK_NITS, 400.0, 2000.0); }
float HLG_SystemGamma() { return 1.2 + 0.42 * log10(HLG_Lw() / 1000.0); }

// ITU-R BT.2100 HLG EOTF (signal -> display light in cd/m2): OOTF F_D = Lw * Ys^(gamma - 1) * E
float3 HLG_EOTF(float3 x)
{
    float  Lw = HLG_Lw();
    float3 E  = HLG_InverseOETF(x);
    float  Ys = dot(E, Luma2020);
    return (Ys > 0.0) ? (Lw * pow(Ys, HLG_SystemGamma() - 1.0)) * E : float3(0.0, 0.0, 0.0);
}

// ITU-R BT.2100 HLG inverse EOTF (display light in cd/m2 -> signal): inverse OOTF, then OETF
float3 HLG_OETF(float3 FD)
{
    float Lw = HLG_Lw();
    float g  = HLG_SystemGamma();
    float Yd = dot(FD, Luma2020);
    if (Yd <= 0.0) return float3(0.0, 0.0, 0.0);
    float3 E = (FD / Lw) * pow(Yd / Lw, (1.0 - g) / g);
    return HLG_OETF_Scene(E);
}

int ActiveSpace()
{
    return (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
}

// SDR white of the container in nits (sRGB swapchain: 80 nits = code 1.0)
float ContainerWhiteNits(int space)
{
    return (space <= 1) ? SCRGB_WHITE_NITS : PHDR_REF_WHITE_NITS;
}

// SDR swapchain transfer (display gamma of the SDR monitor)
float3 SDR_Decode(float3 V)
{
#if PHDR_SDR_DECODE == 1 || PHDR_SDR_DECODE == 2
    return sign(V) * PowNonNegPreserveZero3(abs(V), 2.2);
#elif PHDR_SDR_DECODE == 3
    return sign(V) * PowNonNegPreserveZero3(abs(V), 2.4);
#else
    return sRGB_EOTF(V);
#endif
}

float3 SDR_Encode(float3 L)
{
#if PHDR_SDR_DECODE == 1 || PHDR_SDR_DECODE == 2
    return sign(L) * PowNonNegPreserveZero3(abs(L), 1.0 / 2.2);
#elif PHDR_SDR_DECODE == 3
    return sign(L) * PowNonNegPreserveZero3(abs(L), 1.0 / 2.4);
#else
    return sRGB_OETF(L);
#endif
}

// SDR-origin content in an HDR container was linearized by Windows / the game with piecewise sRGB.
// Re-linearize it with the curve of the display it was made on (in Rec.709, where that curve was
// applied), only within the SDR range: blended to identity between 1.0 and 1.25 x SDR white.
float3 SDRCurveFix709(float3 rgb709_nits, float W)
{
#if PHDR_SDR_DECODE == 0
    return rgb709_nits;
#else
    float3 n  = rgb709_nits / W;
    float  mx = Max3(n);
  #if PHDR_SDR_DECODE == 2
    float Y = dot(n, Luma709);
    if (!(Y > 0.0)) return rgb709_nits;
    float Yf = PowNonNegPreserveZero(sRGB_OETF(Y.xxx).x, 2.2);
    float3 fixed_rgb = rgb709_nits * (Yf / Y);
  #else
    float gamma = (PHDR_SDR_DECODE == 3) ? 2.4 : 2.2;
    float3 fixed_rgb = sign(n) * PowNonNegPreserveZero3(abs(sRGB_OETF(n)), gamma) * W;
  #endif
    return lerp(fixed_rgb, rgb709_nits, smoothstep(1.0, 1.25, mx));
#endif
}

float3 DecodeToLinear(float3 encoded, int space)
{
    [branch] if (space == 4) return HLG_EOTF(encoded);
    [branch] if (space == 3) return PQ_EOTF(encoded);
    [branch] if (space == 2) return encoded * SCRGB_WHITE_NITS;
    return SDR_Decode(encoded) * SCRGB_WHITE_NITS;
}

float3 EncodeFromLinear(float3 lin, int space)
{
    [branch] if (space == 4) return HLG_OETF(lin);
    [branch] if (space == 3) return PQ_InverseEOTF(lin);
    [branch] if (space == 2) return lin / SCRGB_WHITE_NITS;
    return SDR_Encode(lin / SCRGB_WHITE_NITS);
}

// Decode + SDR curve fix (HDR containers only) + sanitation
float3 DecodeForGrading(float3 encoded, int space)
{
    float3 lin = SanitizeLinear(DecodeToLinear(encoded, space));
#if PHDR_SDR_DECODE != 0
    [branch]
    if (space == 2)
    {
        lin = SDRCurveFix709(lin, PHDR_REF_WHITE_NITS);
    }
    else if (space >= 3)
    {
        float3 rgb709 = mul(XYZ_to_RGB709, mul(RGB2020_to_XYZ, lin));
        lin = mul(XYZ_to_RGB2020, mul(RGB709_to_XYZ, SDRCurveFix709(rgb709, PHDR_REF_WHITE_NITS)));
    }
#endif
    return lin;
}

// Hue-preserving fit of a container-RGB color (cd/m2) into [0, white] for SDR output. Negative
// channels are first lifted toward the luminance axis, then channels above white are brought down
// toward it. Luminance is preserved unless it is itself outside [0, white].
float3 FitSDRContainer(float3 rgb, float white, float3 luma_coeffs)
{
    float L = dot(rgb, luma_coeffs);
    if (L <= 0.0)   return float3(0.0, 0.0, 0.0);
    if (L >= white) return white.xxx;

    float mn = Min3(rgb);
    if (mn < 0.0) rgb = L + (L / (L - mn)) * (rgb - L);

    float mx = Max3(rgb);
    if (mx > white) rgb = L + ((white - L) / (mx - L)) * (rgb - L);

    return rgb;
}

// =================================================================================================
// 7. Iapbp Transform, Robust Gamut Tests, Boundary & Cusp Solvers
// =================================================================================================

// Absolute XYZ (cd/m2) <-> Iapbp. White-chromaticity normalization (von Kries, Huang et al. Sec. 2.1)
// with the reference white at 10,000 cd/m2, the range SA-PQ is fitted on.
float3 Iapbp_FromXYZ(float3 xyz_nits, float3 activeD65)
{
    float3 norm_xyz = (xyz_nits / IAPBP_ABS_NORM) / activeD65;
    float3 lms = ProjectiveTransform(M1_XYZ_to_LMS, norm_xyz);
    float3 lms_p = SA_PQ_Forward(max(lms, 0.0));
    return ProjectiveTransform(M2_LMSprime_to_Iapbp, lms_p);
}

float3 Iapbp_ToXYZ(float3 iapbp, float3 activeD65)
{
    float3 lms_p = ProjectiveTransform(M2_Iapbp_to_LMSprime, iapbp);
    float3 lms = SA_PQ_Inverse(lms_p);
    return ProjectiveTransform(M1_LMS_to_XYZ, lms) * activeD65 * IAPBP_ABS_NORM;
}

// Iapbp lightness of the neutral (D65) of luminance Y cd/m2
float NeutralI(float Y_nits, float3 activeD65)
{
    return (Y_nits > 0.0) ? Iapbp_FromXYZ(Y_nits * activeD65 / activeD65.y, activeD65).x : 0.0;
}

// Gamut volume: RGB of `to_target` >= 0 and, when has_upper, max channel of `to_upper` <= upper_nits
struct GamutVolume
{
    float3x3 to_target;
    bool     has_upper;
    float3x3 to_upper;
    float    upper_nits;
};

bool InsideVolumeXYZ(float3 xyz_nits, GamutVolume v)
{
    if (!(xyz_nits.y > 0.0)) return false;
    if (!(Min3(mul(v.to_target, xyz_nits)) >= 0.0)) return false;
    if (v.has_upper && Max3(mul(v.to_upper, xyz_nits)) > v.upper_nits) return false;
    return true;
}

/**
 * IapbpOutside: is the Iapbp color outside the volume?
 * Robust to the clamps of the inverse transform: a sample whose L'M'S' would be clamped by the
 * SA-PQ inverse (at or below black, or at the asymptote), whose projective denominators are not
 * positive, or whose luminance is not positive counts as outside. Without these tests, colors far
 * beyond the boundary decode to black (RGB 0,0,0) and look "inside".
 */
bool IapbpOutside(float3 iapbp, GamutVolume v, float3 activeD65)
{
    float4 h2 = mul(M2_Iapbp_to_LMSprime, float4(iapbp, 1.0));
    if (!(h2.w > 0.0)) return true;
    float3 lms_p = h2.xyz / h2.w;
    if (any(lms_p <= SA_PQ_BLACK) || any(lms_p >= SA_PQ_V_MAX)) return true;

    float3 lms = SA_PQ_Inverse(lms_p);
    float4 h1 = mul(M1_LMS_to_XYZ, float4(lms, 1.0));
    if (!(h1.w > 0.0)) return true;
    float3 xyz = (h1.xyz / h1.w) * activeD65 * IAPBP_ABS_NORM;
    return !InsideVolumeXYZ(xyz, v);
}

// Signed gamut margin (>= 0 exactly where InsideVolumeXYZ / IapbpOutside report inside), continuous
// across the boundary: min(smallest target channel / luminance, ceiling headroom / ceiling).
// Invalid domain -> -1. Used by Brent's method, which interpolates it.
float VolumeMarginXYZ(float3 xyz, GamutVolume v)
{
    if (!(xyz.y > 0.0)) return -1.0;
    float m = Min3(mul(v.to_target, xyz)) / xyz.y;
    if (v.has_upper) m = min(m, (v.upper_nits - Max3(mul(v.to_upper, xyz))) / v.upper_nits);
    return (m == m) ? m : -1.0;
}

float IapbpMargin(float3 iapbp, GamutVolume v, float3 activeD65)
{
    float4 h2 = mul(M2_Iapbp_to_LMSprime, float4(iapbp, 1.0));
    if (!(h2.w > 0.0)) return -1.0;
    float3 lms_p = h2.xyz / h2.w;
    if (any(lms_p <= SA_PQ_BLACK) || any(lms_p >= SA_PQ_V_MAX)) return -1.0;
    float3 lms = SA_PQ_Inverse(lms_p);
    float4 h1 = mul(M1_LMS_to_XYZ, float4(lms, 1.0));
    if (!(h1.w > 0.0)) return -1.0;
    return VolumeMarginXYZ((h1.xyz / h1.w) * activeD65 * IAPBP_ABS_NORM, v);
}

float LineMarginAt(float I_f, float kI, float2 dir, float l, GamutVolume v, float3 activeD65)
{
    float c = exp2(l);
    return IapbpMargin(float3(I_f + kI * c, c * dir), v, activeD65);
}

/**
 * BrentLine: Brent's method (R. P. Brent, "Algorithms for Minimization without Derivatives", 1973;
 * zbrent form) on the signed margin along the line, in log2(chroma). Bracket: margin(l_in) >= 0,
 * margin(l_out) < 0. Each step takes an inverse-quadratic or secant estimate when it is safe and a
 * bisection step otherwise, so it never does worse than bracketing. Stops at the bisection's
 * bracket width (BOUNDARY_LOG2_TOL) and returns the inside end of the final bracket.
 */
float BrentLine(float I_f, float kI, float2 dir, GamutVolume v, float3 activeD65,
                float l_in, float f_in, float l_out, float f_out)
{
    float a = l_in, fa = f_in, b = l_out, fb = f_out;
    float c = a, fc = fa, d = b - a, e = d;
    float tol1 = 0.5 * BOUNDARY_LOG2_TOL;
    [loop]
    for (int it = 0; it < BRENT_MAX_STEPS; it++)
    {
        if ((fb >= 0.0) == (fc >= 0.0)) { c = a; fc = fa; d = b - a; e = d; }
        if (abs(fc) < abs(fb)) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }
        float xm = 0.5 * (c - b);
        if (abs(xm) <= tol1 || fb == 0.0) break;
        if (abs(e) >= tol1 && abs(fa) > abs(fb))
        {
            float s = fb / fa;
            float p;
            float q;
            if (a == c)
            {
                p = 2.0 * xm * s;                      // secant
                q = 1.0 - s;
            }
            else
            {
                float qq = fa / fc;                    // inverse quadratic interpolation
                float r  = fb / fc;
                p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = abs(p);
            float min1 = 3.0 * xm * q - abs(tol1 * q);
            float min2 = abs(e * q);
            if (2.0 * p < min(min1, min2)) { e = d; d = p / q; }
            else { d = xm; e = d; }                    // interpolation unsafe: bisection step
        }
        else
        {
            d = xm; e = d;
        }
        a = b; fa = fb;
        if (abs(d) > tol1) b += d;
        else b += (xm > 0.0) ? tol1 : -tol1;
        fb = LineMarginAt(I_f, kI, dir, b, v, activeD65);
    }
    return (fb >= 0.0) ? b : c;
}

/**
 * SolveBoundaryLine: largest chroma c such that the point (I_f + kI c, c dir) in the hue plane is
 * inside the volume. kI = 0 is the constant-lightness ray from the neutral axis; kI != 0 is a
 * line through the focus point (I_f, 0). Bisection in log2(c) between 1e-9 and the projective
 * domain limit, down to a fixed bracket width (relative precision 1.04e-5 at any brightness, below the float32 floor);
 * march_first brackets the FIRST exit with an outward geometric march before bisecting. The upper
 * (ceiling) test is dropped when the neutral at the focus lightness is itself above it.
 * c_inside > 0: the caller knows the point at that chroma is inside (e.g. the pixel itself); it
 * becomes the lower end of the bracket, which saves the low-end test and a few bisection steps.
 * c_outside > 0: a point known to be outside; it becomes the upper end (no high-end test).
 * f_inside / f_outside: the signed margins at those points (used by Brent's method).
 * PHDR_BOUNDARY_SOLVER selects Brent's method (1) or bisection (0) for the final bracket.
 */
float SolveBoundaryLine(float I_f, float kI, float2 dir, GamutVolume v, float3 activeD65, bool march_first,
                        float c_inside, float c_outside, float f_inside, float f_outside)
{
    // Projective denominator of the inverse M2 along the line: W(c) = W0 + D c
    float W0 = M2_Iapbp_to_LMSprime[3][0] * I_f + M2_Iapbp_to_LMSprime[3][3];
    float D  = M2_Iapbp_to_LMSprime[3][0] * kI + M2_Iapbp_to_LMSprime[3][1] * dir.x + M2_Iapbp_to_LMSprime[3][2] * dir.y;
    float t_high = (D < -1e-5) ? min(3.0, -W0 / D * 0.85) : 3.0;

    // Drop the ceiling only when the neutral is clearly above it (pixels brighter than SDR white).
    // A neutral AT the ceiling must keep it: the cusp table's top sample is the neutral at the
    // display peak, which measures up to 1.3e-4 above the peak in the target's RGB because the
    // published container and target matrices differ in their D65 white in the 4th decimal.
    // Dropping the ceiling there made every cusp land at the peak (V9.0.0 - V9.2.0).
    if (v.has_upper)
    {
        float3 n_xyz = Iapbp_ToXYZ(float3(I_f, 0.0, 0.0), activeD65);
        if (Max3(mul(v.to_upper, n_xyz)) > v.upper_nits * 1.001) v.has_upper = false;
    }

    float l_lo;
    float f_lo = 0.0;
    [branch]
    if (c_inside > BOUNDARY_T_MIN && c_inside < t_high)
    {
        l_lo = log2(c_inside);
        f_lo = f_inside;
    }
    else
    {
#if PHDR_BOUNDARY_SOLVER == 1
        f_lo = LineMarginAt(I_f, kI, dir, log2(BOUNDARY_T_MIN), v, activeD65);
        if (f_lo < 0.0) return 0.0;
#else
        if (IapbpOutside(float3(I_f + kI * BOUNDARY_T_MIN, BOUNDARY_T_MIN * dir), v, activeD65))
            return 0.0;
#endif
        l_lo = log2(BOUNDARY_T_MIN);
    }
    float l_hi;
    float f_hi = -1.0;
    [branch]
    if (c_outside > max(c_inside, BOUNDARY_T_MIN) && c_outside <= t_high)
    {
        l_hi = log2(c_outside);
        f_hi = f_outside;
    }
    else
    {
#if PHDR_BOUNDARY_SOLVER == 1
        f_hi = LineMarginAt(I_f, kI, dir, log2(t_high), v, activeD65);
        if (f_hi >= 0.0) return t_high;
#else
        if (!IapbpOutside(float3(I_f + kI * t_high, t_high * dir), v, activeD65))
            return t_high;
#endif
        l_hi = log2(t_high);
    }

    [branch]
    if (march_first)
    {
        float l_start = max(l_lo, BOUNDARY_MARCH_START_LOG2);
        int   n_march = clamp(int(ceil((l_hi - l_start) / BOUNDARY_MARCH_STEP_LOG2)), 1, BOUNDARY_MARCH_STEPS);
        float l_step  = (l_hi - l_start) / float(n_march);
        [loop]
        for (int k = 0; k < n_march; k++)
        {
            float lk = l_start + l_step * float(k);
            if (lk <= l_lo) continue;                  // already known to be inside
#if PHDR_BOUNDARY_SOLVER == 1
            float mk = LineMarginAt(I_f, kI, dir, lk, v, activeD65);
            if (mk < 0.0)
            {
                l_hi = lk;
                f_hi = mk;
                break;
            }
            l_lo = lk;
            f_lo = mk;
#else
            float c  = exp2(lk);
            if (IapbpOutside(float3(I_f + kI * c, c * dir), v, activeD65))
            {
                l_hi = lk;
                break;
            }
            l_lo = lk;
#endif
        }
    }

#if PHDR_BOUNDARY_SOLVER == 1
    return exp2(BrentLine(I_f, kI, dir, v, activeD65, l_lo, f_lo, l_hi, f_hi));
#else
    // Halvings needed to reach the fixed bracket width (at most BOUNDARY_BISECT_STEPS)
    int steps = int(ceil(log2(max(l_hi - l_lo, BOUNDARY_LOG2_TOL) / BOUNDARY_LOG2_TOL)));
    steps = clamp(steps, 0, BOUNDARY_BISECT_STEPS);
    [loop]
    for (int iter = 0; iter < steps; iter++)
    {
        float lm = 0.5 * (l_lo + l_hi);
        float c  = exp2(lm);
        if (IapbpOutside(float3(I_f + kI * c, c * dir), v, activeD65))
            l_hi = lm;
        else
            l_lo = lm;
    }

    return exp2(l_lo);
#endif
}

// Per-hue cusp (lightness, chroma) of the display volume, from the cusp table (linear in hue)
float2 CuspLookup(float hue)
{
    float t  = frac(hue / (2.0 * PI)) * float(CUSP_LUT_SIZE);
    float t0 = floor(t);
    int   k0 = int(t0) % CUSP_LUT_SIZE;
    int   k1 = (k0 + 1) % CUSP_LUT_SIZE;
    float2 a = tex2Dfetch(PHDR_CuspSampler, int2(k0, 0)).xy;
    float2 b = tex2Dfetch(PHDR_CuspSampler, int2(k1, 0)).xy;
    return lerp(a, b, t - t0);
}

/**
 * CompressPurity (ACES Reference Gamut Compression power curve on radial purity)
 * Identity below threshold t; maps limit l to exactly 1 (the boundary); C1 at t; monotone.
 * l <= 1 (nothing expanded) is an exact identity. t >= 1 is a hard clip.
 */
float CompressPurity(float p, float t, float l)
{
    if (p <= t) return p;
    if (t >= 1.0) return min(p, 1.0);
    if (l <= 1.0 + 1e-6) return p;
    float s = (l - t) / pow(pow((1.0 - t) / (l - t), -RGC_POWER) - 1.0, 1.0 / RGC_POWER);
    float x = (p - t) / s;
    return t + s * x / pow(1.0 + pow(x, RGC_POWER), 1.0 / RGC_POWER);
}

// =================================================================================================
// 8. Physiological Human Visual System & Tone Utilities
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

// Mantiuk et al. 2009 luminance-preserving saturation for a log-log contrast slope c
float MantiukSaturation(float c)
{
    float ck = pow(max(c, 1e-6), MANTIUK_K2);
    return (1.0 + MANTIUK_K1) * ck / (1.0 + MANTIUK_K1 * ck);
}

bool ToneCurveIdentity()
{
    return abs(fContrast - 1.0) < NEUTRAL_EPS && abs(fShadows) < NEUTRAL_EPS && abs(fHighlights) < NEUTRAL_EPS;
}

bool ToneStageActive()
{
    return !ToneCurveIdentity() || fBlackPoint > NEUTRAL_EPS || fHighlightRecovery > NEUTRAL_EPS;
}

/**
 * ToneLuminance: graded luminance for input luminance L (cd/m2, after white balance and exposure)
 *   1. dehaze / black point (ratio)
 *   2. contrast around the pivot in log2: bounded y = x + (g - 1) W tanh(x / W)  (or legacy y = g x)
 *   3. shadows / highlights stop offsets, smoothstep-blended (C1)
 *   4. highlight recovery shoulder: slope 1 - A smoothstep(u), integrated in closed form (C2)
 */
float ToneLuminance(float L, float whitePt, float topNits)
{
    if (!(L > 0.0)) return L;

    float Lb = L;
    if (fBlackPoint > NEUTRAL_EPS)
    {
        Lb = L * ComputeBlackPointRatio(L, fBlackPoint * whitePt, fShadowFloor);
    }

    float Lc = Lb;
    [branch]
    if (!ToneCurveIdentity() && Lb > FLT_MIN)
    {
        float pivot = fContrastPivot * whitePt;
        float x = log2(Lb / pivot);
        float y;
        if (iContrastMode == 1)
        {
            y = x * fContrast;
        }
        else
        {
            float W = (x > 0.0) ? fContrastRolloffHi : fContrastRolloffLo;
            y = x + (fContrast - 1.0) * W * tanh(x / W);
        }
        float S = fShadows * 3.0;
        float H = fHighlights * 3.0;
        float recovery = lerp(S, H, smoothstep(-0.125, 0.125, y));
        y += recovery * (y * y) / (y * y + 6.0);
        Lc = pivot * exp2(y);
    }

    [branch]
    if (fHighlightRecovery > NEUTRAL_EPS && Lc > FLT_MIN)
    {
        float ref = (iRecoveryReference == 1) ? topNits : whitePt;
        float A  = 0.9 * saturate(fHighlightRecovery);
        float w  = max(fRecoveryBlend, 0.01);
        float s  = log2(Lc / ref);
        float u  = (s - (fRecoveryPivot - 0.5 * w)) / w;
        float I  = (u <= 0.0) ? 0.0
                 : (u < 1.0)  ? w * (u * u * u - 0.5 * u * u * u * u)
                 :              w * 0.5 + (u - 1.0) * w;
        Lc *= exp2(-A * I);
    }
    return Lc;
}

// Local log-log slope of the tone chain (central difference, +-1/32 stop)
float ToneSlope(float L, float whitePt, float topNits)
{
    if (!(L > 0.0)) return 1.0;
    float a = ToneLuminance(L * exp2(TONE_SLOPE_H), whitePt, topNits);
    float b = ToneLuminance(L * exp2(-TONE_SLOPE_H), whitePt, topNits);
    if (!(a > 0.0) || !(b > 0.0)) return 1.0;
    return (log2(a) - log2(b)) / (2.0 * TONE_SLOPE_H);
}

// =================================================================================================
// 9. Grading Context & Color Engine
// =================================================================================================

struct GradeContext
{
    int      space;
    bool     sdr;
    float    whitePt;         // reference white (SDR: 80)
    float    topNits;         // display peak (SDR: white)
    float3   lumaCoeffs;      // container luminance coefficients
    float3x3 to_XYZ;          // container RGB -> XYZ
    float3x3 to_container;    // XYZ -> container RGB
    float3   activeD65;       // container-matched D65 (Iapbp white normalization)
    bool     unclamped;       // Bypass (AP0) target
    // Gamut volumes, stored flat (ReShade FX has no nested structs); see TargetVolume() etc.
    float3x3 tvol_to_target;  // mapping volume: target primaries ...
    bool     tvol_has_upper;  // ... [+ ceiling: max channel of tvol_to_upper <= tvol_upper_nits]
    float3x3 tvol_to_upper;
    float    tvol_upper_nits;
    float3x3 reach_to_target; // reach volume: Rec.2020 in HDR, container in SDR (same ceiling)
    float3x3 reach_to_upper;
    float3x3 vib_to_target;   // vibrance reference gamut (no ceiling)
    float3x3 cusp_to_upper;   // cusp-table volume: target primaries, max channel <= topNits
    float3x3 target_to_XYZ;   // for the conformance guard
    float3   lumaTarget;
    float    I_lo;            // Iapbp lightness range of the cusp table
    float    I_top;
    float    I_knee;          // path-to-white start
    float    I_mid;           // BT.2408 18% grey (focus target)
};

// Everything except the Iapbp lightness levels (see ContextLevels / BuildContext)
GradeContext BuildContextCore()
{
    GradeContext ctx;
    ctx.space      = ActiveSpace();
    ctx.sdr        = (ctx.space <= 1);
    ctx.whitePt    = ctx.sdr ? SCRGB_WHITE_NITS : PHDR_REF_WHITE_NITS;
    ctx.topNits    = ctx.sdr ? SCRGB_WHITE_NITS : PHDR_DISPLAY_PEAK_NITS;
    ctx.unclamped  = (iGamutTarget == 4);

    if (ctx.space >= 3)
    {
        ctx.to_XYZ       = RGB2020_to_XYZ;
        ctx.to_container = XYZ_to_RGB2020;
        ctx.lumaCoeffs   = Luma2020;
    }
    else
    {
        ctx.to_XYZ       = RGB709_to_XYZ;
        ctx.to_container = XYZ_to_RGB709;
        ctx.lumaCoeffs   = Luma709;
    }
    ctx.activeD65 = mul(ctx.to_XYZ, float3(1.0, 1.0, 1.0));

    // Target primaries
    float3x3 to_T = XYZ_to_DisplayRGB();
    float3x3 T_to = DisplayRGB_to_XYZ();
    if (ctx.sdr)
    {
        to_T = XYZ_to_RGB709;
        T_to = RGB709_to_XYZ;
    }
    [flatten] if (iGamutTarget == 1) { to_T = XYZ_to_RGB709;  T_to = RGB709_to_XYZ; }
    [flatten] if (iGamutTarget == 2) { to_T = XYZ_to_P3D65;   T_to = P3D65_to_XYZ; }
    [flatten] if (iGamutTarget == 3) { to_T = XYZ_to_RGB2020; T_to = RGB2020_to_XYZ; }
    ctx.target_to_XYZ = T_to;
    ctx.lumaTarget    = T_to[1];

    // Ceiling: SDR container <= white (no ceiling in HDR)
    bool     has_upper = false;
    float3x3 to_upper  = XYZ_to_RGB709;
    float    upper     = 1e30;
    if (ctx.sdr && !ctx.unclamped)
    {
        has_upper = true;  to_upper = XYZ_to_RGB709;  upper = SCRGB_WHITE_NITS;
    }
    ctx.tvol_to_target  = to_T;
    ctx.tvol_has_upper  = has_upper;
    ctx.tvol_to_upper   = to_upper;
    ctx.tvol_upper_nits = upper;

    float3x3 to_R = XYZ_to_RGB2020;
    float3x3 to_R_upper = XYZ_to_RGB2020;
    if (ctx.sdr)
    {
        to_R = to_T;
        to_R_upper = to_upper;
    }
    ctx.reach_to_target = to_R;
    ctx.reach_to_upper  = to_R_upper;

    float3x3 to_V = XYZ_to_RGB709;
    [flatten] if (iVibranceReference == 1) to_V = XYZ_to_P3D65;
    [flatten] if (iVibranceReference == 2) to_V = XYZ_to_RGB2020;
    ctx.vib_to_target = to_V;

    ctx.cusp_to_upper = to_T;
    if (ctx.sdr) ctx.cusp_to_upper = XYZ_to_RGB709;

    ctx.I_lo   = 0.0;
    ctx.I_top  = 0.0;
    ctx.I_knee = 0.0;
    ctx.I_mid  = 0.0;
    return ctx;
}

// Iapbp lightness levels (I_lo, I_top, I_knee, I_mid). They depend only on uniforms and macros, so
// they are computed per vertex in normal mode and fold to literals in performance mode.
float4 ContextLevels(GradeContext ctx)
{
    return float4(NeutralI(1e-3, ctx.activeD65),
                  NeutralI(ctx.topNits, ctx.activeD65),
                  NeutralI(ctx.topNits * exp2(-fPathToWhiteStart), ctx.activeD65),
                  NeutralI(BT2408_GREY_FRACTION * ctx.whitePt, ctx.activeD65));
}

GradeContext WithLevels(GradeContext ctx, float4 lv)
{
    ctx.I_lo   = lv.x;
    ctx.I_top  = lv.y;
    ctx.I_knee = lv.z;
    ctx.I_mid  = lv.w;
    return ctx;
}

GradeContext BuildContext()
{
    GradeContext ctx = BuildContextCore();
    return WithLevels(ctx, ContextLevels(ctx));
}

GamutVolume MakeVolume(float3x3 to_target, bool has_upper, float3x3 to_upper, float upper_nits)
{
    GamutVolume v;
    v.to_target  = to_target;
    v.has_upper  = has_upper;
    v.to_upper   = to_upper;
    v.upper_nits = upper_nits;
    return v;
}

// Mapping volume (target primaries [+ ceiling])
GamutVolume TargetVolume(GradeContext ctx)
{
    return MakeVolume(ctx.tvol_to_target, ctx.tvol_has_upper, ctx.tvol_to_upper, ctx.tvol_upper_nits);
}

// Reach volume (Rec.2020 in HDR, container in SDR [+ the same ceiling])
GamutVolume ReachVolume(GradeContext ctx)
{
    return MakeVolume(ctx.reach_to_target, ctx.tvol_has_upper, ctx.reach_to_upper, ctx.tvol_upper_nits);
}

// Vibrance reference gamut
GamutVolume VibranceVolume(GradeContext ctx)
{
    return MakeVolume(ctx.vib_to_target, false, ctx.vib_to_target, 1e30);
}

// Bounded display volume used for the cusp table
GamutVolume CuspVolume(GradeContext ctx)
{
    return MakeVolume(ctx.tvol_to_target, true, ctx.cusp_to_upper, ctx.topNits);
}

// AP0 (physical limit for Bypass)
GamutVolume PhysVolume()
{
    return MakeVolume(XYZ_to_AP0, false, XYZ_to_AP0, 1e30);
}

struct EngineOut
{
    float3 xyz;        // output XYZ (cd/m2)
    bool   changed;    // false: the input is returned untouched (caller uses its exact path)
    float4 dbg;        // x: chroma factor (tone x path to white), y: kept fraction in gamut mapping,
                       // z: 1 if the color arrived outside the target, w: skin confidence
    float3 iab_in;     // Iapbp (display space) at the engine input
    float3 iab_out;    // Iapbp (display space) at the engine output
    float  focus_I;    // focus lightness used for gamut mapping
};

/**
 * ApplyColorEngine: chroma work in Iapbp.
 *   1. tone-dependent chroma correction (Mantiuk 2009) from the tone-chain slope
 *   2. creative purity: vibrance (bounded) and saturation, relative to the target boundary
 *   3. path to white keyed to the display peak
 *   4. gamut mapping along a line toward the focus lightness (ACES 2.0 style), RGC power curve
 *   5. Abney hue compensation (heuristic), conformance guard
 */
EngineOut ApplyColorEngine(float3 xyz, float c_tone, GradeContext ctx)
{
    EngineOut o;
    o.xyz = xyz;
    o.changed = false;
    o.dbg = float4(1.0, 1.0, 0.0, 0.0);
    o.iab_in = float3(0.0, 0.0, 0.0);
    o.iab_out = float3(0.0, 0.0, 0.0);
    o.focus_I = 0.0;

    float3 xd = xyz;

    float3 iab = Iapbp_FromXYZ(xd, ctx.activeD65);
    o.iab_in = iab;
    o.iab_out = iab;
    float C = SqrtIEEE(dot(iab.yz, iab.yz));

    // Monochrome collapse (Saturation = 0): neutral of the same luminance
    if (fSaturation <= NEUTRAL_EPS)
    {
        float3 gray = (xd.y / ctx.activeD65.y) * ctx.activeD65;
        o.xyz = gray;
        o.changed = true;
        o.iab_out = float3(iab.x, 0.0, 0.0);
        return o;
    }
    if (!(C > 1e-12)) return o;

    float2 dir = iab.yz / C;
    float hue  = atan2(dir.y, dir.x);
    float gate = smoothstep(CHROMA_NOISE_LO, CHROMA_NOISE_HI, C);
    float luma_norm = xd.y / max(ctx.whitePt, FLT_MIN);
    float ct = saturate((luma_norm - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);

    // 1. Tone-dependent chroma correction (Mantiuk 2009) from the tone-chain slope
    float s_eff = 1.0;
    if (fToneChromaCorrection > NEUTRAL_EPS && abs(c_tone - 1.0) > 1e-6)
    {
        s_eff = 1.0 + (MantiukSaturation(c_tone) - 1.0) * saturate(fToneChromaCorrection) * gate;
    }
    float C1 = C * s_eff;

    // 2. Creative purity relative to the target boundary at constant lightness
    bool creative = abs(fSaturation - 1.0) > NEUTRAL_EPS || abs(fVibrance) > NEUTRAL_EPS;
    float max_t = 0.0;
    float p0 = 0.0;
    float S_eff = 1.0;
    float C2 = C1;
    float skin = 0.0;
    [branch]
    if (creative)
    {
        skin = Evaluate3DSkinLocusLinear(xd / max(ctx.whitePt, FLT_MIN), luma_norm);
        float skin_mask = saturate(skin * fSkinProtection);
        GamutVolume creative_vol = TargetVolume(ctx);
        if (ctx.unclamped) creative_vol = ReachVolume(ctx);

        // The constant-lightness boundary is needed by vibrance and Abney. Saturation alone is a
        // radial scale whose purity form p0 S max_t equals C1 S, so the search is skipped in HDR.
        // (SDR keeps the search; where it returns 0 the purity is undefined, see C2 below.)
        bool need_max_t = abs(fVibrance) > NEUTRAL_EPS || fAbneyCorrection > NEUTRAL_EPS || ctx.tvol_has_upper;
        float p1 = 0.0;
        [branch]
        if (need_max_t)
        {
            // The pixel itself (chroma C at this lightness) is a known inside point when it is in the volume
            float f_in_t = VolumeMarginXYZ(xd, creative_vol);
            float c_in_t = (f_in_t >= 0.0) ? C * (1.0 - 1e-4) : 0.0;
            max_t = SolveBoundaryLine(iab.x, 0.0, dir, creative_vol, ctx.activeD65, false, c_in_t, 0.0, f_in_t, 0.0);
            p0 = C1 / max(max_t, FLT_MIN);
            p1 = p0;
        }

        // Vibrance: demand from purity in the SOURCE gamut, bounded expansion p' = 1 - (1 - p)^(1 + V w)
        [branch]
        if (abs(fVibrance) > NEUTRAL_EPS && p0 < 1.0)
        {
            float f_in_s = VolumeMarginXYZ(xd, VibranceVolume(ctx));
            float c_in_s = (f_in_s >= 0.0) ? C * (1.0 - 1e-4) : 0.0;
            float max_s = SolveBoundaryLine(iab.x, 0.0, dir, VibranceVolume(ctx), ctx.activeD65, false, c_in_s, 0.0, f_in_s, 0.0);
            float ps = saturate(C1 / max(max_s, FLT_MIN));
            float demand = 1.0 - smoothstep(0.0, 1.0, ps);
            float w = demand * (1.0 - skin_mask) * chroma_reliability * gate;
            float g = max(1.0 + fVibrance * w, 0.0);
            p1 = 1.0 - PowNonNegPreserveZero(1.0 - p0, g);
        }

        // Saturation: uniform radial scale (boosts skin-protected, reliability- and gate-weighted)
        S_eff = fSaturation;
        if (S_eff > 1.0)
        {
            float master_boost = (S_eff - 1.0) * (1.0 - skin_mask * 0.85);
            S_eff = 1.0 + master_boost * chroma_reliability * gate;
        }
        else
        {
            S_eff = lerp(1.0, S_eff, gate);
        }
        // No displayable chroma at this lightness (max_t = 0, e.g. above SDR white): purity is
        // undefined, apply the plain radial scale (before V9.2.2 this collapsed the color to white)
        C2 = (need_max_t && max_t > FLT_MIN) ? (p1 * S_eff * max_t) : (C1 * S_eff);
    }

    // 3. Path to white, keyed to the display peak; less for colors already near the cusp
    float ptw = 1.0;
    float2 cusp = float2(0.0, 0.0);
    bool need_cusp = (fPathToWhite > NEUTRAL_EPS) || (fGamutFocus > NEUTRAL_EPS);
    if (need_cusp && !ctx.unclamped) cusp = CuspLookup(hue);
    [branch]
    if (fPathToWhite > NEUTRAL_EPS && !ctx.unclamped)
    {
        float t = smoothstep(ctx.I_knee, ctx.I_top, iab.x);
        if (t > 0.0)
        {
            ptw = 1.0 - saturate(fPathToWhite) * t * (1.0 - 0.5 * saturate(C2 / max(cusp.y, FLT_MIN)));
        }
    }
    float C3 = C2 * ptw;

    // 4. Gamut mapping
    float I4 = iab.x;
    float C4 = C3;
    bool mapped = false;
    [branch]
    if (ctx.unclamped)
    {
        // Bypass: physically realizable colors only (inside AP0, positive luminance)
        if (abs(C3 - C) > 0.0)
        {
            float max_phys = SolveBoundaryLine(iab.x, 0.0, dir, PhysVolume(), ctx.activeD65, true, 0.0, 0.0, 0.0, 0.0);
            C4 = min(C3, max_phys * (1.0 - 1e-5));
            mapped = true;
        }
    }
    else
    {
        bool inside_in = InsideVolumeXYZ(xd, TargetVolume(ctx));
        if (!inside_in) o.dbg.z = 1.0;
        bool reach_mode = (iOutOfGamutMode == 1);
        if (reach_mode || !inside_in || C3 > C * (1.0 + 1e-6))
        {
            float I_f = iab.x;
            if (fGamutFocus > NEUTRAL_EPS && C3 > 1e-4)
            {
                float I_focus = clamp(0.5 * (cusp.x + ctx.I_mid), ctx.I_lo, ctx.I_top * 0.999);
                I_f = lerp(iab.x, I_focus, saturate(fGamutFocus));
            }
            float kI = (C3 > FLT_MIN) ? (iab.x - I_f) / C3 : 0.0;
            float knee = saturate(fGamutGuardKnee);
            float t_knee = 1.0 - knee;
            o.focus_I = I_f;

            // Without a ceiling (HDR) every mapping line leaves the volume exactly once (measured:
            // 0 re-entries in 12,960 lines, P3 / Rec.709 / Rec.2020). Then the point at chroma C3 / t
            // on the line is inside exactly when q = C3 / cb <= t, where the compression curve is
            // the identity: one test replaces the boundary search, and no outward march is needed.
            // With the SDR white ceiling a line can exit and re-enter, so the full march is kept.
            bool  identity = false;
            float c_in_m  = 0.0;
            float c_out_m = 0.0;
            float f_in_m  = 0.0;
            float f_out_m = 0.0;
            [branch]
            if (!ctx.tvol_has_upper && t_knee > 0.0)
            {
                // Exact bracket for the search: C3 / t is outside; the color itself (chroma C3) is
                // usually still inside (compressed only because it is past the knee)
                float c_t = C3 / t_knee;
#if PHDR_BOUNDARY_SOLVER == 1
                float m_t = IapbpMargin(float3(I_f + kI * c_t, c_t * dir), TargetVolume(ctx), ctx.activeD65);
                identity = (m_t >= 0.0);
                if (!identity)
                {
                    c_out_m = c_t;
                    f_out_m = m_t;
                    float m_3 = IapbpMargin(float3(iab.x, C3 * dir), TargetVolume(ctx), ctx.activeD65);
                    if (m_3 >= 0.0) { c_in_m = C3; f_in_m = m_3; }
                    else            { c_out_m = C3; f_out_m = m_3; }
                }
#else
                identity = !IapbpOutside(float3(I_f + kI * c_t, c_t * dir), TargetVolume(ctx), ctx.activeD65);
                if (!identity)
                {
                    c_out_m = c_t;
                    if (!IapbpOutside(float3(iab.x, C3 * dir), TargetVolume(ctx), ctx.activeD65))
                        c_in_m = C3;
                    else
                        c_out_m = C3;
                }
#endif
            }

            [branch]
            if (!identity)
            {
                bool  march = (fGamutFocus > NEUTRAL_EPS) && ctx.tvol_has_upper;
                float cb = SolveBoundaryLine(I_f, kI, dir, TargetVolume(ctx), ctx.activeD65, march, c_in_m, c_out_m, f_in_m, f_out_m);
                float q  = C3 / max(cb, FLT_MIN);
                float expansion = max(S_eff, 1.0) * max(s_eff, 1.0);
                float lim = max(expansion, 1.0);
                if (reach_mode)
                {
                    float cr = SolveBoundaryLine(I_f, kI, dir, ReachVolume(ctx), ctx.activeD65, march, 0.0, 0.0, 0.0, 0.0);
                    lim = max(lim, cr / max(cb, FLT_MIN));
                }
                if (q > lim) lim = q;
                float q2 = q;
                if (q > t_knee)
                {
                    q2 = min(CompressPurity(q, t_knee, lim), 1.0 - 1e-5);
                }
                C4 = q2 * cb;
                I4 = I_f + kI * C4;
                o.dbg.y = C4 / max(C3, FLT_MIN);
                mapped = true;
            }
        }
    }

    // 5. Abney hue compensation (heuristic): rotation proportional to the purity change
    float2 out_dir = dir;
    bool rotated = false;
    [branch]
    if (creative && fAbneyCorrection > NEUTRAL_EPS && max_t > FLT_MIN)
    {
        float angle = hue;
        float abney_profile = 0.15 * sin(2.0 * angle + 0.4) * (1.0 + 0.3 * cos(angle));
        float purity_change = saturate(C4 / max_t) - saturate(p0);
        angle += abney_profile * purity_change * saturate(fAbneyCorrection) * chroma_reliability;
        out_dir = float2(cos(angle), sin(angle));
        rotated = true;
        GamutVolume keep = TargetVolume(ctx);
        if (ctx.unclamped) keep = PhysVolume();
        if (IapbpOutside(float3(I4, C4 * out_dir), keep, ctx.activeD65))
        {
            float c_new = SolveBoundaryLine(I4, 0.0, out_dir, keep, ctx.activeD65, ctx.unclamped, 0.0, 0.0, 0.0, 0.0);
            C4 = min(C4, c_new * (1.0 - 1e-5));
        }
    }

    o.dbg.x = s_eff * ptw;
    o.dbg.w = skin;

    // Exact passthrough when nothing changed
    if (!mapped && !rotated && C3 == C)
    {
        return o;
    }

    float3 iab_out = float3(I4, C4 * out_dir);
    o.iab_out = iab_out;
    float3 out_d = Iapbp_ToXYZ(iab_out, ctx.activeD65);

    // Strict target-primary non-negativity conformance guard (numerical safety)
    if (!ctx.unclamped)
    {
        float3 rgb_b = mul(ctx.tvol_to_target, out_d);
        float min_c = Min3(rgb_b);
        if (min_c < 0.0)
        {
            float luma_b = dot(rgb_b, ctx.lumaTarget);
            float t_desat = saturate(-min_c / max(luma_b - min_c, FLT_MIN));
            rgb_b = max(lerp(rgb_b, luma_b.xxx, t_desat), 0.0);
            out_d = mul(ctx.target_to_XYZ, rgb_b);
        }
    }

    o.xyz = out_d;
    o.changed = true;
    return o;
}

// =================================================================================================
// 10. White Balance (Planckian locus + two-step CAT16)
// =================================================================================================

// Planckian locus (Krystek 1985 rational approximation, CIE 1960 uv, 1000-15000 K, ~8e-5 in uv)
float2 PlanckianUV(float T)
{
    float T2 = T * T;
    float u = (0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T2) / (1.0 + 8.42420235e-4 * T + 7.08145163e-7 * T2);
    float v = (0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T2) / (1.0 - 2.89741816e-5 * T + 1.61456053e-7 * T2);
    return float2(u, v);
}

// CIE XYZ (Y = 1) of the white at correlated color temperature T (K) and distance Duv from the locus
float3 WhitePointXYZ(float T, float duv)
{
    float2 uv  = PlanckianUV(T);
    float2 tng = PlanckianUV(T + 1.0) - PlanckianUV(T - 1.0);          // locus tangent (increasing T)
    float2 nrm = float2(tng.y, -tng.x) / max(length(tng), FLT_MIN);   // normal toward +v (green side)
    uv += duv * nrm;
    float den = 2.0 * uv.x - 8.0 * uv.y + 4.0;
    float x = 3.0 * uv.x / den;
    float y = 2.0 * uv.y / den;
    return float3(x / y, 1.0, (1.0 - x - y) / y);
}

bool WhiteBalanceActive()
{
    return abs(fWhiteBalanceK - WB_REF_KELVIN) > 0.5 || abs(fWhiteBalanceDuv - WB_REF_DUV) > 1e-6;
}

// Two-step CAT16 gains (source white -> equal-energy E -> reference white) with degree D,
// normalized so neutral (D65) luminance is preserved. D = 1 equals the one-step von Kries ratio.
float3 WhiteBalanceGains()
{
    float3 srcWhite = WhitePointXYZ(clamp(fWhiteBalanceK, 1000.0, 15000.0), fWhiteBalanceDuv);
    float3 refWhite = WhitePointXYZ(WB_REF_KELVIN, WB_REF_DUV);
    float  D = saturate(fAdaptationDegree);
    float3 Rs = mul(XYZ_to_CAT16, srcWhite);
    float3 Rd = mul(XYZ_to_CAT16, refWhite);
    float3 gains = (D / Rs + (1.0 - D)) / (D / Rd + (1.0 - D));

    const float3 standardD65 = float3(0.950456, 1.000000, 1.089058);
    float3 adaptedXYZ = mul(CAT16_to_XYZ, mul(XYZ_to_CAT16, standardD65) * gains);
    return gains / max(adaptedXYZ.y, FLT_MIN);
}

// =================================================================================================
// 11. Full Pixel Grade
// =================================================================================================

struct GradeResult
{
    float3 color;      // container RGB (cd/m2)
    bool   graded;     // false: bit-exact passthrough candidate (output equals input)
    float4 dbg;
    float3 iab_in;
    float3 iab_out;
    float  focus_I;
    float  skin;
};

bool DebugActive()
{
    return iDebugMode != 0 || bSplitView;
}

GradeResult GradePixel(float3 original_lin, float3 wb_gains, GradeContext ctx)
{
    GradeResult r;
    r.dbg = float4(1.0, 1.0, 0.0, 0.0);
    r.iab_in = float3(0.0, 0.0, 0.0);
    r.iab_out = float3(0.0, 0.0, 0.0);
    r.focus_I = 0.0;
    r.skin = 0.0;
    r.graded = true;

    // Stage 1: linear scene grade (CAT16 white balance, exposure)
    float3 lin_xyz = mul(ctx.to_XYZ, original_lin);
    bool wb_active = WhiteBalanceActive();
    if (wb_active)
    {
        lin_xyz = mul(CAT16_to_XYZ, mul(XYZ_to_CAT16, lin_xyz) * wb_gains);
    }
    float exposure_gain = (abs(fExposure) > NEUTRAL_EPS) ? exp2(fExposure) : 1.0;
    lin_xyz *= exposure_gain;

    // Stage 2: tone (luminance ratio) and its local log-log slope
    float luma = lin_xyz.y;
    float tone_ratio = 1.0;
    float c_tone = 1.0;
    [branch]
    if (ToneStageActive() && luma > 0.0)
    {
        tone_ratio = ToneLuminance(luma, ctx.whitePt, ctx.topNits) / luma;
        if (fToneChromaCorrection > NEUTRAL_EPS)
        {
            c_tone = ToneSlope(luma, ctx.whitePt, ctx.topNits);
        }
    }
    lin_xyz *= tone_ratio;

    // Stage 3: chroma engine (Iapbp)
    bool rgb_neutral = !wb_active && (original_lin.r == original_lin.g) && (original_lin.g == original_lin.b);

    // The engine cannot change a color when no chroma operation is active and the color is inside
    // the target (or the target is Bypass): skip it, including the Iapbp transform. Debug views
    // keep the full path for their diagnostics.
    bool chroma_ops = abs(fSaturation - 1.0) > NEUTRAL_EPS || abs(fVibrance) > NEUTRAL_EPS ||
                      fPathToWhite > NEUTRAL_EPS || iOutOfGamutMode == 1 ||
                      (fToneChromaCorrection > NEUTRAL_EPS && abs(c_tone - 1.0) > 1e-6);
    bool engine_identity = !chroma_ops && !DebugActive() &&
                           (ctx.unclamped || InsideVolumeXYZ(lin_xyz, TargetVolume(ctx)));
    float3 color;
    [branch]
    if (fSaturation <= NEUTRAL_EPS)
    {
        // Bit-exact monochrome collapse (Saturation = 0.00): R = G = B = container luminance
        float3 graded_rgb = wb_active ? mul(ctx.to_container, lin_xyz) : original_lin * (exposure_gain * tone_ratio);
        color = dot(graded_rgb, ctx.lumaCoeffs).xxx;
    }
    else if (rgb_neutral)
    {
        color = original_lin * (exposure_gain * tone_ratio);
    }
    else if (engine_identity)
    {
        // Same as the engine's "unchanged" result below
        color = wb_active ? mul(ctx.to_container, lin_xyz) : original_lin * (exposure_gain * tone_ratio);
    }
    else
    {
        EngineOut e = ApplyColorEngine(lin_xyz, c_tone, ctx);
        r.dbg = e.dbg;
        r.iab_in = e.iab_in;
        r.iab_out = e.iab_out;
        r.focus_I = e.focus_I;
        r.skin = e.dbg.w;
        if (e.changed)
            color = mul(ctx.to_container, e.xyz);
        else
            color = wb_active ? mul(ctx.to_container, lin_xyz) : original_lin * (exposure_gain * tone_ratio);
    }

    bool is_invalid = any(IsNan3(color)) || any(IsInf3(color));
    color = is_invalid ? original_lin : color;

    // Hue-preserving SDR ceiling: no per-channel clipping above white on any grading path
    [branch]
    if (ctx.sdr && !ctx.unclamped)
    {
        color = FitSDRContainer(color, ctx.whitePt, ctx.lumaCoeffs);
    }
    r.color = color;
    return r;
}

bool GradeIsIdentity()
{
    return abs(fExposure) < NEUTRAL_EPS && fBlackPoint < NEUTRAL_EPS &&
           ToneCurveIdentity() && !WhiteBalanceActive() &&
           abs(fSaturation - 1.0) < NEUTRAL_EPS && abs(fVibrance) < NEUTRAL_EPS &&
           fHighlightRecovery < NEUTRAL_EPS && fPathToWhite < NEUTRAL_EPS &&
           iGamutTarget == 0 && iOutOfGamutMode == 0 &&
           (PHDR_SDR_DECODE == 0 || ActiveSpace() <= 1);
}

// =================================================================================================
// 12. Debug Diagnostic Functions
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

// 0 -> black, 0.25 -> blue, 0.5 -> green, 0.75 -> yellow, 1 -> red
float3 Heatmap(float t)
{
    t = saturate(t);
    if (t < 0.25) return float3(0.0, 0.0, t / 0.25);
    if (t < 0.5)  return float3(0.0, (t - 0.25) / 0.25, 1.0 - (t - 0.25) / 0.25);
    if (t < 0.75) return float3((t - 0.5) / 0.25, 1.0, 0.0);
    return float3(1.0, 1.0 - (t - 0.75) / 0.25, 0.0);
}

// ITU-R BT.2100 ICtCp from Rec.2020 RGB (cd/m2)
float3 ICtCp_FromRGB2020(float3 rgb_nits)
{
    float3 lms = mul(RGB2020_to_LMS_ICtCp, max(rgb_nits, 0.0)) / PQ_PEAK_LUMINANCE;
    float3 lp  = float3(PQ_EncodeNorm(lms.x), PQ_EncodeNorm(lms.y), PQ_EncodeNorm(lms.z));
    return float3(0.5 * lp.x + 0.5 * lp.y,
                  dot(lp, float3( 6610.0, -13613.0,  7003.0)) / 4096.0,
                  dot(lp, float3(17933.0, -17390.0,  -543.0)) / 4096.0);
}

// ITU-R BT.2124 Delta E ITP (1 = one just-noticeable difference)
float DeltaE_ITP(float3 a_rgb2020, float3 b_rgb2020)
{
    float3 a = ICtCp_FromRGB2020(a_rgb2020);
    float3 b = ICtCp_FromRGB2020(b_rgb2020);
    float3 d = float3(a.x - b.x, 0.5 * (a.y - b.y), a.z - b.z);
    return 720.0 * sqrt(dot(d, d));
}

// Hue-slice scope (Iapbp lightness vs chroma for the hue under the mouse), drawn in a corner inset
static const int SCOPE_SIZE   = 360;
static const int SCOPE_MARGIN = 16;

#if !PHDR_PERF_MODE
float3 DrawScope(int2 pos, float3 wb_gains, GradeContext ctx)
{
    int2 dims = int2(BUFFER_WIDTH, BUFFER_HEIGHT);
    int2 mp = clamp(int2(MousePoint), int2(0, 0), dims - 1);
    float3 m_lin = DecodeForGrading(tex2Dfetch(SamplerBackBuffer, mp).rgb, ctx.space);
    GradeResult g = GradePixel(m_lin, wb_gains, ctx);

    float3 iab_in  = g.iab_in;
    float3 iab_out = g.iab_out;
    float  c_in  = SqrtIEEE(dot(iab_in.yz, iab_in.yz));
    float  c_out = SqrtIEEE(dot(iab_out.yz, iab_out.yz));
    float2 dir = (c_in > 1e-9) ? iab_in.yz / c_in : float2(1.0, 0.0);
    float2 cusp = ctx.unclamped ? float2(0.0, 0.0) : CuspLookup(atan2(dir.y, dir.x));

    float c_axis = 1.25 * max(max(max(cusp.y, c_in), c_out), 0.05);
    float i_axis = 1.10 * max(max(ctx.I_top, iab_in.x), iab_out.x);

    int2 origin = int2(SCOPE_MARGIN, dims.y - SCOPE_MARGIN - SCOPE_SIZE);
    float2 uv = float2(pos - origin) / float(SCOPE_SIZE);
    float C = uv.x * c_axis;
    float I = (1.0 - uv.y) * i_axis;
    float px_c = c_axis / float(SCOPE_SIZE);
    float px_i = i_axis / float(SCOPE_SIZE);

    float3 col = float3(0.015, 0.015, 0.02);
    if (!IapbpOutside(float3(I, C * dir), ReachVolume(ctx), ctx.activeD65)) col = float3(0.06, 0.06, 0.07);
    GamutVolume tv = TargetVolume(ctx);
    if (ctx.unclamped) tv = PhysVolume();
    if (!IapbpOutside(float3(I, C * dir), tv, ctx.activeD65)) col = float3(0.16, 0.16, 0.17);

    // Reference lightness lines: 18% grey, reference white, display peak
    if (abs(I - ctx.I_mid) < px_i) col = float3(0.2, 0.6, 0.2);
    if (abs(I - NeutralI(ctx.whitePt, ctx.activeD65)) < px_i) col = float3(0.8, 0.8, 0.8);
    if (abs(I - ctx.I_top) < px_i) col = float3(0.8, 0.2, 0.2);

    // Input -> output segment and markers
    float2 p_in  = float2(c_in / px_c, (i_axis - iab_in.x) / px_i);
    float2 p_out = float2(c_out / px_c, (i_axis - iab_out.x) / px_i);
    float2 p_cus = float2(cusp.y / px_c, (i_axis - cusp.x) / px_i);
    float2 p_foc = float2(0.0, (i_axis - g.focus_I) / px_i);
    float2 here  = float2(pos - origin);
    float2 seg = p_out - p_in;
    float  t = saturate(dot(here - p_in, seg) / max(dot(seg, seg), 1e-6));
    if (length(here - (p_in + t * seg)) < 1.0) col = float3(0.2, 0.9, 0.3);
    if (!ctx.unclamped && length(here - p_cus) < 4.0) col = float3(1.0, 0.9, 0.1);
    if (g.focus_I > 0.0 && length(here - p_foc) < 4.0) col = float3(0.1, 0.9, 1.0);
    if (length(here - p_in) < 4.0)  col = float3(1.0, 1.0, 1.0);
    if (length(here - p_out) < 3.0) col = float3(0.2, 1.0, 0.3);

    // Frame
    if (here.x < 1.0 || here.y < 1.0 || here.x > float(SCOPE_SIZE) - 2.0 || here.y > float(SCOPE_SIZE) - 2.0)
        col = float3(0.5, 0.5, 0.5);
    return col;
}
#endif

// =================================================================================================
// 13. Pass Scheduling & Vertex Shaders
// =================================================================================================
//
// Work that depends only on settings is done once, not per pixel or per frame:
// - A pass whose result is not needed is skipped in its vertex shader: the triangle is moved
//   outside the viewport, no pixel shader runs, and the target keeps its contents exactly. For a
//   back-buffer pass this leaves the image bit-identical to a pass-through.
// - The cusp table is rebuilt only when its inputs (CuspTableKey) differ from those it was built
//   for. In performance mode those inputs are compile-time constants: it is built once.
// - White balance gains and the Iapbp lightness levels: per vertex in normal mode (3 evaluations
//   per frame instead of one per pixel); compile-time literals in performance mode.

// Clip-space position outside the viewport: a triangle made of it rasterizes no pixels
static const float4 PHDR_CULLED_POSITION = float4(-2.0, -2.0, 0.0, 1.0);

// Bump when the cusp table algorithm or layout changes (invalidates stored tables)
static const float CUSP_TABLE_VERSION = 90202.0;

// The scene grade is bit-transparent and no debug / split view is shown: skip the pass entirely
bool SceneGradeSkippable()
{
    return !DebugActive() && GradeIsIdentity();
}

// The cusp table is read by path to white, the focus lightness of gamut mapping and the scope
bool CuspTableNeeded()
{
    if (iGamutTarget == 4) return false;               // Bypass (AP0): no cusp anywhere
    if (SceneGradeSkippable()) return false;           // the grading pass does not run
    if (iDebugMode == 12) return true;                 // hue-slice scope draws the cusp
    return fPathToWhite > NEUTRAL_EPS || fGamutFocus > NEUTRAL_EPS;
}

// FNV-1a of the custom display primaries (0 for the built-in sets); exact integer arithmetic
uint DisplayPrimariesHash()
{
#if PHDR_DISPLAY_PRIMARIES == 3
    uint h = 2166136261u;
    h = (h ^ uint(PHDR_DISPLAY_RX * 100000.0 + 0.5)) * 16777619u;
    h = (h ^ uint(PHDR_DISPLAY_RY * 100000.0 + 0.5)) * 16777619u;
    h = (h ^ uint(PHDR_DISPLAY_GX * 100000.0 + 0.5)) * 16777619u;
    h = (h ^ uint(PHDR_DISPLAY_GY * 100000.0 + 0.5)) * 16777619u;
    h = (h ^ uint(PHDR_DISPLAY_BX * 100000.0 + 0.5)) * 16777619u;
    h = (h ^ uint(PHDR_DISPLAY_BY * 100000.0 + 0.5)) * 16777619u;
    return h & 0xFFFFFFu;
#else
    return 0u;
#endif
}

// Everything the cusp table depends on, as exactly representable floats (bit-exact comparison).
// Runtime inputs: active color space (incl. override) and gamut target. Macro inputs (display
// peak / primaries) also force a recompile, which clears the key anyway.
float4 CuspTableKey()
{
    uint code = uint(ActiveSpace()) + 8u * uint(iGamutTarget) + 64u * uint(PHDR_DISPLAY_PRIMARIES) + 1u;
    return float4(float(code), PHDR_DISPLAY_PEAK_NITS, float(DisplayPrimariesHash()), CUSP_TABLE_VERSION);
}

struct VS_Output
{
    float4 vpos : SV_Position;
    float2 texcoord : TEXCOORD0;
#if !PHDR_PERF_MODE
    nointerpolation float3 wbGainCAT16 : TEXCOORD1;
    nointerpolation float4 levels      : TEXCOORD2;   // I_lo, I_top, I_knee, I_mid
#endif
};

VS_Output VS_PhotorealHDR(uint id : SV_VertexID)
{
    VS_Output output;
    output.texcoord.x = (id == 2) ? 2.0 : 0.0;
    output.texcoord.y = (id == 1) ? 2.0 : 0.0;
    output.vpos = float4(output.texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    if (SceneGradeSkippable()) output.vpos = PHDR_CULLED_POSITION;
#if !PHDR_PERF_MODE
    output.wbGainCAT16 = WhiteBalanceGains();
    output.levels      = ContextLevels(BuildContextCore());
#endif
    return output;
}

// Cusp table pass: skipped when the table is not needed or is already valid for the current key
void VS_CuspLUT(uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
    texcoord.x = (id == 2) ? 2.0 : 0.0;
    texcoord.y = (id == 1) ? 2.0 : 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

    bool skip = true;
    if (CuspTableNeeded())
    {
        float4 stored = tex2Dfetch(PHDR_CuspKeySampler, int2(0, 0));
        skip = all(stored == CuspTableKey());
    }
    if (skip) position = PHDR_CULLED_POSITION;
}

// =================================================================================================
// 14. Pixel Shaders
// =================================================================================================

// Cusp table: for each of 1024 hues, the (lightness, chroma) of maximum chroma in the bounded
// display volume. 64-point scan of exact constant-lightness boundaries, then 40 golden-section steps.
void PS_CuspLUT(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 cuspOut : SV_Target)
{
    GradeContext ctx = BuildContext();
    cuspOut = float4(0.0, 0.0, 0.0, 1.0);
    if (ctx.unclamped) return;

    float hue = 2.0 * PI * floor(vpos.x) / float(CUSP_LUT_SIZE);
    float2 dir = float2(cos(hue), sin(hue));
    float I_lo = ctx.I_lo;
    float I_hi = ctx.I_top;

    float best_I = I_lo;
    float best_C = -1.0;
    int   best_k = 0;
    [loop]
    for (int k = 0; k < CUSP_SCAN_STEPS; k++)
    {
        float I = lerp(I_lo, I_hi, float(k) / float(CUSP_SCAN_STEPS - 1));
        float c = SolveBoundaryLine(I, 0.0, dir, CuspVolume(ctx), ctx.activeD65, false, 0.0, 0.0, 0.0, 0.0);
        if (c > best_C) { best_C = c; best_I = I; best_k = k; }
    }

    float step_I = (I_hi - I_lo) / float(CUSP_SCAN_STEPS - 1);
    float a = max(best_I - step_I, I_lo);
    float b = min(best_I + step_I, I_hi);
    const float g = 0.6180339887498949;
    float x1 = b - g * (b - a);
    float x2 = a + g * (b - a);
    float f1 = SolveBoundaryLine(x1, 0.0, dir, CuspVolume(ctx), ctx.activeD65, false, 0.0, 0.0, 0.0, 0.0);
    float f2 = SolveBoundaryLine(x2, 0.0, dir, CuspVolume(ctx), ctx.activeD65, false, 0.0, 0.0, 0.0, 0.0);
    [loop]
    for (int it = 0; it < CUSP_GOLDEN_STEPS; it++)
    {
        if (f1 > f2)
        {
            b = x2;  x2 = x1;  f2 = f1;
            x1 = b - g * (b - a);
            f1 = SolveBoundaryLine(x1, 0.0, dir, CuspVolume(ctx), ctx.activeD65, false, 0.0, 0.0, 0.0, 0.0);
        }
        else
        {
            a = x1;  x1 = x2;  f1 = f2;
            x2 = a + g * (b - a);
            f2 = SolveBoundaryLine(x2, 0.0, dir, CuspVolume(ctx), ctx.activeD65, false, 0.0, 0.0, 0.0, 0.0);
        }
    }
    float I_c = 0.5 * (a + b);
    float C_c = SolveBoundaryLine(I_c, 0.0, dir, CuspVolume(ctx), ctx.activeD65, false, 0.0, 0.0, 0.0, 0.0);
    if (C_c < best_C) { C_c = best_C; I_c = best_I; }
    cuspOut = float4(I_c, C_c, 0.0, 1.0);
}

// Records which inputs the cusp table now holds (runs after PS_CuspLUT, one pixel)
void PS_CuspKey(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 keyOut : SV_Target)
{
    keyOut = CuspTableNeeded() ? CuspTableKey() : float4(0.0, 0.0, 0.0, 0.0);
}

void PS_PhotorealHDR(VS_Output input, out float4 fragColor : SV_Target)
{
    int2 pos   = int2(input.vpos.xy);
    float4 src = tex2Dfetch(SamplerBackBuffer, pos);

#if !PHDR_PERF_MODE
    // Split view: ungraded left part (with a divider line)
    [branch]
    if (bSplitView)
    {
        float split_x = fSplitPosition * float(BUFFER_WIDTH);
        if (abs(float(pos.x) + 0.5 - split_x) < 1.0)
        {
            fragColor = float4(EncodeDebug(float3(0.6, 0.6, 0.6), ActiveSpace()), src.a);
            return;
        }
        if (float(pos.x) + 0.5 < split_x)
        {
            fragColor = src;
            return;
        }
    }
#endif

    // Bit-transparent bypass check
    [branch]
    if (iDebugMode == 0 && GradeIsIdentity())
    {
        fragColor = src;
        return;
    }

#if PHDR_PERF_MODE
    // Uniforms are compile-time constants: the gains and the context fold to literals
    float3 wb_gains = WhiteBalanceGains();
    GradeContext ctx = BuildContext();
#else
    float3 wb_gains = input.wbGainCAT16;
    GradeContext ctx = WithLevels(BuildContextCore(), input.levels);
#endif
    int space = ctx.space;

#if !PHDR_PERF_MODE
    // Scope inset
    [branch]
    if (iDebugMode == 12)
    {
        int2 origin = int2(SCOPE_MARGIN, BUFFER_HEIGHT - SCOPE_MARGIN - SCOPE_SIZE);
        int2 rel = pos - origin;
        if (all(rel >= 0) && all(rel < SCOPE_SIZE))
        {
            fragColor = float4(EncodeDebug(DrawScope(pos, wb_gains, ctx), space), src.a);
            return;
        }
    }
#endif

    float3 original_lin = DecodeForGrading(src.rgb, space);
    GradeResult g = GradePixel(original_lin, wb_gains, ctx);
    float3 color = g.color;

#if !PHDR_PERF_MODE
    // ---------------------------------------------------------------------------------------------
    // DEBUG VISUALIZATION (Operating directly on graded output)
    // ---------------------------------------------------------------------------------------------
    [branch]
    if (iDebugMode != 0 && iDebugMode != 12)
    {
        float3 debug_out = float3(0.0, 0.0, 0.0);
        float3 lumaCoeffs = ctx.lumaCoeffs;
        float  whitePt = ctx.whitePt;

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
        else if (iDebugMode == 3) // CAT16 Cone Response
        {
            float3 c16 = mul(XYZ_to_CAT16, mul(ctx.to_XYZ, color));
            float max_c16 = max(max(abs(c16.r), abs(c16.g)), abs(c16.b));
            if (max_c16 > FLT_MIN)
                debug_out = abs(c16) / max_c16;
        }
        else if (iDebugMode == 4) // Iapbp Chroma / Saturation (Graded Output)
        {
            float3 iapbp_dbg = Iapbp_FromXYZ(mul(ctx.to_XYZ, color), ctx.activeD65);
            float ch = SqrtIEEE(dot(iapbp_dbg.yz, iapbp_dbg.yz));
            float v = saturate(ch * 14.0);
            debug_out = float3(v, v * 0.7, v * 0.3);
        }
        else if (iDebugMode == 5) // Iapbp Hue Wheel (Graded Output, Perceptually Aligned)
        {
            float3 iapbp_dbg = Iapbp_FromXYZ(mul(ctx.to_XYZ, color), ctx.activeD65);
            float ch_sq = dot(iapbp_dbg.yz, iapbp_dbg.yz);
            if (ch_sq > 1e-12)
            {
                // Align Iapbp coordinate rotation (+36.5 deg) with standard hue wheel
                float raw_angle = atan2(iapbp_dbg.z, iapbp_dbg.y) + 0.637045;
                float hue = frac(raw_angle / (2.0 * PI) + 1.0);
                float br = saturate(SqrtIEEE(ch_sq) * 21.0);
                debug_out = HueToRGB(hue) * br;
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
            float3 xyz_s = mul(ctx.to_XYZ, original_lin);
            float skin = Evaluate3DSkinLocusLinear(xyz_s / max(whitePt, FLT_MIN), xyz_s.y / max(whitePt, FLT_MIN));
            debug_out = lerp(float3(0.0, 0.1, 0.3), float3(1.0, 0.2, 0.8), skin);
        }
        else if (iDebugMode == 8) // Gamut Compression Amount
        {
            float removed = 1.0 - g.dbg.y;
            debug_out = (g.dbg.z > 0.5) ? float3(1.0, 0.0, 1.0) : ((removed > 1e-4) ? Heatmap(0.25 + 1.5 * removed) : float3(0.0, 0.0, 0.0));
        }
        else if (iDebugMode == 9) // Chroma Correction (Tone + Path to White)
        {
            float f = g.dbg.x;
            float d = log2(max(f, 1e-6));
            debug_out = (d >= 0.0) ? float3(1.0, 0.5, 0.0) * saturate(d * 4.0) : float3(0.1, 0.4, 1.0) * saturate(-d * 2.0);
        }
        else if (iDebugMode == 10) // Reference Levels (BT.2408 / Display)
        {
            float Yd = dot(color, lumaCoeffs);
            float grey = BT2408_GREY_FRACTION * whitePt;
            float g_level = saturate(log2(max(Yd, 1e-4) / max(ctx.topNits, FLT_MIN)) / 14.0 + 1.0) * 0.6;
            debug_out = g_level.xxx;
            if (Yd < 1.0) debug_out = float3(0.0, 0.0, 0.5);
            if (abs(log2(max(Yd, 1e-6) / grey)) < 0.25) debug_out = float3(0.1, 0.8, 0.1);
            if (abs(log2(max(Yd, 1e-6) / whitePt)) < 0.25) debug_out = float3(1.0, 1.0, 1.0);
            else if (Yd > whitePt) debug_out = float3(0.9, 0.9, 0.2);
            if (!ctx.sdr && Yd > ctx.topNits * 1.0001) debug_out = float3(1.0, 0.0, 0.0);
        }
        else if (iDebugMode == 11) // Delta E ITP vs Input (BT.2124)
        {
            float3 a2020 = mul(XYZ_to_RGB2020, mul(ctx.to_XYZ, original_lin));
            float3 b2020 = mul(XYZ_to_RGB2020, mul(ctx.to_XYZ, color));
            float de = DeltaE_ITP(a2020, b2020);
            debug_out = (de < 0.05) ? float3(0.0, 0.0, 0.0) : Heatmap(0.25 * saturate(de) + 0.75 * saturate(de / 10.0));
        }

        fragColor = float4(EncodeDebug(debug_out, space), src.a);
        return;
    }
#endif

    // ---------------------------------------------------------------------------------------------
    // FINAL ENCODE & OUTPUT
    // ---------------------------------------------------------------------------------------------
    float3 encoded = EncodeFromLinear(color, space);

    // TPDF dither (+-1 code value) on quantized outputs; bypassed pixels never reach this point
    [branch]
    if (bDither && BUFFER_COLOR_BIT_DEPTH < 16 && space != 2)
    {
        encoded += TPDF(pos) / (exp2(float(BUFFER_COLOR_BIT_DEPTH)) - 1.0);
    }
    // scRGB: dither of +-1 code value of the downstream 10-bit PQ link (Rec.2020), same noise per channel
    [branch]
    if (bFloatDither && space == 2)
    {
        float3 rgb2020 = mul(XYZ_to_RGB2020, mul(RGB709_to_XYZ, color));
        if (all(rgb2020 > 0.0) && Max3(rgb2020) < PQ_PEAK_LUMINANCE)
        {
            float  n  = TPDF(pos) / 1023.0;
            float3 pq = PQ_InverseEOTF(rgb2020) + n;
            float3 d2020 = PQ_EOTF(saturate(pq));
            encoded = mul(XYZ_to_RGB709, mul(RGB2020_to_XYZ, d2020)) / SCRGB_WHITE_NITS;
        }
    }

    [flatten]
    if (space <= 1)
    {
        encoded = saturate(encoded);
    }

    fragColor = float4(encoded, src.a);
}

// -------------------------------------------------------------------------------------------------
// Deband (separate technique, place FIRST): quantization-aware, in source code values.
// Four iterations of 8 samples on a randomly rotated circle of radius R x i (i = 1..4, random
// 50-100%). An iteration's average replaces the pixel only if it is within Threshold / i code
// values of the current value AND every sample is within 2 x Threshold of the original (so edges
// and texture are kept); iterations touching the image border are skipped (no clamp bias).
// Pixels outside the SDR range of the source are not touched; untouched pixels stay bit-exact.
// -------------------------------------------------------------------------------------------------

int DebandBits()
{
    if (iDebandSourceBits == 1) return 10;
    if (iDebandSourceBits == 2) return 6;
    if (iDebandSourceBits == 3) return 5;
    return 8;
}

// Container value -> source code values (Rec.709, piecewise sRGB as the source was decoded)
float3 DebandToCodes(float3 encoded, int space, float max_code)
{
    if (space <= 1) return encoded * max_code;
    float3 lin = SanitizeLinear(DecodeToLinear(encoded, space));
    if (space >= 3) lin = mul(XYZ_to_RGB709, mul(RGB2020_to_XYZ, lin));
    return sRGB_OETF(lin / PHDR_REF_WHITE_NITS) * max_code;
}

float3 DebandFromCodes(float3 codes, int space, float max_code)
{
    if (space <= 1) return codes / max_code;
    float3 lin = sRGB_EOTF(codes / max_code) * PHDR_REF_WHITE_NITS;
    if (space >= 3) lin = mul(XYZ_to_RGB2020, mul(RGB709_to_XYZ, lin));
    return EncodeFromLinear(lin, space);
}

void PS_Deband(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 fragColor : SV_Target)
{
    int2 pos   = int2(vpos.xy);
    float4 src = tex2Dfetch(SamplerBackBuffer, pos);
    fragColor  = src;

    int   space    = ActiveSpace();
    float max_code = exp2(float(DebandBits())) - 1.0;
    float3 center  = DebandToCodes(src.rgb, space, max_code);
    if (any(center < -0.5) || any(center > max_code + 0.5)) return;

    int2  dims = int2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float T    = fDebandThreshold;
    float3 res = center;
    bool changed = false;

    [loop]
    for (int it = 1; it <= 4; it++)
    {
        float u = Hash12(float2(pos) + 0.5 + float2(17.0, 31.0) * float(it));
        float v = Hash12(float2(pos) + 0.5 + float2(53.0, 7.0) * float(it));
        float r = fDebandRadius * float(it) * (0.5 + 0.5 * u);
        float th0 = 2.0 * PI * v;

        float3 sum = float3(0.0, 0.0, 0.0);
        bool ok = true;
        [loop]
        for (int j = 0; j < 8; j++)
        {
            float th = th0 + (2.0 * PI / 8.0) * float(j);
            int2 q = pos + int2(round(r * cos(th)), round(r * sin(th)));
            if (any(q < 0) || any(q >= dims)) { ok = false; break; }
            float3 s = DebandToCodes(tex2Dfetch(SamplerBackBuffer, q).rgb, space, max_code);
            if (any(abs(s - center) > 2.0 * T)) { ok = false; break; }
            sum += s;
        }
        if (!ok) continue;
        float3 avg = sum / 8.0;
        if (all(abs(avg - res) <= T / float(it)))
        {
            res = avg;
            changed = true;
        }
    }

    if (!changed) return;

    float3 encoded = DebandFromCodes(res, space, max_code);
    if (BUFFER_COLOR_BIT_DEPTH < 16 && space != 2)
    {
        encoded += TPDF(pos) / (exp2(float(BUFFER_COLOR_BIT_DEPTH)) - 1.0);
    }
    if (space <= 1) encoded = saturate(encoded);
    fragColor = float4(encoded, src.a);
}

// =================================================================================================
// 15. Techniques
// =================================================================================================

technique PhotorealHDR_SceneGrade <
    ui_label = "Photoreal HDR Scene Grader V9.2.2 (Linear & Iapbp Edition)";
    ui_tooltip = "Reference scene grading in linear light and the Iapbp color space.\n\n"
                 "V9.2.2: Brent's method for gamut searches (1.2-1.46x faster in HDR), SDR white-band fix.\n"
                 "V9.2.1: cusp table fix (Lightness Focus / Path to White aimed correctly).\n"
                 "V9.2.0: 2.5-10x fewer gamut-boundary tests per pixel than V9.1 (V8-level cost),\n"
                 "        same output (< 0.01 Delta E ITP).\n"
                 "V9.1.0: BT.2390 EETF prediction removed (self-contained grader).\n"
                 "V9.0.2: cusp table cached, unneeded passes skipped, per-frame constants hoisted;\n"
                 "        debug views compiled out in ReShade performance mode. Output unchanged.\n"
                 "V9.0.1: ReShade FX compile fix (no nested structs).\n"
                 "V9.0.0:\n"
                 "  - Tone-dependent chroma correction (Mantiuk 2009) and path to white (display peak).\n"
                 "  - Bounded contrast; BT.2408 pivot.\n"
                 "  - ACES 2.0-style gamut mapping: per-hue cusp table, focus lightness, reach option.\n"
                 "  - Display primaries target (P3 default), SDR gamma 2.2 decode option.\n"
                 "  - Stable SA-PQ / PQ, two-step CAT16 with adaptation degree.\n"
                 "  - QA views, split view, scRGB PQ-step dither.\n"
                 "Shared settings: global preprocessor definitions PHDR_* (see the file header).\n"
                 "Factory defaults are bit-transparent.";
>
{
    pass CuspLUT
    {
        VertexShader      = VS_CuspLUT;
        PixelShader       = PS_CuspLUT;
        RenderTarget      = PHDR_CuspTex;
    }
    pass CuspKey
    {
        VertexShader      = PostProcessVS;
        PixelShader       = PS_CuspKey;
        RenderTarget      = PHDR_CuspKeyTex;
    }
    pass SceneGrade
    {
        VertexShader      = VS_PhotorealHDR;
        PixelShader       = PS_PhotorealHDR;
        VertexCount       = 3;
        PrimitiveTopology = TRIANGLELIST;
    }
}

technique PhotorealHDR_Deband <
    ui_label = "Photoreal HDR Deband (place FIRST)";
    ui_tooltip = "Quantization-aware debanding of SDR-origin content, in source code values.\n"
                 "Place it before BilateralContrast and the scene grader. Settings: category 6 of\n"
                 "Photoreal HDR. Edges, texture and HDR highlights are not touched.";
>
{
    pass Deband
    {
        VertexShader      = PostProcessVS;
        PixelShader       = PS_Deband;
    }
}
