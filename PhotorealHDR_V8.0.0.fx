// =================================================================================================
// Photoreal HDR Scene Grader (V8.0.0 - Reference Laboratory Edition)
// =================================================================================================
//
// Design Philosophy: REFERENCE SCIENTIFIC PRECISION OVER APPROXIMATION
// - True IEEE 754 Math: Guarded linear paths, NaN healing, and scale-invariant solvers.
// - Exact Container-Matched Tristimulus D65: Dynamically derived from active color matrix to eliminate
//   the 0.021% blue shift on Rec.2020 / P3 diffuse white points.
// - Exact Achromatic Axis: Iapbp matrices re-fitted to satisfy the paper's own white-point
//   constraints (Eq. 18) exactly, so every neutral maps to ap = bp = 0 analytically.
// - True Stop-Domain Scene Grading: Linear EV exposure, CIECAM16 (CAT16:2022), filmic contrast.
// - White Balance in Kelvin + Duv: Planckian locus (Krystek 1985) with a Duv offset, adapted to the
//   reference white with a CAT16 von Kries transform.
// - Projection-Based Iapbp Space: Advanced perceptual color space with SA-PQ non-linear transfer
//   (Optics Express Vol. 32, Issue 17, pp. 30742–30755, August 2024), fed absolute luminance
//   normalized to 10,000 cd/m2 like PQ (the range SA-PQ was fitted on).
// - ITU-R BT.2100 HLG with the reference OOTF (system gamma 1.2 at 1000 cd/m2 nominal peak).
// - Hue-Preserving SDR Ceiling: no per-channel clipping above SDR white on any grading path.
// - Source-Relative Vibrance: demand measured against the content's own gamut (Rec.709 for
//   SDR-origin games); bounded expansion that can never cross the target gamut boundary.
// - ACES-RGC-Style Gamut Compression: smooth power-curve roll-off toward the boundary that adapts
//   to the requested expansion (identity when nothing is expanded).
// - Highlight Recovery Shoulder: stop-domain soft knee with a smoothstep (C2) blend.
// - Enforced Bit-Exact Neutrality: Master Saturation = 0.00 forces exact R=G=B monochrome collapse.
// - 24-Step Scale-Invariant Gamut Solver: Dynamic binary search ray-tracer (delta t < 1.79e-7).
// - Calibrated Perceptual Hue Wheel: Angle-aligned diagnostic visualization.
// - TPDF Dither on graded 8/10-bit output; untouched pixels stay bit-exact.
// - Gamut Boundary Targets: Auto, Rec.709, Display P3, Rec.2020, and Bypass (Unclamped).
// - 3D Melanin-Hemoglobin skin locus protecting Fitzpatrick I-VI.
//
// References:
// - Y. Huang et al., "Towards perceptual uniformity and HDR-WCG image processing: a projection-based
//   color space," Optics Express 32(17), 30742–30755 (2024). https://doi.org/10.1364/OE.530213
// - C. Li et al., "Comprehensive color appearance model for full range of dynamic vision: CAM16,"
//   Color Research & Application 42(6), 703–718 (2017).
// - CIE 15:2004, "Colorimetry," 3rd Edition, Commission Internationale de l'Eclairage.
// - ITU-R BT.2100-2, "Image parameter values for high dynamic range television" (HLG OOTF).
// - ITU-R BT.2408, "Guidance for operational practices in HDR television production" (203 cd/m2).
// - ReShade Shaders Reference: https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md
//
// V8.0.0 changes (since V7.9.0):
// - Fixed: gamut boundary search returned boundaries 100-3750x too far out for dark colors (mostly
//   below ~5 cd/m2) and some hues. Beyond the real boundary the SA-PQ inverse clamps negative
//   L'M'S' to zero, the color decodes to black (RGB 0,0,0) and passed the inside test. Vibrance then
//   treated those colors as nearly neutral and expanded them far outside Rec.2020, and the AP0 cap
//   (same search) failed too: the out-of-Rec.2020 / "invalid" pixels and negative luminance seen in
//   analysis. The inside test now also rejects any sample where the inverse would clamp (L'M'S' at
//   or below SA-PQ black or at its asymptote), a non-positive projective denominator, or
//   non-positive luminance. The search bisects in log2(chroma), so its relative precision is the
//   same at every brightness (1.3e-6). On 2160 rays (10 brightness levels x 72 hues) it matches a
//   200k-point scan to the scan's resolution for Rec.709, P3 and Rec.2020; the old search was wrong
//   on 22-27% of rays.
// - Fixed: Bypass mode could still output negative luminance (AP0's blue primary lies below y = 0).
//   The physical limit now also requires positive luminance, and its search first marches outward
//   to bracket the first exit (the AP0 inside set is not always one interval far from the axis).
// - New: White Balance in Kelvin (2000-12000 K) + Duv. The chosen white is rendered as the
//   reference white (6504 K, Duv +0.0032 = D65 within 7e-5 in xy) with a CAT16 von Kries transform;
//   higher Kelvin = warmer image, positive Duv = more magenta (raw-converter convention). Neutral
//   (D65) luminance is preserved. Replaces the old arbitrary Temperature / Tint cone gains.
// - New: HLG Display Peak (400-2000 cd/m2): BT.2100 system gamma 1.2 + 0.42 log10(Lw / 1000).
//
// V7.9.0 changes (since V7.8.0):
// - Fixed: Smart Saturation (Vibrance) measured purity against the guard boundary (Rec.2020 in HDR),
//   where fully saturated Rec.709 colors read as pastels (0.40-0.95 of Rec.2020), so it boosted them
//   1.2x and pushed colors above 0.925 of the boundary past it (up to 1.06x; unbounded with the guard
//   bypassed). Vibrance now (a) takes its demand from purity relative to a selectable source gamut
//   (default Rec.709) and (b) expands purity p (relative to the target boundary) with
//   p' = 1 - (1 - p)^(1 + V * demand), which stays inside the boundary for any V, even in Bypass.
//   Measured on Rec.709 content in scRGB at Vibrance 2.0: pastels 2.43x, 709-saturated 1.01x,
//   max Rec.2020-relative purity 0.95.
// - Fixed: Saturation > 1 hard-clipped at the boundary with the default knee of 0 (at Saturation 2.0
//   every color from 0.54 to 1.0 of the boundary collapsed onto it). The gamut guard now uses the
//   ACES Reference Gamut Compression power curve (power 1.2) on radial purity, with threshold
//   1 - Knee and a limit equal to the requested expansion, so it is an exact identity when nothing
//   is expanded. At Saturation 2.0 / Knee 0.20, 2.5% of colors end within 0.1% of the boundary.
//   Default Knee 0.00 -> 0.20. The ad-hoc "diminishing returns" curve is removed.
// - New: Highlight Recovery shoulder (Amount, Pivot in stops vs Reference White, smoothstep Blend
//   Width): slope 1 - 0.9 * Amount * smoothstep(), integrated in closed form, monotone and C2;
//   applied as a luminance ratio (chroma preserved).
// - New: Vibrance Reference Gamut (Rec.709 / Display P3 / Rec.2020).
// - Changed: Knee and Abney alone no longer disable the bit-exact bypass / luminance-only path
//   (neither changes anything without a saturation or vibrance change).
// - Fixed: Bypass / Unclamped could push colors past the valid domain of the Iapbp inverse
//   (Saturation 2.0 produced values around -1e10). Bypass now keeps colors physically realizable
//   (inside ACES AP0, which encloses the spectral locus); it is otherwise unlimited.
//
// V7.8.0 changes (since V7.7.0):
// - Fixed: Abney Hue Compensation rotated hues even at Saturation 1.00 (up to ~8.6 deg on green /
//   purple). The rotation now follows the purity CHANGE, so it is zero when nothing is boosted.
// - Fixed: in SDR the gamut solver only tested negative primaries; boosted bright colors exceeded
//   white and were clipped per channel (bright sky at Saturation 1.50: -5.9 deg hue, -2.5% luminance).
//   The solver now also bounds the container channels at SDR white, and a final hue-preserving
//   ceiling covers every path (exposure, contrast, white balance).
// - Fixed: NaN pixels became 18% grey. NaN -> 0, +/-Inf -> +/-10000 cd/m2 per channel.
// - Fixed: bit-exact passthrough no longer depends on Color Space Override; the contrast stage is
//   skipped when it is an identity; luminance-only grades (exposure, dehaze, contrast, shadows /
//   highlights) of in-gamut pixels are applied directly to RGB with no XYZ round trip, and exactly
//   neutral pixels (R = G = B) stay exactly neutral.
// - Fixed: shadows / highlights blend uses smoothstep (C1); the old linear ramp had a 5% slope kink.
// - Changed: Iapbp input normalized to 10,000 cd/m2 (was: paper white = 1, which put 203 nits at the
//   top of SA-PQ and HDR highlights beyond its fitted 0-1 range). Saturation strength per slider
//   value changes slightly; re-check presets.
// - Changed: constraint-exact Iapbp matrices. The published 6-decimal M1 / M2 (Eqs. 20-21) miss the
//   paper's Eq. (18) row-sum constraints by up to 2.0e-3 (rounding explains at most 1.5e-6), which
//   left D65 white at ap / bp = 0.0012 / 0.0010. Rows are minimally rescaled / shifted (<= 0.14%) so
//   Eq. (18) holds exactly; the neutral-anchor solver is no longer needed. Coordinates move by at
//   most 4e-4 (I) and 3.4e-3 (ap, bp) over the BT.2020 cube.
// - Changed: HLG decode / encode use the BT.2100 OOTF, so HLG nits are display nits (75% HLG signal
//   = 203 cd/m2, matching BT.2408).
// - New: chroma below the float32 noise floor of the SA-PQ transform (< 5e-4, fading in to 2e-3) is
//   not boosted, so saturation cannot amplify numerical noise into near-neutral highlights.
// - New: TPDF dither (+-1 code value) on graded 8/10-bit output.
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

#ifndef BUFFER_COLOR_BIT_DEPTH
#define BUFFER_COLOR_BIT_DEPTH 8
#endif

static const float FLT_MIN              = 1.175494351e-38;
static const float PROJ_EPS             = 1e-7;
static const float SCRGB_WHITE_NITS     = 80.0;
static const float NEUTRAL_EPS          = 1e-6;
static const float PI                   = 3.14159265358979323846;

// Iapbp absolute luminance normalization (PQ convention: 1.0 = 10,000 cd/m2)
static const float IAPBP_ABS_NORM       = 10000.0;

// float32 noise floor of the Iapbp forward transform (SA-PQ exponent n' = 78.2 amplifies rounding):
// measured <= 1e-5 below 1000 cd/m2, up to 4.6e-4 at 4000-9000 cd/m2. Chroma below LO is not boosted.
static const float CHROMA_NOISE_LO      = 5e-4;
static const float CHROMA_NOISE_HI      = 2e-3;

// SA-PQ inverse validity: L'M'S' at or below c1^n decodes to black (clamped), at or above
// K_MAX^n hits the asymptote clamp. Samples there are outside any gamut.
static const float SA_PQ_BLACK          = 1.4206949e-9 * 1.001;
static const float SA_PQ_V_MAX          = 1.4970863;

// Gamut boundary search: log2-domain bisection from this chroma; outward march for AP0
static const float BOUNDARY_T_MIN       = 1e-9;
static const int   BOUNDARY_MARCH_STEPS = 48;

// White balance reference (D65: 6504 K, Duv +0.0032 on the Krystek Planckian locus)
static const float WB_REF_KELVIN        = 6504.0;
static const float WB_REF_DUV           = 0.0032;

// ACES Reference Gamut Compression power (ACES 1.3 default)
static const float RGC_POWER            = 1.2;

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

// -------------------------------------------------------------------------------------------------
// CIE CIECAM16:2022 Chromatic Adaptation Matrices
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
                  "Monotone and smooth; applied as a luminance ratio, so hue and saturation are kept.\n"
                  "0 = off (bit-exact).";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fRecoveryPivot <
    ui_type     = "slider";
    ui_min      = -4.00; ui_max = 6.00; ui_step = 0.01;
    ui_label    = "Recovery Pivot (Stops vs White)";
    ui_tooltip  = "Center of the shoulder, in stops relative to Reference White (0 = paper white).\n"
                  "Half of the compression is reached here. SDR content tops out at 0; HDR highlights\n"
                  "sit above it (+2 = 4x paper white, about 812 nits at 203).";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fRecoveryBlend <
    ui_type     = "slider";
    ui_min      = 0.10; ui_max = 6.00; ui_step = 0.01;
    ui_label    = "Recovery Blend Width (Stops)";
    ui_tooltip  = "Width of the smoothstep transition into the shoulder. Small = firm knee,\n"
                  "large = gradual roll-off starting well below the pivot.";
    ui_category = "1. Scene Grade";
> = 1.00;

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
    ui_tooltip  = "Boosts muted colors and leaves already-saturated ones alone.\n"
                  "Demand is measured against the Vibrance Reference Gamut (the content's own gamut),\n"
                  "and the expansion is bounded by the gamut guard boundary: it never crosses it,\n"
                  "even with the guard bypassed. Negative values mute pastels first.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform int iVibranceReference <
    ui_type     = "combo";
    ui_label    = "Vibrance Reference Gamut";
    ui_items    = "Rec. 709 (SDR-origin content)\0"
                  "Display P3 (D65)\0"
                  "Rec. 2020 (native WCG content)\0";
    ui_tooltip  = "Gamut the content was made in. A color at this gamut's edge counts as fully saturated\n"
                  "and gets no vibrance boost. Use Rec. 709 for SDR games (also in HDR / scRGB output).";
    ui_category = "1. Scene Grade";
> = 0;

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
    ui_tooltip  = "Counteracts perceived hue shifts caused by CHANGING saturation (Abney effect).\n"
                  "The rotation follows the purity change, so it has no effect at Saturation 1.00 / Vibrance 0.\n"
                  "The hue profile is a heuristic model, not fitted data.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform int iGamutTarget <
    ui_type     = "combo";
    ui_label    = "Gamut Guard Target Limit";
    ui_items    = "Auto (Container Gamut)\0"
                  "Rec. 709 (SDR Standard)\0"
                  "Display P3 (D65)\0"
                  "Rec. 2020 (UHD Display)\0"
                  "Bypass / Unclamped (Physical Limit, AP0)\0";
    ui_tooltip  = "Selects physical gamut boundary to soft-compress/clamp against.\n"
                  "- Auto: Matches container (Rec.709 in SDR, Rec.2020 in HDR).\n"
                  "- Bypass / Unclamped: no display-gamut limit; colors are only kept physically realizable\n"
                  "  (inside ACES AP0). Vibrance stays inside the container gamut even here.\n"
                  "In SDR the container channels are also kept at or below white without per-channel\n"
                  "clipping (hue and luminance preserved), except in Bypass mode.";
    ui_category = "1. Scene Grade";
> = 0;

uniform float fGamutGuardKnee <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label    = "Gamut Guard Knee";
    ui_tooltip  = "Width of the soft compression zone below the gamut boundary (fraction of the boundary).\n"
                  "ACES Reference Gamut Compression power curve: colors below 1 - Knee are untouched,\n"
                  "colors pushed beyond are rolled off smoothly instead of clipped.\n"
                  "Only acts when colors are expanded (Saturation > 1) or arrive out of gamut. 0 = hard clip.";
    ui_category = "1. Scene Grade";
> = 0.20;

uniform int iColorSpaceOverride <
    ui_type     = "combo";
    ui_label    = "Color Space Override";
    ui_items    = "Auto (Default via ReShade)\0"
                  "sRGB (SDR)\0"
                  "scRGB (HDR Linear)\0"
                  "HDR10 (PQ)\0"
                  "HLG (HDR)\0";
    ui_tooltip  = "Container format detection override.";
    ui_category = "2. System";
> = 0;

uniform float fWhitePoint <
    ui_type     = "slider";
    ui_min      = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_label    = "Reference White (Nits)";
    ui_tooltip  = "Diffuse paper white anchor for HDR mapping (default 203 nits ITU-R BT.2408).\nIgnored for standard SDR sRGB (80 nits, IEC 61966-2-1).\n"
                  "Sets the zones, contrast pivot, dehaze and skin gates. The Iapbp color math always uses\n"
                  "absolute luminance (nits / 10,000), independent of this setting.";
    ui_category = "2. System";
> = 203.0;

uniform float fHLGPeak <
    ui_type     = "slider";
    ui_min      = 400.0; ui_max = 2000.0; ui_step = 1.0;
    ui_label    = "HLG Display Peak (Nits)";
    ui_tooltip  = "Nominal peak luminance Lw of the HLG display (HLG output only).\n"
                  "Sets the BT.2100 system gamma 1.2 + 0.42 log10(Lw / 1000) (defined for 400-2000).\n"
                  "1000 = the BT.2100 / BT.2408 reference (75% signal = 203 nits).";
    ui_category = "2. System";
> = 1000.0;

uniform bool bDither <
    ui_label    = "Dither Graded Output";
    ui_tooltip  = "Adds triangular-PDF dither (+-1 code value) to graded pixels on 8/10-bit outputs\n"
                  "(sRGB, HDR10, HLG), so grading does not create new banding. No effect on scRGB,\n"
                  "and bypassed pixels stay bit-exact.";
    ui_category = "2. System";
> = true;

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

// Triangular-PDF noise in (-1, 1)
float TPDF(int2 pos)
{
    float2 p = float2(pos) + 0.5;
    return Hash12(p) + Hash12(p + float2(127.31, 311.77)) - 1.0;
}

// =================================================================================================
// 5. Projective Transformations & SA-PQ Non-linear Transfer
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
    float3 den = max(SA_PQ_C2 - SA_PQ_C3 * k, PROJ_EPS);
    return PowNonNegPreserveZero3(num / den, SA_PQ_INV_M);
}

// With the constraint-exact M1 / M2 the achromatic locus is ap = bp = 0 at every lightness, so no
// per-pixel neutral anchor is required (V7.7.0 solved one with ~12 pow() per pixel).

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
    float3 den = max(PQ_C2 - PQ_C3 * Np, PROJ_EPS);
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

// ITU-R BT.2100 HLG reference display: nominal peak Lw (UI), black 0,
// system gamma 1.2 + 0.42 log10(Lw / 1000) (BT.2100 Table 5, valid for Lw 400-2000 cd/m2)
float HLG_Lw()          { return clamp(fHLGPeak, 400.0, 2000.0); }
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

// Hue-preserving fit of a container-RGB color (cd/m2) into [0, white] for SDR output. Negative
// channels are first lifted toward the luminance axis, then channels above white are brought down
// toward it. Luminance is preserved unless it is itself outside [0, white].
float3 FitSDRContainer(float3 rgb, float white, float3 luma_coeffs)
{
    float L = dot(rgb, luma_coeffs);
    if (L <= 0.0)   return float3(0.0, 0.0, 0.0);
    if (L >= white) return white.xxx;

    float mn = min(min(rgb.r, rgb.g), rgb.b);
    if (mn < 0.0) rgb = L + (L / (L - mn)) * (rgb - L);

    float mx = max(max(rgb.r, rgb.g), rgb.b);
    if (mx > white) rgb = L + ((white - L) / (mx - L)) * (rgb - L);

    return rgb;
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
 * IapbpRayOutside: is the color at chroma t along a hue ray (lightness Ia) outside the gamut?
 * Robust to the clamps of the inverse transform: a sample whose L'M'S' would be clamped by the
 * SA-PQ inverse (at or below black, or at the asymptote), whose projective denominators are not
 * positive, or whose luminance is not positive counts as outside. Without these tests, colors far
 * beyond the boundary decode to black (RGB 0,0,0) and look "inside".
 */
bool IapbpRayOutside(
    float Ia,
    float t,
    float2 chroma_dir,
    float3x3 XYZ_to_targetRGB,
    float3x3 XYZ_to_containerRGB,
    float3 activeD65,
    bool check_upper,
    float upper_norm)
{
    float4 h2 = mul(M2_Iapbp_to_LMSprime, float4(Ia, t * chroma_dir.x, t * chroma_dir.y, 1.0));
    if (!(h2.w > 0.0)) return true;
    float3 lms_p = h2.xyz / h2.w;
    if (any(lms_p <= SA_PQ_BLACK) || any(lms_p >= SA_PQ_V_MAX)) return true;

    float3 lms = SA_PQ_Inverse(lms_p);
    float4 h1 = mul(M1_LMS_to_XYZ, float4(lms, 1.0));
    if (!(h1.w > 0.0)) return true;
    float3 norm_xyz = (h1.xyz / h1.w) * activeD65;
    if (!(norm_xyz.y > 0.0)) return true;

    float3 rgb = mul(XYZ_to_targetRGB, norm_xyz);
    float min_rgb = min(min(rgb.r, rgb.g), rgb.b);
    if (!(min_rgb >= 0.0)) return true;

    if (check_upper)
    {
        float3 rgb_c = mul(XYZ_to_containerRGB, norm_xyz);
        if (max(max(rgb_c.r, rgb_c.g), rgb_c.b) > upper_norm) return true;
    }
    return false;
}

/**
 * SolveGamutBoundaryIapbp (Scale-Invariant Primary Non-Negativity + SDR Ceiling Solver)
 * Largest chroma along a hue ray from the achromatic axis (ap = bp = 0) that stays inside the gamut:
 * all boundary-gamut primaries >= 0 and, when check_upper, every CONTAINER channel <= upper_norm
 * (SDR white in nits / 10000). 24-step bisection in log2(chroma) between 1e-9 and the projective
 * domain limit: relative precision 1.3e-6 at any brightness. march_first brackets the FIRST exit
 * with an outward geometric march before bisecting (used for AP0, whose inside set is not always
 * a single interval far from the axis). The ceiling test is dropped if the neutral at this
 * lightness is already above it.
 */
float SolveGamutBoundaryIapbp(
    float Ia,
    float2 chroma_dir,
    float3x3 XYZ_to_targetRGB,
    float3x3 XYZ_to_containerRGB,
    float3 activeD65,
    bool check_upper,
    float upper_norm,
    bool march_first)
{
    // Projective denominator of the inverse M2 along the ray (ap = bp = 0 at t = 0)
    float W0  = M2_Iapbp_to_LMSprime[3][0] * Ia + M2_Iapbp_to_LMSprime[3][3];
    float D_W = M2_Iapbp_to_LMSprime[3][1] * chroma_dir.x + M2_Iapbp_to_LMSprime[3][2] * chroma_dir.y;
    float t_high = (D_W < -1e-5) ? min(3.0, -W0 / D_W * 0.85) : 3.0;

    if (check_upper)
    {
        float3 n_lms = SA_PQ_Inverse(ProjectiveTransform(M2_Iapbp_to_LMSprime, float3(Ia, 0.0, 0.0)));
        float3 n_xyz = ProjectiveTransform(M1_LMS_to_XYZ, n_lms) * activeD65;
        float3 n_rgb = mul(XYZ_to_containerRGB, n_xyz);
        check_upper = (max(max(n_rgb.r, n_rgb.g), n_rgb.b) <= upper_norm);
    }

    if (IapbpRayOutside(Ia, BOUNDARY_T_MIN, chroma_dir, XYZ_to_targetRGB, XYZ_to_containerRGB, activeD65, check_upper, upper_norm))
        return 0.0;
    if (!IapbpRayOutside(Ia, t_high, chroma_dir, XYZ_to_targetRGB, XYZ_to_containerRGB, activeD65, check_upper, upper_norm))
        return t_high;

    float l_lo = log2(BOUNDARY_T_MIN);
    float l_hi = log2(t_high);

    [branch]
    if (march_first)
    {
        float l_start = l_lo;
        float l_step  = (l_hi - l_lo) / float(BOUNDARY_MARCH_STEPS);
        [loop]
        for (int k = 1; k <= BOUNDARY_MARCH_STEPS; k++)
        {
            float lk = l_start + l_step * float(k);
            if (IapbpRayOutside(Ia, exp2(lk), chroma_dir, XYZ_to_targetRGB, XYZ_to_containerRGB, activeD65, check_upper, upper_norm))
            {
                l_hi = lk;
                break;
            }
            l_lo = lk;
        }
    }

    [loop]
    for (int iter = 0; iter < 24; iter++)
    {
        float lm = 0.5 * (l_lo + l_hi);
        if (IapbpRayOutside(Ia, exp2(lm), chroma_dir, XYZ_to_targetRGB, XYZ_to_containerRGB, activeD65, check_upper, upper_norm))
            l_hi = lm;
        else
            l_lo = lm;
    }

    return exp2(l_lo);
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
    float3 activeD65,
    bool check_upper,
    float upper_norm,
    float3x3 to_RGB_vibrance_ref,
    out float out_skin_confidence)
{
    float luma_nits = graded_lin_xyz.y;
    float luma_norm = luma_nits / max(whitePt, FLT_MIN);
    out_skin_confidence = Evaluate3DSkinLocusLinear(graded_lin_xyz / max(whitePt, FLT_MIN), luma_norm);
    
    bool is_unclamped = (gamut_target_mode == 4);
    
    // Bit-exact monochrome collapse (Master Saturation = 0.00)
    if (purity_scale <= NEUTRAL_EPS)
    {
        float3 rgb_direct = mul(to_TargetRGB, graded_lin_xyz);
        float gray = dot(rgb_direct, lumaCoeffs);
        return gray.xxx;
    }

    // Knee and Abney change nothing without a purity change
    bool chromaIdentity = 
        abs(purity_scale - 1.0) < NEUTRAL_EPS &&
        abs(vibrance_amount) < NEUTRAL_EPS;

    // Fast path: if chromaticity is untouched and pixel is in-gamut, bypass iterative solver
    [branch]
    if (chromaIdentity)
    {
        if (is_unclamped)
            return mul(to_TargetRGB, graded_lin_xyz);
            
        float3 rgb_b = mul(to_RGB_boundary, graded_lin_xyz);
        if (all(rgb_b >= 0.0))
            return mul(to_TargetRGB, graded_lin_xyz);
    }

    // White-chromaticity normalization (von Kries, Huang et al. Sec. 2.1) at absolute luminance:
    // reference white XYZ scaled to 10,000 cd/m2, the range SA-PQ is fitted on.
    float3 norm_xyz = (graded_lin_xyz / IAPBP_ABS_NORM) / activeD65;
    float3 lms = ProjectiveTransform(M1_XYZ_to_LMS, norm_xyz);
    float3 lms_p = SA_PQ_Forward(max(lms, 0.0));
    float3 iapbp = ProjectiveTransform(M2_LMSprime_to_Iapbp, lms_p);
    
    // Achromatic axis is exactly ap = bp = 0 (constraint-exact matrices)
    float2 chroma_offset = iapbp.yz;
    float chroma = SqrtIEEE(dot(chroma_offset, chroma_offset));
    
    // Chroma below the float32 noise floor of the transform is not boosted
    float chroma_gate = smoothstep(CHROMA_NOISE_LO, CHROMA_NOISE_HI, chroma);
    
    float ct = saturate((luma_norm - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);
    
    float2 chroma_dir = (chroma > FLT_MIN) ? (chroma_offset / chroma) : float2(1.0, 0.0);
    // Target-relative purity: 1.0 = guard boundary (plus the SDR white ceiling). May exceed 1 for
    // out-of-gamut input.
    float max_t = SolveGamutBoundaryIapbp(iapbp.x, chroma_dir, to_RGB_boundary, to_TargetRGB, activeD65, check_upper, upper_norm, false);
    float p0 = chroma / max(max_t, FLT_MIN);
    
    float effective_skin_mask = saturate(out_skin_confidence * skin_protection);
    
    // Smart Saturation (Vibrance)
    //   demand from purity relative to the SOURCE gamut (fully saturated there -> no boost);
    //   expansion p' = 1 - (1 - p)^(1 + V * w) relative to the TARGET boundary: bounded by 1 for any V.
    float p1 = p0;
    [branch]
    if (abs(vibrance_amount) > NEUTRAL_EPS && p0 < 1.0)
    {
        float max_s = SolveGamutBoundaryIapbp(iapbp.x, chroma_dir, to_RGB_vibrance_ref, to_TargetRGB, activeD65, false, 0.0, false);
        float ps = saturate(chroma / max(max_s, FLT_MIN));
        float demand = 1.0 - smoothstep(0.0, 1.0, ps);
        float w = demand * (1.0 - effective_skin_mask) * chroma_reliability * chroma_gate;
        float g = max(1.0 + vibrance_amount * w, 0.0);
        p1 = 1.0 - PowNonNegPreserveZero(1.0 - p0, g);
    }
    
    // Purity / Saturation: uniform radial scale (boosts are skin-protected, reliability- and
    // noise-floor-weighted; reductions are noise-floor-weighted)
    float S = purity_scale;
    if (S > 1.0)
    {
        float master_boost = (S - 1.0) * (1.0 - effective_skin_mask * 0.85);
        S = 1.0 + master_boost * chroma_reliability * chroma_gate;
    }
    else
    {
        S = lerp(1.0, S, chroma_gate);
    }
    float p2 = p1 * S;
    
    // Gamut Guard: ACES-RGC power-curve compression toward the boundary. The limit is the largest
    // purity this expansion can produce, so nothing is compressed when nothing is expanded.
    float p3 = p2;
    if (!is_unclamped)
    {
        float limit = max(max(S, 1.0), p2);
        p3 = CompressPurity(p2, 1.0 - knee, limit);
        if (p3 > 1.0 - 1e-5) p3 = 1.0 - 1e-5;
    }
    
    // Abney Hue Compensation: rotation proportional to the purity CHANGE (zero when unchanged)
    float2 out_dir = chroma_dir;
    float  out_max = max_t;
    if (abney_correction > NEUTRAL_EPS && chroma > FLT_MIN)
    {
        float angle = atan2(chroma_dir.y, chroma_dir.x);
        float abney_profile = 0.15 * sin(2.0 * angle + 0.4) * (1.0 + 0.3 * cos(angle));
        float purity_change = saturate(p3) - saturate(p0);
        angle += abney_profile * purity_change * abney_correction * chroma_reliability;
        out_dir = float2(cos(angle), sin(angle));
        
        // Keep the relative purity against the boundary of the new hue direction
        if (!is_unclamped)
            out_max = SolveGamutBoundaryIapbp(iapbp.x, out_dir, to_RGB_boundary, to_TargetRGB, activeD65, check_upper, upper_norm, false);
    }
    
    // Reconstruct Iapbp. Bypass mode is still limited to physically realizable colors (inside AP0):
    // beyond that the projective inverse leaves its valid domain (values up to ~1e10 were measured).
    float out_chroma = p3 * out_max;
    [branch]
    if (is_unclamped && out_chroma > FLT_MIN)
    {
        float max_phys = SolveGamutBoundaryIapbp(iapbp.x, out_dir, XYZ_to_AP0, to_TargetRGB, activeD65, false, 0.0, true);
        out_chroma = min(out_chroma, max_phys * (1.0 - 1e-5));
    }
    iapbp.yz = out_dir * out_chroma;
    
    // Invert back: Iapbp -> L'M'S' -> LMS -> Normalized XYZ -> Physical Linear XYZ (cd/m2)
    float3 out_lms_p = ProjectiveTransform(M2_Iapbp_to_LMSprime, iapbp);
    float3 out_lms = SA_PQ_Inverse(out_lms_p);
    float3 out_norm_xyz = ProjectiveTransform(M1_LMS_to_XYZ, out_lms) * activeD65;
    float3 out_xyz = out_norm_xyz * IAPBP_ABS_NORM;
    
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

// -------------------------------------------------------------------------------------------------
// White balance: Planckian locus (Krystek 1985 rational approximation, CIE 1960 uv, 1000-15000 K,
// accuracy ~8e-5 in uv) with a Duv offset along the locus normal (positive = above the locus).
// -------------------------------------------------------------------------------------------------
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

// =================================================================================================
// 9. Vertex Shader
// =================================================================================================

struct VS_Output
{
    float4 vpos : SV_Position;
    float2 texcoord : TEXCOORD0;
    nointerpolation float3 wbGainCAT16 : TEXCOORD1;
};

VS_Output VS_PhotorealHDR(uint id : SV_VertexID)
{
    VS_Output output;
    output.texcoord.x = (id == 2) ? 2.0 : 0.0;
    output.texcoord.y = (id == 1) ? 2.0 : 0.0;
    output.vpos = float4(output.texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    
    // CAT16 von Kries adaptation (complete, D = 1): the chosen white (Kelvin, Duv) is rendered as
    // the reference white (6504 K, Duv +0.0032 = D65). Relative to the reference point, so the
    // default settings are an exact identity.
    float3 srcWhite = WhitePointXYZ(clamp(fWhiteBalanceK, 1000.0, 15000.0), fWhiteBalanceDuv);
    float3 refWhite = WhitePointXYZ(WB_REF_KELVIN, WB_REF_DUV);
    float3 rawGains = mul(XYZ_to_CAT16, refWhite) / mul(XYZ_to_CAT16, srcWhite);
    
    // Preserve the luminance of neutral (D65) content
    const float3 standardD65 = float3(0.950456, 1.000000, 1.089058);
    float3 cat16D65 = mul(XYZ_to_CAT16, standardD65);
    float3 adaptedCAT16 = cat16D65 * rawGains;
    float3 adaptedXYZ = mul(CAT16_to_XYZ, adaptedCAT16);
    float lumaPreserveScale = 1.0 / max(adaptedXYZ.y, FLT_MIN);
    
    output.wbGainCAT16 = rawGains * lumaPreserveScale;
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
        abs(fExposure) < NEUTRAL_EPS && abs(fBlackPoint) < NEUTRAL_EPS &&
        abs(fContrast - 1.0) < NEUTRAL_EPS && abs(fShadows) < NEUTRAL_EPS &&
        abs(fHighlights) < NEUTRAL_EPS && !WhiteBalanceActive() &&
        abs(fSaturation - 1.0) < NEUTRAL_EPS &&
        abs(fVibrance) < NEUTRAL_EPS && fHighlightRecovery < NEUTRAL_EPS &&
        iGamutTarget == 0)
    {
        fragColor = src;
        return;
    }
    
    // Decode to linear nits & sanitize (NaN -> 0, +/-Inf -> +/-10000 cd/m2, per channel)
    float3 original_lin = SanitizeLinear(DecodeToLinear(src.rgb, space));
    
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

    // Exact Container-Matched Tristimulus D65
    float3 activeD65 = mul(to_XYZ, float3(1.0, 1.0, 1.0));
    
    // ---------------------------------------------------------------------------------------------
    // STAGE 1: LINEAR SCENE GRADE (Exposure, CAT16 White Balance)
    // ---------------------------------------------------------------------------------------------
    float3 lin_xyz = mul(to_XYZ, original_lin);
    
    // Physiological Von Kries adaptation in linear CAT16 cone space
    bool wb_active = WhiteBalanceActive();
    if (wb_active)
    {
        float3 cat16 = mul(XYZ_to_CAT16, lin_xyz);
        cat16 *= input.wbGainCAT16;
        lin_xyz = mul(CAT16_to_XYZ, cat16);
    }
    
    // Linear Stop-Domain Exposure
    float exposure_gain = 1.0;
    if (abs(fExposure) > NEUTRAL_EPS)
    {
        exposure_gain = exp2(fExposure);
        lin_xyz *= exposure_gain;
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
    
    // Skip the tone stage entirely when it is an identity (no exp2/log2 round-off)
    bool tone_identity = abs(fContrast - 1.0) < NEUTRAL_EPS &&
                         abs(fShadows) < NEUTRAL_EPS && abs(fHighlights) < NEUTRAL_EPS;
    
    [branch]
    if (!tone_identity && absLuma > FLT_MIN && luma > 0.0)
    {
        float pivot = fContrastPivot * whitePt;
        float logRatio = log2(absLuma / pivot);
        float x = logRatio * fContrast;
        float S = fShadows * 3.0;
        float H = fHighlights * 3.0;
        
        float rational_factor = (x * x) / (x * x + 6.0);
        float blend_t = smoothstep(-0.125, 0.125, x);    // C1-continuous shadow / highlight blend
        float recovery = lerp(S, H, blend_t);
        
        x += recovery * rational_factor;
        
        float contrastLuma = pivot * exp2(x);
        float ratio = contrastLuma / absLuma;
        
        float excess = max(ratio - 80.0, 0.0);
        contrast_ratio = min(ratio, 80.0) + (excess / (1.0 + excess / 20.0));
    }
    
    // Highlight Recovery shoulder: s' = s - A * integral(smoothstep), s = log2(L / white).
    // Slope 1 - A * smoothstep(u) >= 1 - A > 0 (monotone, C2). Luminance ratio only.
    float recovery_ratio = 1.0;
    [branch]
    if (fHighlightRecovery > NEUTRAL_EPS)
    {
        float L_tone = luma * bp_ratio * contrast_ratio;
        if (L_tone > FLT_MIN)
        {
            float A  = 0.9 * saturate(fHighlightRecovery);
            float w  = max(fRecoveryBlend, 0.01);
            float s  = log2(L_tone / whitePt);
            float u  = (s - (fRecoveryPivot - 0.5 * w)) / w;
            float I  = (u <= 0.0) ? 0.0
                     : (u < 1.0)  ? w * (u * u * u - 0.5 * u * u * u * u)
                     :              w * 0.5 + (u - 1.0) * w;
            recovery_ratio = exp2(-A * I);
        }
    }
    
    lin_xyz *= bp_ratio * contrast_ratio * recovery_ratio;
    
    // ---------------------------------------------------------------------------------------------
    // STAGE 3: PERCEPTUAL COLOR SPACE GRADING & GAMUT GUARD (Iapbp Space, Huang et al. 2024)
    // ---------------------------------------------------------------------------------------------
    bool  sdr_ceiling = (space <= 1) && (iGamutTarget != 4);
    
    float3x3 to_RGB_vibrance_ref = XYZ_to_RGB709;
    [flatten] if (iVibranceReference == 1) to_RGB_vibrance_ref = XYZ_to_P3D65;
    [flatten] if (iVibranceReference == 2) to_RGB_vibrance_ref = XYZ_to_RGB2020;
    float upper_norm  = whitePt / IAPBP_ABS_NORM;
    
    // Knee and Abney change nothing without a saturation / vibrance change
    bool chroma_identity = abs(fSaturation - 1.0) < NEUTRAL_EPS && abs(fVibrance) < NEUTRAL_EPS;
    bool rgb_neutral = (original_lin.r == original_lin.g) && (original_lin.g == original_lin.b);
    bool in_boundary = (iGamutTarget == 4) || all(mul(to_RGB_boundary, lin_xyz) >= 0.0);
    
    float3 color;
    float skin_confidence = 0.0;
    
    [branch]
    if (!wb_active && fSaturation > NEUTRAL_EPS && (rgb_neutral || (chroma_identity && in_boundary)))
    {
        // Luminance-only grade (exposure, dehaze, contrast, shadows / highlights) of an in-gamut or
        // exactly neutral pixel: one scalar ratio on container RGB, no XYZ / Iapbp round trip.
        color = original_lin * (exposure_gain * bp_ratio * contrast_ratio * recovery_ratio);
        
        if (iDebugMode == 7)
        {
            skin_confidence = Evaluate3DSkinLocusLinear(lin_xyz / max(whitePt, FLT_MIN), lin_xyz.y / max(whitePt, FLT_MIN));
        }
    }
    else
    {
        color = ApplyIapbpSaturationAndGamutGuard(
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
            activeD65,
            sdr_ceiling,
            upper_norm,
            to_RGB_vibrance_ref,
            skin_confidence
        );
    }
    
    bool is_invalid = any(IsNan3(color)) || any(IsInf3(color));
    color = is_invalid ? original_lin : color;
    
    // Hue-preserving SDR ceiling: no per-channel clipping above white on any grading path
    [branch]
    if (sdr_ceiling)
    {
        color = FitSDRContainer(color, whitePt, lumaCoeffs);
    }
    
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
        else if (iDebugMode == 3) // CAT16 Cone Response
        {
            float3 c16 = mul(XYZ_to_CAT16, mul(to_XYZ, color));
            float max_c16 = max(max(abs(c16.r), abs(c16.g)), abs(c16.b));
            if (max_c16 > FLT_MIN)
                debug_out = abs(c16) / max_c16;
        }
        else if (iDebugMode == 4) // Iapbp Chroma / Saturation (Graded Output)
        {
            float3 graded_xyz = mul(to_XYZ, color);
            float3 norm_xyz_dbg = (graded_xyz / IAPBP_ABS_NORM) / activeD65;
            float3 lms_p_dbg = SA_PQ_Forward(max(ProjectiveTransform(M1_XYZ_to_LMS, norm_xyz_dbg), 0.0));
            float3 iapbp_dbg = ProjectiveTransform(M2_LMSprime_to_Iapbp, lms_p_dbg);
            float2 c_off = iapbp_dbg.yz;    // achromatic axis is ap = bp = 0
            float ch = SqrtIEEE(dot(c_off, c_off));
            float v = saturate(ch * 14.0);    // absolute normalization gives ~3.5x smaller chroma than V7.7.0
            debug_out = float3(v, v * 0.7, v * 0.3);
        }
        else if (iDebugMode == 5) // Iapbp Hue Wheel (Graded Output, Perceptually Aligned)
        {
            float3 graded_xyz = mul(to_XYZ, color);
            float3 norm_xyz_dbg = (graded_xyz / IAPBP_ABS_NORM) / activeD65;
            float3 lms_p_dbg = SA_PQ_Forward(max(ProjectiveTransform(M1_XYZ_to_LMS, norm_xyz_dbg), 0.0));
            float3 iapbp_dbg = ProjectiveTransform(M2_LMSprime_to_Iapbp, lms_p_dbg);
            float2 c_off = iapbp_dbg.yz;    // achromatic axis is ap = bp = 0
            float ch_sq = dot(c_off, c_off);
            
            if (ch_sq > 1e-12)
            {
                // Align Iapbp coordinate rotation (+36.5 deg) with standard hue wheel
                float raw_angle = atan2(c_off.y, c_off.x) + 0.637045;
                float hue = frac(raw_angle / (2.0 * PI) + 1.0);
                float br = saturate(SqrtIEEE(ch_sq) * 21.0);    // rescaled for absolute normalization
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
            debug_out = lerp(float3(0.0, 0.1, 0.3), float3(1.0, 0.2, 0.8), skin_confidence);
        }
        
        fragColor = float4(EncodeDebug(debug_out, space), src.a);
        return;
    }
    
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
    ui_label = "Photoreal HDR Scene Grader V8.0.0 (Linear & Iapbp Edition)";
    ui_tooltip = "Reference scene grading in linear light and the Iapbp color space.\n\n"
                 "V8.0.0:\n"
                 "  - Fixed gamut boundary search for dark colors (Vibrance / AP0 escapes).\n"
                 "  - White balance in Kelvin + Duv (CAT16 von Kries).\n"
                 "  - HLG display peak (BT.2100 system gamma).\n\n"
                 "V7.9.0:\n"
                 "  - Vibrance: source-gamut demand, bounded expansion (never crosses the boundary).\n"
                 "  - Gamut guard: ACES-RGC power-curve compression adapted to the expansion.\n"
                 "  - Highlight Recovery shoulder with smoothstep blend.\n"
                 "  - Bypass / Unclamped limited to physically realizable colors (AP0).\n\n"
                 "V7.8.0:\n"
                 "  - Constraint-exact Iapbp matrices (paper Eq. 18): exact achromatic axis.\n"
                 "  - Iapbp fed absolute luminance / 10,000 cd/m2 (SA-PQ's fitted range).\n"
                 "  - BT.2100 HLG with OOTF; NaN -> 0, Inf -> 10,000 cd/m2.\n"
                 "  - SDR: gamut solver bounds channels at white; hue-preserving ceiling on all paths.\n"
                 "  - Abney compensation follows the purity change (none at Saturation 1.00).\n"
                 "  - C1 shadows / highlights blend; TPDF dither on 8/10-bit output.\n"
                 "  - Factory defaults are bit-transparent, with or without Color Space Override.";
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