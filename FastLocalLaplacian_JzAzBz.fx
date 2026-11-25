// =================================================================================================
// Fast Local Laplacian Filter (FLLF) in JzAzBz Perceptual Color Space for ReShade
// =================================================================================================
//
// References:
//   [1] M. Safdar, G. Cui, Y. J. Kim, and M. R. Luo,
//       "Perceptually uniform color space for image signals including high dynamic range and
//       wide gamut," Optics Express, vol. 25, no. 13, pp. 15131-15151, 2017.
//       https://opg.optica.org/oe/fulltext.cfm?uri=oe-25-13-15131
//   [2] M. Aubry, S. Paris, S. W. Hasinoff, J. Kautz, and F. Durand,
//       "Fast Local Laplacian Filters: Theory and Applications,"
//       ACM Transactions on Graphics (SIGGRAPH), vol. 33, no. 5, 2014.
//   [3] S. Paris, S. W. Hasinoff, and J. Kautz,
//       "Local Laplacian Filters: Edge-aware Image Processing with a Laplacian Pyramid,"
//       ACM Transactions on Graphics (SIGGRAPH), vol. 30, no. 4, 2011.
//   [4] P. J. Burt and E. H. Adelson,
//       "The Laplacian Pyramid as a Compact Image Code,"
//       IEEE Transactions on Communications, vol. 31, no. 4, pp. 532-540, 1983.
//   [5] ReShade FX Shading Language Reference:
//       https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md
//
// Architectural & Precision Highlights:
//   - Full 32-bit IEEE 754 floating-point pipeline (RGBA32F / R32F) across all working buffers,
//     Gaussian/Laplacian pyramids, and intensity slices — zero FP16 underflow of Safdar's
//     d0 = 1.6295499532821566e-11 zero-point offset and zero 10-bit/16-bit banding.
//   - Delta-Residual FLLF formulation (Delta_R = r(Jz; g_k) - Jz): mathematically identical to
//     Aubry et al. (2014) by linearity of Burt-Adelson pyramids, while avoiding base-image
//     subtraction cancellation and guaranteeing bit-exact identity (0.0 error) at neutral settings.
//   - Exact discrete Burt-Adelson 5-tap Reduce ([1,4,6,4,1]/16) and exact parity-aware Expand
//     ([1,6,1]/8 even, [1,1]/2 odd) using exact integer texel fetches (tex2Dfetch) and exact
//     binary-rational float32 weights — eliminates 8-bit GPU TMU bilinear quantization and
//     odd-resolution half-texel drift (e.g., 1080p, 1440p, ultrawide, arbitrary window sizes).
//   - Algebraically stabilized Safdar (2017) JzAzBz and SMPTE ST 2084 PQ non-linearities
//     (5th-order log1p/expm1 Taylor branches within |u|, w < 0.02 + cancellation-free
//     (1 - c1) + c3*(1 - vp) inverse denominator + 64-bit pre-folded RGB<->LMS matrices),
//     eliminating float32 cancellation near black and cutting near-peak HDR primary round-trip
//     error by >35x while maintaining <1.5e-9 Taylor truncation error at branch boundaries.
//   - Native SDR & HDR support via BUFFER_COLOR_SPACE (sRGB/Rec.709, scRGB wide-gamut FP16,
//     HDR10 ST2084 PQ / Rec.2020, HDR10 BT.2100 HLG / Rec.2020) with automatic per-frame HDR
//     peak ceiling detection so highlights >1000 nits are never compared against a clamped slice.
//   - Adaptive Constant-(Jz, h_z) Soft-Knee Gamut Mapping: unchanged in-gamut saturated colors
//     (including pure primaries) pass through intact with 0 LSB shift while newly out-of-gamut
//     excursions roll off smoothly.
//   - 4-slice-per-RGBA32F MRT packing (8 / 12 / 16 slices) with Catmull-Rom cubic slice
//     interpolation (>70 dB PSNR vs exact O(N^2) Local Laplacian Filtering at 16 slices).
//   - Approximate VRAM footprint (K=16, L=9, all float32): ~149 MB at 1080p / ~597 MB at 4K
//     (the separable H1_R intermediate is ~66 MB at 1080p / ~265 MB at 4K for 5x lower remap ALU;
//     setting FLLF_INTENSITY_SLICES=12 or 8 reduces slice pyramid VRAM proportionally).
// =================================================================================================

// -------------------------------------------------------------------------------------------------
// User-Configurable Preprocessor Definitions (exposed in ReShade's Preprocessor Definitions UI)
// -------------------------------------------------------------------------------------------------

// Number of discretized Jz intensity slices for Fast Local Laplacian interpolation.
// Valid values: 8 (2 MRTs, fast), 12 (3 MRTs, high precision), 16 (4 MRTs, maximum precision).
#ifndef FLLF_INTENSITY_SLICES
    #if __RENDERER__ == 0x9000
        #define FLLF_INTENSITY_SLICES 8
    #else
        #define FLLF_INTENSITY_SLICES 16
    #endif
#endif

// Number of pyramid levels (Level 0 through Level N-1).
// Valid values: 5..9. Default 9 reaches ~512px (1080p) / ~1024px (4K) spatial scale.
#ifndef FLLF_PYRAMID_LEVELS
    #define FLLF_PYRAMID_LEVELS 9
#endif

// -------------------------------------------------------------------------------------------------
// Internal Preprocessor Validation & Helpers (leading underscore hides them from ReShade UI)
// -------------------------------------------------------------------------------------------------

#if (FLLF_INTENSITY_SLICES <= 8)
    #define _FLLF_SLICES    8
    #define _FLLF_MRT_COUNT 2
#elif (FLLF_INTENSITY_SLICES <= 12)
    #define _FLLF_SLICES    12
    #define _FLLF_MRT_COUNT 3
#else
    #define _FLLF_SLICES    16
    #define _FLLF_MRT_COUNT 4
#endif

#if (FLLF_PYRAMID_LEVELS < 5)
    #define _FLLF_LEVELS 5
#elif (FLLF_PYRAMID_LEVELS > 9)
    #define _FLLF_LEVELS 9
#else
    #define _FLLF_LEVELS FLLF_PYRAMID_LEVELS
#endif

// Exact ceiling-halving pyramid dimensions (associative: ceil(ceil(n/2)/2) == ceil(n/4), always >= 1)
#define _FLLF_W0 BUFFER_WIDTH
#define _FLLF_H0 BUFFER_HEIGHT
#define _FLLF_W1 ((BUFFER_WIDTH  + 1)   / 2)
#define _FLLF_H1 ((BUFFER_HEIGHT + 1)   / 2)
#define _FLLF_W2 ((BUFFER_WIDTH  + 3)   / 4)
#define _FLLF_H2 ((BUFFER_HEIGHT + 3)   / 4)
#define _FLLF_W3 ((BUFFER_WIDTH  + 7)   / 8)
#define _FLLF_H3 ((BUFFER_HEIGHT + 7)   / 8)
#define _FLLF_W4 ((BUFFER_WIDTH  + 15)  / 16)
#define _FLLF_H4 ((BUFFER_HEIGHT + 15)  / 16)
#define _FLLF_W5 ((BUFFER_WIDTH  + 31)  / 32)
#define _FLLF_H5 ((BUFFER_HEIGHT + 31)  / 32)
#define _FLLF_W6 ((BUFFER_WIDTH  + 63)  / 64)
#define _FLLF_H6 ((BUFFER_HEIGHT + 63)  / 64)
#define _FLLF_W7 ((BUFFER_WIDTH  + 127) / 128)
#define _FLLF_H7 ((BUFFER_HEIGHT + 127) / 128)
#define _FLLF_W8 ((BUFFER_WIDTH  + 255) / 256)
#define _FLLF_H8 ((BUFFER_HEIGHT + 255) / 256)

// Exact 8x8 frame-peak max reduction pyramid dimensions (100% pixel coverage up to 8K)
#define _FLLF_PW1 ((BUFFER_WIDTH  + 7) / 8)
#define _FLLF_PH1 ((BUFFER_HEIGHT + 7) / 8)
#define _FLLF_PW2 ((_FLLF_PW1 + 7) / 8)
#define _FLLF_PH2 ((_FLLF_PH1 + 7) / 8)
#define _FLLF_PW3 ((_FLLF_PW2 + 7) / 8)
#define _FLLF_PH3 ((_FLLF_PH2 + 7) / 8)

// Exact un-filtered texel fetch across ReShade 4.x / 5.x / 6.x and D3D9..D3D12 / OpenGL / Vulkan
#if (__RENDERER__ == 0x9000)
    #define _FLLF_FETCH(s, p, sz) tex2Dlod(s, float4((float2(p) + 0.5f) / float2(sz), 0.0f, 0.0f))
    #define _FLLF_PIXEL_COORD(vpos, uv, sz) clamp(int2(floor((uv) * float2(sz))), int2(0, 0), (sz) - 1)
#elif defined(__RESHADE__) && (__RESHADE__ < 50000)
    #define _FLLF_FETCH(s, p, sz) tex2Dfetch(s, int4(p, 0, 0))
    #define _FLLF_PIXEL_COORD(vpos, uv, sz) clamp(int2((vpos).xy), int2(0, 0), (sz) - 1)
#else
    #define _FLLF_FETCH(s, p, sz) tex2Dfetch(s, int2(p), 0)
    #define _FLLF_PIXEL_COORD(vpos, uv, sz) clamp(int2((vpos).xy), int2(0, 0), (sz) - 1)
#endif

// -------------------------------------------------------------------------------------------------
// Uniform UI Parameters
// -------------------------------------------------------------------------------------------------

// --- Category 1: Local Laplacian Filter ---
uniform float DetailContrast <
    ui_category = "Local Laplacian Filter";
    ui_label = "Detail Contrast (Clarity / Smooth)";
    ui_tooltip = "Controls fine and medium local detail below the Edge Threshold (Sigma).\n"
                 "Positive values enhance micro-contrast and clarity without halos.\n"
                 "Negative values perform edge-preserving smoothing.\n"
                 "0.0 leaves details unchanged.";
    ui_type = "slider";
    ui_min = -0.95;
    ui_max = 2.00;
    ui_step = 0.01;
> = 0.35;

uniform float EdgeSigma <
    ui_category = "Local Laplacian Filter";
    ui_label = "Edge Threshold (Sigma)";
    ui_tooltip = "Perceptual lightness threshold separating local details from large edges.\n"
                 "Specified relative to reference white Jz so behavior is consistent across SDR and HDR.\n"
                 "Smaller values restrict enhancement to finer contrast transitions;\n"
                 "larger values include broader shading gradients.";
    ui_type = "slider";
    ui_min = 0.04;
    ui_max = 1.00;
    ui_step = 0.005;
> = 0.30;

uniform float EdgeToneScale <
    ui_category = "Local Laplacian Filter";
    ui_label = "Large-Scale Contrast (Beta)";
    ui_tooltip = "Controls large-scale edge amplitude above Sigma (classic LLF beta).\n"
                 "< 1.0 compresses large dynamic range differences (halo-free local tone mapping).\n"
                 "> 1.0 expands broad contrast across large edges.\n"
                 "1.0 preserves large-scale edges identically.";
    ui_type = "slider";
    ui_min = 0.15;
    ui_max = 2.00;
    ui_step = 0.01;
> = 1.00;

uniform float HighlightCompression <
    ui_category = "Local Laplacian Filter";
    ui_label = "Highlight Recovery / Compression";
    ui_tooltip = "Asymmetric large-scale remapping for bright regions above local surround.\n"
                 "Positive values compress bright highlights locally without dark halos (1.0 = 0.5x, 2.0 = 0.25x slope);\n"
                 "negative values expand highlight punch.";
    ui_type = "slider";
    ui_min = -1.00;
    ui_max = 2.00;
    ui_step = 0.01;
> = 0.00;

uniform float ShadowLift <
    ui_category = "Local Laplacian Filter";
    ui_label = "Shadow Lift / Deepening";
    ui_tooltip = "Asymmetric large-scale remapping for dark regions below local surround.\n"
                 "Positive values lift deep shadows locally without bright halos (1.0 = 0.5x, 2.0 = 0.25x slope);\n"
                 "negative values deepen shadows.";
    ui_type = "slider";
    ui_min = -1.00;
    ui_max = 2.00;
    ui_step = 0.01;
> = 0.00;

uniform int RemappingCurveMode <
    ui_category = "Local Laplacian Filter";
    ui_label = "Remapping Function";
    ui_tooltip = "Smooth C-infinity (Aubry et al. 2014): Gaussian-modulated detail + tanh-asymptotic\n"
                 "edge compression. Completely avoids noise explosion at zero gradient and slope kinks.\n"
                 "Classic Piecewise (Paris et al. 2011): Exact |d/sigma|^alpha power-law detail curve\n"
                 "with alpha = 2^(-DetailContrast) joined to linear beta edges.";
    ui_type = "combo";
    ui_items = "Smooth C-Infinity (Aubry / Halo-Free)\0Classic Piecewise Power-Law (Paris 2011)\0";
> = 0;

uniform int SliceInterpolationMode <
    ui_category = "Local Laplacian Filter";
    ui_label = "Intensity Slice Interpolation";
    ui_tooltip = "Cubic (Catmull-Rom): O(h^4) accuracy across Jz slices (>70 dB PSNR at 16 slices).\n"
                 "Linear (Classic Aubry): Standard O(h^2) piecewise linear interpolation.";
    ui_type = "combo";
    ui_items = "Cubic Catmull-Rom (Max Precision)\0Linear (Classic Aubry)\0";
> = 0;

uniform float FilterBlend <
    ui_category = "Local Laplacian Filter";
    ui_label = "Master Filter Blend";
    ui_tooltip = "Blends the filtered Jz lightness delta with the original image (0.0 = bypass, 1.0 = full).";
    ui_type = "slider";
    ui_min = 0.00;
    ui_max = 1.00;
    ui_step = 0.01;
> = 1.00;

// --- Category 2: Multi-Scale Band Weights ---
uniform float ScaleWeightFine <
    ui_category = "Multi-Scale Band Weights";
    ui_category_closed = true;
    ui_label = "Fine Detail Weight (Levels 0-1)";
    ui_tooltip = "Multiplies Laplacian bands 0 and 1 (pixel-level micro-texture).\n"
                 "Keep all three scale weights at 1.0 for canonical Fast Local Laplacian Filtering.";
    ui_type = "slider";
    ui_min = 0.00;
    ui_max = 2.00;
    ui_step = 0.01;
> = 1.00;

uniform float ScaleWeightMedium <
    ui_category = "Multi-Scale Band Weights";
    ui_category_closed = true;
    ui_label = "Medium Clarity Weight (Levels 2-4)";
    ui_tooltip = "Multiplies Laplacian bands 2, 3, and 4 (mid-frequency local clarity and structure).";
    ui_type = "slider";
    ui_min = 0.00;
    ui_max = 2.00;
    ui_step = 0.01;
> = 1.00;

uniform float ScaleWeightCoarse <
    ui_category = "Multi-Scale Band Weights";
    ui_category_closed = true;
    ui_label = "Coarse Shading Weight (Levels 5+)";
    ui_tooltip = "Multiplies Laplacian bands 5 and coarser (broad local tone mapping and shadow/highlight reach).";
    ui_type = "slider";
    ui_min = 0.00;
    ui_max = 2.00;
    ui_step = 0.01;
> = 1.00;

// --- Category 3: JzAzBz Color & Precision ---
uniform int ColorReconstructionMode <
    ui_category = "JzAzBz Color & Precision";
    ui_label = "Color Reconstruction Mode";
    ui_tooltip = "JzAzBz Perceptual (Chroma Scaling): Reconstructs directly in JzAzBz with proportional\n"
                 "(az, bz) scaling — preserves Safdar (2017) perceptual hue linearity.\n"
                 "Linear Luminance Ratio (RGB Scale): Uses FLLF strictly in Jz to compute the perceptual\n"
                 "achromatic luminance ratio (Y_gray(Jz_new) / Y_gray(Jz_orig)) and scales linear RGB —\n"
                 "guarantees zero negative channels when darkening and exact xy chromaticity preservation.";
    ui_type = "combo";
    ui_items = "JzAzBz Perceptual (Chroma Scaling)\0Linear Luminance Ratio (RGB Scale)\0";
> = 0;

uniform float ChromaPreservation <
    ui_category = "JzAzBz Color & Precision";
    ui_label = "Chroma Proportionality (JzAzBz Mode)";
    ui_tooltip = "Controls how (az, bz) chroma scales with Jz_new / Jz_orig in JzAzBz Perceptual mode.\n"
                 "0.0: Pure Jz lightness filtering (az, bz unchanged).\n"
                 "1.0: Full proportional scaling smoothly down to 0.0 near black (prevents gamut\n"
                 "excursions in darkened saturated edges and washed-out lifted shadows).";
    ui_type = "slider";
    ui_min = 0.00;
    ui_max = 1.00;
    ui_step = 0.01;
> = 0.75;

uniform int GamutMappingMode <
    ui_category = "JzAzBz Color & Precision";
    ui_label = "Gamut Mapping Mode";
    ui_tooltip = "Controls how out-of-gamut colors are mapped back into the target RGB gamut:\n"
                 "  JzAzBz Adaptive Soft Roll-Off: Adapts the C1 hyperbolic soft-knee to max(knee, rho_orig)\n"
                 "    at exact constant (Jz, hue h_z), so already-in-gamut saturated colors (including pure\n"
                 "    primaries) are never desaturated when unchanged, while newly pushed excursions roll\n"
                 "    off smoothly into Rec.709 (SDR) or Rec.2020 (HDR10/HLG/scRGB).\n"
                 "  JzAzBz Hard Boundary: Exact constant-(Jz, h_z) projection onto gamut boundary;\n"
                 "    100% untouched inside gamut.\n"
                 "  Linear RGB Luminance-Preserving: Fast desaturation toward luminance Y in linear\n"
                 "    Rec.709 (SDR) or Rec.2020 (HDR10/HLG/scRGB).\n"
                 "  Unrestricted Wide Gamut: Disables Rec.2020 boundary on scRGB to allow wide gamut\n"
                 "    beyond Rec.2020 (including signed negative LMS); container-clamps SDR/HDR10.";
    ui_type = "combo";
    ui_items = "JzAzBz Adaptive Soft Roll-Off (Rec.709 / Rec.2020)\0JzAzBz Hard Boundary (Rec.709 / Rec.2020)\0Linear RGB Luminance-Preserving (Rec.709 / Rec.2020)\0Unrestricted Wide Gamut (Beyond Rec.2020 / Signed LMS)\0";
> = 0;

uniform float GamutSoftKnee <
    ui_category = "JzAzBz Color & Precision";
    ui_label = "Gamut Soft-Knee Threshold";
    ui_tooltip = "Minimum fraction of target gamut chroma radius where C1 soft roll-off begins.\n"
                 "Automatically raised to each pixel's original gamut radius so unchanged saturated\n"
                 "in-gamut colors pass through untouched with 0 LSB shift.";
    ui_type = "slider";
    ui_min = 0.50;
    ui_max = 0.98;
    ui_step = 0.01;
> = 0.85;

uniform bool EnableOutputDithering <
    ui_category = "JzAzBz Color & Precision";
    ui_label = "TPDF Output Dithering (8/10-bit)";
    ui_tooltip = "Applies zero-mean Triangular PDF (+/- 1 LSB) dithering on 8-bit and 10-bit\n"
                 "integer backbuffers to preserve full 32-bit internal gradient smoothness.\n"
                 "Automatically disabled on 16-bit floating-point (scRGB) buffers and neutral bypass.";
> = true;

// --- Category 4: SDR / HDR Color Space ---
uniform int ColorSpaceOverride <
    ui_category = "SDR / HDR Color Space";
    ui_label = "Color Space Mode";
    ui_tooltip = "Auto uses ReShade's BUFFER_COLOR_SPACE macro:\n"
                 "  0/1 = SDR (sRGB / Rec.709)\n"
                 "  2   = HDR (scRGB FP16, 1.0 = 80 nits)\n"
                 "  3   = HDR10 (SMPTE ST 2084 PQ / Rec.2020)\n"
                 "  4   = HDR10 (ITU-R BT.2100 HLG / Rec.2020)\n"
                 "Use manual override if a game or wrapper reports BUFFER_COLOR_SPACE = 0 in HDR.";
    ui_type = "combo";
    ui_items = "Auto (BUFFER_COLOR_SPACE)\0SDR (sRGB / Rec.709)\0HDR (scRGB FP16)\0HDR10 (ST 2084 PQ / Rec.2020)\0HDR10 (BT.2100 HLG / Rec.2020)\0";
> = 0;

uniform int SDRTransferFunction <
    ui_category = "SDR / HDR Color Space";
    ui_label = "SDR Transfer Function";
    ui_tooltip = "Electro-Optical Transfer Function used when operating in SDR mode.\n"
                 "IEC 61966-2-1 sRGB uses exact C0-continuous piecewise thresholds.";
    ui_type = "combo";
    ui_items = "Exact IEC 61966-2-1 sRGB\0Pure Power Gamma 2.2\0BT.1886 / Gamma 2.4\0";
> = 0;

uniform float SDRWhiteNits <
    ui_category = "SDR / HDR Color Space";
    ui_label = "SDR Reference White (nits)";
    ui_tooltip = "Absolute peak white luminance (cd/m^2) used to map SDR [0, 1] into Safdar JzAzBz.\n"
                 "100 nits matches Safdar (2017) & ITU-R BT.709; 203 nits matches ITU-R BT.2408.";
    ui_type = "slider";
    ui_min = 80.0;
    ui_max = 400.0;
    ui_step = 1.0;
    ui_units = " nits";
> = 100.0;

uniform float HDRPaperWhiteNits <
    ui_category = "SDR / HDR Color Space";
    ui_label = "HDR Reference White (nits)";
    ui_tooltip = "Diffuse paper white luminance (cd/m^2) used to scale Edge Threshold (Sigma) in HDR.\n"
                 "203 nits is the ITU-R BT.2408 HDR graphics white reference.";
    ui_type = "slider";
    ui_min = 80.0;
    ui_max = 500.0;
    ui_step = 1.0;
    ui_units = " nits";
> = 203.0;

uniform bool AutoDetectHDRPeak <
    ui_category = "SDR / HDR Color Space";
    ui_label = "Auto-Detect Frame HDR Peak";
    ui_tooltip = "Measures the exact maximum Jz across every pixel of each HDR frame so all specular\n"
                 "highlights (2000-10000 nits) sit inside the FLLF slice range and are filtered against\n"
                 "their true local level with maximum slice density. Inactive in SDR.";
> = true;

uniform float HDRPeakNits <
    ui_category = "SDR / HDR Color Space";
    ui_label = "Manual HDR Slice Ceiling (nits)";
    ui_tooltip = "Upper luminance ceiling across which the Jz intensity slices are distributed in HDR\n"
                 "when Auto-Detect Frame HDR Peak is disabled (default 4000 nits).\n"
                 "Set >= your content's true peak: highlights exceeding a manual ceiling are evaluated\n"
                 "against the top slice instead of their own local level (keep Auto-Detect ON to avoid this).";
    ui_type = "slider";
    ui_min = 250.0;
    ui_max = 10000.0;
    ui_step = 50.0;
    ui_units = " nits";
> = 4000.0;

uniform float HLGPeakNits <
    ui_category = "SDR / HDR Color Space";
    ui_label = "HLG Nominal Display Peak (nits)";
    ui_tooltip = "Nominal peak luminance L_W for ITU-R BT.2100 HLG OOTF/Inverse-OOTF system gamma\n"
                 "(gamma = 1.2 + 0.42 * log10(L_W / 1000)). Only used in HLG mode.";
    ui_type = "slider";
    ui_min = 400.0;
    ui_max = 4000.0;
    ui_step = 50.0;
    ui_units = " nits";
> = 1000.0;

// --- Category 5: Diagnostics & Comparison ---
uniform int DiagnosticView <
    ui_category = "Diagnostics & Comparison";
    ui_category_closed = true;
    ui_label = "Output View Mode";
    ui_tooltip = "Normal: Final filtered SDR/HDR output.\n"
                 "Jz Lightness: Visualizes internal filtered Jz normalized to active white/peak.\n"
                 "Delta Jz Map: Visualizes signed FLLF lightness delta (mid-gray = 0, bright = +, dark = -).";
    ui_type = "combo";
    ui_items = "Normal Output\0Filtered Jz Lightness\0Delta Jz Contrast Map (5x Gain)\0";
> = 0;

uniform bool SplitScreenCompare <
    ui_category = "Diagnostics & Comparison";
    ui_category_closed = true;
    ui_label = "Split-Screen Before / After";
    ui_tooltip = "Shows original untouched frame on the left and FLLF output on the right.";
> = false;

uniform float SplitPosition <
    ui_category = "Diagnostics & Comparison";
    ui_category_closed = true;
    ui_label = "Split-Screen Position";
    ui_type = "slider";
    ui_min = 0.00;
    ui_max = 1.00;
    ui_step = 0.005;
> = 0.50;

// =================================================================================================
// Isolated Implementation Namespace
// =================================================================================================
namespace FLLF_JzAzBz
{
    // ---------------------------------------------------------------------------------------------
    // Exact Constants (Safdar et al. 2017 OE-25-13-15131, ST 2084, BT.2100 HLG, IEC 61966-2-1)
    // ---------------------------------------------------------------------------------------------

    // SMPTE ST 2084 / Safdar (2017) shared PQ constants (exact dyadic rationals in IEEE 754 float32)
    static const float PQ_C1         = 0.8359375f;         // 3424 / 4096
    static const float PQ_ONE_M_C1   = 0.1640625f;         // 1 - c1 == c2 - c3 == 21 / 128
    static const float PQ_C2         = 18.8515625f;        // 2413 / 128
    static const float PQ_C3         = 18.6875f;           // 2392 / 128
    static const float PQ_K_NB       = 3.863901854f;       // (c2 - c1 * c3) / c1 (near-black log1p factor)
    static const float PQ_M1         = 0.1593017578125f;   // 2610 / 16384 (also Safdar n)
    static const float PQ_INV_M1     = 6.277394636015326f; // 16384 / 2610
    static const float PQ_M2         = 78.84375f;          // 2523 / 32
    static const float PQ_INV_M2     = 0.012683313515656f; // 32 / 2523
    static const float PQ_ZERO_FLOOR = 7.3095588959e-07f;  // c1^m2 (exact ST 2084 floor at 0 nits)

    // Safdar et al. (2017) JzAzBz specific constants (Eq. 9-12 & Appendix A Eq. 17-23)
    static const float JZ_P          = 134.034375f;        // 1.7 * 2523 / 32
    static const float JZ_INV_P      = 0.007460772656268f; // 1 / (1.7 * 2523 / 32)
    static const float JZ_D          = -0.56f;
    static const float JZ_ONE_P_D    = 0.44f;              // 1 + d
    static const float JZ_D0         = 1.6295499532821566e-11f;
    static const float JZ_LMSP_ZERO  = 3.7035226210190005e-11f; // c1^p (exact cone floor at LMS=0)

    // Pre-folded 64-bit computed RGB (linear D65 nits) <-> Safdar LMS matrices:
    //   M_RGB_to_LMS = M_XYZp_to_LMS * M_bg(b=1.15, g=0.66) * M_RGB_to_XYZ_D65
    // Folding eliminates intermediate rounding and preserves exact round-trip precision.
    static const float3x3 M_REC709_TO_LMS = float3x3(
        +3.585156044e-01f, +5.091821123e-01f, +1.040994830e-01f,
        +2.204480339e-01f, +5.922728796e-01f, +1.595436831e-01f,
        +7.938834228e-02f, +2.303321427e-01f, +6.631989858e-01f
    );

    static const float3x3 M_LMS_TO_REC709 = float3x3(
        +5.929591325e+00f, -5.224543357e+00f, +3.261095130e-01f,
        -2.223887833e+00f, +3.822134207e+00f, -5.704046643e-01f,
        +6.256406420e-02f, -7.020404660e-01f, +1.666910322e+00f
    );

    static const float3x3 M_REC2020_TO_LMS = float3x3(
        +5.300035764e-01f, +3.557036333e-01f, +8.608998994e-02f,
        +2.893882689e-01f, +5.253948228e-01f, +1.574815048e-01f,
        +9.109808251e-02f, +1.475875817e-01f, +7.342338065e-01f
    );

    static const float3x3 M_LMS_TO_REC2020 = float3x3(
        +2.990669997e+00f, -2.049742417e+00f, +8.897679425e-02f,
        -1.634525136e+00f, +3.145628209e+00f, -4.830368859e-01f,
        -4.250511185e-02f, -3.779832174e-01f, +1.448019175e+00f
    );

    // Safdar (2017) Eq. (11) & Appendix A Eq. (18): LMS' <-> Izazbz
    static const float3x3 M_LMSP_TO_IZAZBZ = float3x3(
        +0.500000000e+00f, +0.500000000e+00f, +0.000000000e+00f,
        +3.524000000e+00f, -4.066708000e+00f, +5.427080000e-01f,
        +1.990760000e-01f, +1.096799000e+00f, -1.295875000e+00f
    );

    static const float3x3 M_IZAZBZ_TO_LMSP = float3x3(
        +1.000000000e+00f, +1.386050433e-01f, +5.804731616e-02f,
        +1.000000000e+00f, -1.386050433e-01f, -5.804731616e-02f,
        +1.000000000e+00f, -9.601924203e-02f, -8.118918961e-01f
    );

    // Exact 64-bit computed Rec.2020 <-> Rec.709 linear matrices (for scRGB Rec.2020 gamut mapping)
    static const float3x3 M_REC2020_TO_REC709 = float3x3(
        +1.660491002e+00f, -5.876411388e-01f, -7.284986332e-02f,
        -1.245504745e-01f, +1.132899897e+00f, -8.349422604e-03f,
        -1.815076335e-02f, -1.005788980e-01f, +1.118729661e+00f
    );

    static const float3x3 M_REC709_TO_REC2020 = float3x3(
        +6.274038959e-01f, +3.292830384e-01f, +4.331306569e-02f,
        +6.909728936e-02f, +9.195403951e-01f, +1.136231557e-02f,
        +1.639143888e-02f, +8.801330788e-02f, +8.955952532e-01f
    );

    // Exact CIE D65 luminance coefficients (Y row of M_RGB_to_XYZ)
    static const float3 LUMA_REC709  = float3(0.2126390059f, 0.7151686788f, 0.0721923154f);
    static const float3 LUMA_REC2020 = float3(0.2627002120f, 0.6779980715f, 0.0593017165f);

    // D65 unit-luminance LMS response (row sums of M_REC709_TO_LMS == M_REC2020_TO_LMS)
    static const float2 D65_UNIT_LM     = float2(0.971797200f, 0.972264597f);
    static const float  D65_UNIT_LM_AVG = 0.972030898f;

    // ITU-R BT.2100 HLG constants
    static const float HLG_A = 0.17883277f;
    static const float HLG_B = 0.28466892f; // 1 - 4a
    static const float HLG_C = 0.55991073f; // 0.5 - a * ln(4a)

    // Exact C0-continuous IEC 61966-2-1 sRGB threshold
    static const float SRGB_LIN_THRESH = 0.00313066844f;
    static const float SRGB_ENC_THRESH = 0.04044823628f;

    // Exact Burt-Adelson 5-tap generating kernel [1, 4, 6, 4, 1] / 16 (exact dyadic floats)
    static const float BURT_W5[5] = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };

    // ---------------------------------------------------------------------------------------------
    // Textures & Point-Filtered Samplers (100% IEEE 754 float32 — RGBA32F / R32F)
    // ---------------------------------------------------------------------------------------------

    texture2D TexBackBuffer : COLOR;
    sampler2D SampBackBuffer
    {
        Texture = TexBackBuffer;
        MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;
        AddressU = CLAMP; AddressV = CLAMP;
        SRGBTexture = false;
    };

    // Full-resolution JzAzBz buffer: .r = Jz (= G0), .g = az, .b = bz, .a = original backbuffer alpha
    texture2D TexJzAzBz { Width = _FLLF_W0; Height = _FLLF_H0; Format = RGBA32F; };
    sampler2D SampJzAzBz
    {
        Texture = TexJzAzBz;
        MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;
        AddressU = CLAMP; AddressV = CLAMP;
    };

    // Exact 4-pass 8x8 per-frame HDR peak Jz reduction buffers (~0.52 MB total at 4K)
    texture2D TexPeak1 { Width = _FLLF_PW1; Height = _FLLF_PH1; Format = R32F; };
    sampler2D SampPeak1 { Texture = TexPeak1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexPeak2 { Width = _FLLF_PW2; Height = _FLLF_PH2; Format = R32F; };
    sampler2D SampPeak2 { Texture = TexPeak2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexPeak3 { Width = _FLLF_PW3; Height = _FLLF_PH3; Format = R32F; };
    sampler2D SampPeak3 { Texture = TexPeak3; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexPeak4 { Width = 1; Height = 1; Format = R32F; };
    sampler2D SampPeak4 { Texture = TexPeak4; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

    // Scalar Jz Gaussian Pyramid (G1..G{L-2}) — reused bottom-up as Collapse targets O{L-3}..O1
    texture2D TexG1 { Width = _FLLF_W1; Height = _FLLF_H1; Format = R32F; };
    sampler2D SampG1 { Texture = TexG1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexG2 { Width = _FLLF_W2; Height = _FLLF_H2; Format = R32F; };
    sampler2D SampG2 { Texture = TexG2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexG3 { Width = _FLLF_W3; Height = _FLLF_H3; Format = R32F; };
    sampler2D SampG3 { Texture = TexG3; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#if _FLLF_LEVELS >= 6
    texture2D TexG4 { Width = _FLLF_W4; Height = _FLLF_H4; Format = R32F; };
    sampler2D SampG4 { Texture = TexG4; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _FLLF_LEVELS >= 7
    texture2D TexG5 { Width = _FLLF_W5; Height = _FLLF_H5; Format = R32F; };
    sampler2D SampG5 { Texture = TexG5; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _FLLF_LEVELS >= 8
    texture2D TexG6 { Width = _FLLF_W6; Height = _FLLF_H6; Format = R32F; };
    sampler2D SampG6 { Texture = TexG6; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _FLLF_LEVELS >= 9
    texture2D TexG7 { Width = _FLLF_W7; Height = _FLLF_H7; Format = R32F; };
    sampler2D SampG7 { Texture = TexG7; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif

    // Interpolated Output Laplacian Delta Pyramid (L1..L{L-2}; L0 is fused into final reconstruction!)
    texture2D TexL1 { Width = _FLLF_W1; Height = _FLLF_H1; Format = R32F; };
    sampler2D SampL1 { Texture = TexL1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexL2 { Width = _FLLF_W2; Height = _FLLF_H2; Format = R32F; };
    sampler2D SampL2 { Texture = TexL2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexL3 { Width = _FLLF_W3; Height = _FLLF_H3; Format = R32F; };
    sampler2D SampL3 { Texture = TexL3; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#if _FLLF_LEVELS >= 6
    texture2D TexL4 { Width = _FLLF_W4; Height = _FLLF_H4; Format = R32F; };
    sampler2D SampL4 { Texture = TexL4; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _FLLF_LEVELS >= 7
    texture2D TexL5 { Width = _FLLF_W5; Height = _FLLF_H5; Format = R32F; };
    sampler2D SampL5 { Texture = TexL5; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _FLLF_LEVELS >= 8
    texture2D TexL6 { Width = _FLLF_W6; Height = _FLLF_H6; Format = R32F; };
    sampler2D SampL6 { Texture = TexL6; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _FLLF_LEVELS >= 9
    texture2D TexL7 { Width = _FLLF_W7; Height = _FLLF_H7; Format = R32F; };
    sampler2D SampL7 { Texture = TexL7; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif

    // Helper macro to declare an MRT pack of 2, 3, or 4 RGBA32F textures + samplers
#if _FLLF_MRT_COUNT == 2
    #define _FLLF_DECL_MRT_PACK(NAME, TW, TH) \
        texture2D Tex##NAME##_0 { Width = TW; Height = TH; Format = RGBA32F; }; \
        sampler2D Samp##NAME##_0 { Texture = Tex##NAME##_0; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; }; \
        texture2D Tex##NAME##_1 { Width = TW; Height = TH; Format = RGBA32F; }; \
        sampler2D Samp##NAME##_1 { Texture = Tex##NAME##_1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#elif _FLLF_MRT_COUNT == 3
    #define _FLLF_DECL_MRT_PACK(NAME, TW, TH) \
        texture2D Tex##NAME##_0 { Width = TW; Height = TH; Format = RGBA32F; }; \
        sampler2D Samp##NAME##_0 { Texture = Tex##NAME##_0; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; }; \
        texture2D Tex##NAME##_1 { Width = TW; Height = TH; Format = RGBA32F; }; \
        sampler2D Samp##NAME##_1 { Texture = Tex##NAME##_1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; }; \
        texture2D Tex##NAME##_2 { Width = TW; Height = TH; Format = RGBA32F; }; \
        sampler2D Samp##NAME##_2 { Texture = Tex##NAME##_2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#else
    #define _FLLF_DECL_MRT_PACK(NAME, TW, TH) \
        texture2D Tex##NAME##_0 { Width = TW; Height = TH; Format = RGBA32F; }; \
        sampler2D Samp##NAME##_0 { Texture = Tex##NAME##_0; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; }; \
        texture2D Tex##NAME##_1 { Width = TW; Height = TH; Format = RGBA32F; }; \
        sampler2D Samp##NAME##_1 { Texture = Tex##NAME##_1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; }; \
        texture2D Tex##NAME##_2 { Width = TW; Height = TH; Format = RGBA32F; }; \
        sampler2D Samp##NAME##_2 { Texture = Tex##NAME##_2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; }; \
        texture2D Tex##NAME##_3 { Width = TW; Height = TH; Format = RGBA32F; }; \
        sampler2D Samp##NAME##_3 { Texture = Tex##NAME##_3; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif

    // Level 0->1 separable horizontal intermediate buffer (W1 x H0) for 5x lower remapping ALU
    _FLLF_DECL_MRT_PACK(H1_R, _FLLF_W1, _FLLF_H0)

    // Remapped Delta Gaussian Pyramid Packs G1[Delta_R] .. G{L-1}[Delta_R]
    _FLLF_DECL_MRT_PACK(G1_R, _FLLF_W1, _FLLF_H1)
    _FLLF_DECL_MRT_PACK(G2_R, _FLLF_W2, _FLLF_H2)
    _FLLF_DECL_MRT_PACK(G3_R, _FLLF_W3, _FLLF_H3)
    _FLLF_DECL_MRT_PACK(G4_R, _FLLF_W4, _FLLF_H4)
#if _FLLF_LEVELS >= 6
    _FLLF_DECL_MRT_PACK(G5_R, _FLLF_W5, _FLLF_H5)
#endif
#if _FLLF_LEVELS >= 7
    _FLLF_DECL_MRT_PACK(G6_R, _FLLF_W6, _FLLF_H6)
#endif
#if _FLLF_LEVELS >= 8
    _FLLF_DECL_MRT_PACK(G7_R, _FLLF_W7, _FLLF_H7)
#endif
#if _FLLF_LEVELS >= 9
    _FLLF_DECL_MRT_PACK(G8_R, _FLLF_W8, _FLLF_H8)
#endif

    // ---------------------------------------------------------------------------------------------
    // Numerically Stabilized Math Primitives
    // ---------------------------------------------------------------------------------------------

    float SafePow(float x, float e)
    {
        return (x > 0.0f) ? pow(abs(x), e) : 0.0f;
    }

    float SafeTanh(float x)
    {
        // tanh(15) = 1.0 - 1.87e-13 (exact 1.0 in float32); avoids Inf/Inf NaN on GPUs
        return tanh(clamp(x, -15.0f, 15.0f));
    }

    // Stabilized forward PQ-family rational transfer: ((c1 + c2*x) / (1 + c3*x))^expPow
    // Eliminates float32 cancellation BOTH near black (via c1^expPow * exp(expPow * log1p(w)))
    // and near peak (via exp(expPow * log1p(-u))).
    float StabilizedPQForwardChannel(float nitsVal, float expPow, float zeroFloor)
    {
        float x = SafePow(max(nitsVal, 0.0f) * 1.0e-4f, PQ_M1);
        float w = (PQ_K_NB * x) / (1.0f + PQ_C3 * x);
        float res;
        if (w < 0.02f)
        {
            // Near-black branch: 5th-order log1p(w) around c1^expPow (<1.5e-9 truncation error)
            float w2 = w * w;
            float ln1p = w - 0.5f * w2 + (1.0f / 3.0f) * w2 * w - 0.25f * w2 * w2 + 0.2f * w2 * w2 * w;
            res = zeroFloor * exp(expPow * ln1p);
        }
        else
        {
            float u = (PQ_ONE_M_C1 * (1.0f - x)) / (1.0f + PQ_C3 * x);
            if (abs(u) < 0.02f)
            {
                // Near-peak branch: 5th-order log1p(-u) around 1.0 (<1.5e-9 truncation error)
                float u2 = u * u;
                float lnVal = -u - 0.5f * u2 - (1.0f / 3.0f) * u2 * u - 0.25f * u2 * u2 - 0.2f * u2 * u2 * u;
                res = exp(expPow * lnVal);
            }
            else
            {
                res = SafePow(max(1.0f - u, 0.0f), expPow);
            }
        }
        return res;
    }

    // Stabilized inverse PQ-family rational transfer returning absolute nits [0, 10000]:
    // Eliminates float32 cancellation BOTH near peak (1 - vp = -expm1(t), den = (1-c1) + c3*(1-vp))
    // and near black (vp - c1 = c1 * expm1(ln(v / c1^expPow) / expPow)).
    float StabilizedPQInverseChannel(float encodedVal, float invExpPow, float zeroFloor)
    {
        float res = 0.0f;
        if (encodedVal > zeroFloor)
        {
            float num;
            float den;
            if (encodedVal > 0.5f)
            {
                float t = log(encodedVal) * invExpPow;
                float t2 = t * t;
                float oneMinusVp = -t - 0.5f * t2 - (1.0f / 6.0f) * t2 * t - (1.0f / 24.0f) * t2 * t2;
                float vp = 1.0f - oneMinusVp;
                num = max(vp - PQ_C1, 0.0f);
                den = max(PQ_ONE_M_C1 + PQ_C3 * oneMinusVp, 1.0e-20f);
            }
            else if (encodedVal < zeroFloor * 128.0f)
            {
                float t = log(encodedVal / zeroFloor) * invExpPow;
                float t2 = t * t;
                float em1 = t + 0.5f * t2 + (1.0f / 6.0f) * t2 * t + (1.0f / 24.0f) * t2 * t2;
                num = max(PQ_C1 * em1, 0.0f);
                float oneMinusVp = PQ_ONE_M_C1 - num;
                den = max(PQ_ONE_M_C1 + PQ_C3 * oneMinusVp, 1.0e-20f);
            }
            else
            {
                float vp = SafePow(encodedVal, invExpPow);
                float oneMinusVp = 1.0f - vp;
                num = max(vp - PQ_C1, 0.0f);
                den = max(PQ_ONE_M_C1 + PQ_C3 * oneMinusVp, 1.0e-20f);
            }
            res = 10000.0f * SafePow(num / den, PQ_INV_M1);
        }
        return res;
    }

    float3 StabilizedPQForward3(float3 nitsVal, float expPow, float zeroFloor)
    {
        return float3(
            StabilizedPQForwardChannel(nitsVal.x, expPow, zeroFloor),
            StabilizedPQForwardChannel(nitsVal.y, expPow, zeroFloor),
            StabilizedPQForwardChannel(nitsVal.z, expPow, zeroFloor)
        );
    }

    float3 StabilizedPQInverse3(float3 encodedVal, float invExpPow, float zeroFloor)
    {
        return float3(
            StabilizedPQInverseChannel(encodedVal.x, invExpPow, zeroFloor),
            StabilizedPQInverseChannel(encodedVal.y, invExpPow, zeroFloor),
            StabilizedPQInverseChannel(encodedVal.z, invExpPow, zeroFloor)
        );
    }

    // Odd-symmetric sign-preserving JzAzBz PQ curve centered at c1^p (JZ_LMSP_ZERO):
    //   lms >= 0: identical to Safdar (2017) Eq. (10).
    //   lms <  0: c1^p - (pq(|lms|) - c1^p), C0-continuous at 0 with cancellation-free expm1 near 0.
    // Preserves imaginary / extreme negative scRGB LMS values without NaN or hard zero-clipping.
    float SignedJzPQForwardChannel(float lmsNits)
    {
        float clamped = clamp(lmsNits, -10000.0f, 10000.0f);
        float res;
        if (clamped >= 0.0f)
        {
            res = StabilizedPQForwardChannel(clamped, JZ_P, JZ_LMSP_ZERO);
        }
        else
        {
            float ax = -clamped;
            float x = SafePow(ax * 1.0e-4f, PQ_M1);
            float w = (PQ_K_NB * x) / (1.0f + PQ_C3 * x);
            if (w < 0.02f)
            {
                float w2 = w * w;
                float ln1p = w - 0.5f * w2 + (1.0f / 3.0f) * w2 * w - 0.25f * w2 * w2 + 0.2f * w2 * w2 * w;
                float a = JZ_P * ln1p;
                if (a < 0.02f)
                {
                    float a2 = a * a;
                    float em1 = a + 0.5f * a2 + (1.0f / 6.0f) * a2 * a + (1.0f / 24.0f) * a2 * a2;
                    res = JZ_LMSP_ZERO * (1.0f - em1);
                }
                else
                {
                    res = JZ_LMSP_ZERO * (2.0f - exp(a));
                }
            }
            else
            {
                res = (2.0f * JZ_LMSP_ZERO) - StabilizedPQForwardChannel(ax, JZ_P, JZ_LMSP_ZERO);
            }
        }
        return res;
    }

    float SignedJzPQInverseChannel(float lmspVal)
    {
        float res;
        if (lmspVal >= JZ_LMSP_ZERO)
        {
            res = StabilizedPQInverseChannel(lmspVal, JZ_INV_P, JZ_LMSP_ZERO);
        }
        else
        {
            float mirrored = (2.0f * JZ_LMSP_ZERO) - lmspVal;
            res = -StabilizedPQInverseChannel(mirrored, JZ_INV_P, JZ_LMSP_ZERO);
        }
        return res;
    }

    float3 SignedJzPQForward3(float3 lmsNits)
    {
        return float3(
            SignedJzPQForwardChannel(lmsNits.x),
            SignedJzPQForwardChannel(lmsNits.y),
            SignedJzPQForwardChannel(lmsNits.z)
        );
    }

    float3 SignedJzPQInverse3(float3 lmspVal)
    {
        return float3(
            SignedJzPQInverseChannel(lmspVal.x),
            SignedJzPQInverseChannel(lmspVal.y),
            SignedJzPQInverseChannel(lmspVal.z)
        );
    }

    // ---------------------------------------------------------------------------------------------
    // Active Color Space Resolution & SDR / HDR EOTF / Inverse-EOTF
    // ---------------------------------------------------------------------------------------------

    // Returns: 1 = SDR sRGB/Rec.709, 2 = HDR scRGB, 3 = HDR10 PQ, 4 = HDR10 HLG
    int GetActiveColorSpace()
    {
    #if defined(BUFFER_COLOR_SPACE) && (BUFFER_COLOR_SPACE == 2)
        int cs = 2;
    #elif defined(BUFFER_COLOR_SPACE) && (BUFFER_COLOR_SPACE == 3)
        int cs = 3;
    #elif defined(BUFFER_COLOR_SPACE) && (BUFFER_COLOR_SPACE == 4)
        int cs = 4;
    #else
        int cs = 1;
    #endif
        if (ColorSpaceOverride > 0)
            cs = ColorSpaceOverride;
        return cs;
    }

    float3 SanitizeRGB(float3 c)
    {
    #if __RENDERER__ == 0x9000
        return clamp(c, -10000.0f, 10000.0f);
    #else
        // IEEE 754 exponent == 0xFF catches BOTH NaN and +/-Inf reliably even without /Gis
        uint3 expBits = asuint(c) & 0x7F800000u;
        return float3(
            (expBits.r == 0x7F800000u) ? 0.0f : c.r,
            (expBits.g == 0x7F800000u) ? 0.0f : c.g,
            (expBits.b == 0x7F800000u) ? 0.0f : c.b
        );
    #endif
    }

    float SDRToLinearChannel(float u)
    {
        u = max(u, 0.0f);
        float res;
        if (SDRTransferFunction == 1)
            res = SafePow(u, 2.2f);
        else if (SDRTransferFunction == 2)
            res = SafePow(u, 2.4f);
        else
            res = (u <= SRGB_ENC_THRESH) ? (u / 12.92f) : SafePow((u + 0.055f) / 1.055f, 2.4f);
        return res;
    }

    float LinearToSDRChannel(float l)
    {
        l = max(l, 0.0f);
        float res;
        if (SDRTransferFunction == 1)
            res = SafePow(l, 1.0f / 2.2f);
        else if (SDRTransferFunction == 2)
            res = SafePow(l, 1.0f / 2.4f);
        else
            res = (l <= SRGB_LIN_THRESH) ? (l * 12.92f) : (1.055f * SafePow(l, 1.0f / 2.4f) - 0.055f);
        return res;
    }

    float3 SDRToLinear(float3 c)
    {
        return float3(SDRToLinearChannel(c.r), SDRToLinearChannel(c.g), SDRToLinearChannel(c.b));
    }

    float3 LinearToSDR(float3 c)
    {
        return float3(LinearToSDRChannel(c.r), LinearToSDRChannel(c.g), LinearToSDRChannel(c.b));
    }

    float HLGInverseOETFChannel(float e)
    {
        e = max(e, 0.0f);
        return (e <= 0.5f) ? ((e * e) / 3.0f) : ((exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0f);
    }

    float HLGOETFChannel(float es)
    {
        es = max(es, 0.0f);
        return (es <= (1.0f / 12.0f)) ? sqrt(3.0f * es) : (HLG_A * log(max(12.0f * es - HLG_B, 1.0e-20f)) + HLG_C);
    }

    float3 HLGToLinearNits(float3 hlg)
    {
        float3 es = float3(
            HLGInverseOETFChannel(hlg.r),
            HLGInverseOETFChannel(hlg.g),
            HLGInverseOETFChannel(hlg.b)
        );
        float Lw = max(HLGPeakNits, 100.0f);
        float sysGamma = 1.2f + 0.42f * log10(Lw / 1000.0f);
        float Ys = dot(es, LUMA_REC2020);
        float scale = (Ys > 0.0f) ? (Lw * SafePow(Ys, sysGamma - 1.0f)) : 0.0f;
        return es * scale;
    }

    float3 LinearNitsToHLG(float3 nits2020)
    {
        float3 rgbD = max(nits2020, 0.0f);
        float Lw = max(HLGPeakNits, 100.0f);
        float sysGamma = 1.2f + 0.42f * log10(Lw / 1000.0f);
        float Yd = dot(rgbD, LUMA_REC2020);
        float invScale = (Yd > 0.0f) ? (SafePow(Yd / Lw, (1.0f - sysGamma) / sysGamma) / Lw) : 0.0f;
        float3 es = rgbD * invScale;
        return float3(HLGOETFChannel(es.r), HLGOETFChannel(es.g), HLGOETFChannel(es.b));
    }

    // ---------------------------------------------------------------------------------------------
    // Exact Safdar et al. (2017) JzAzBz Forward & Inverse Transformations
    // ---------------------------------------------------------------------------------------------

    // Scalar D65 luminance (nits) -> Jz (used to calibrate active slice ceiling & sigma)
    float LuminanceNitsToJz(float nits)
    {
        float2 lm = D65_UNIT_LM * max(nits, 0.0f);
        float lp = StabilizedPQForwardChannel(lm.x, JZ_P, JZ_LMSP_ZERO);
        float mp = StabilizedPQForwardChannel(lm.y, JZ_P, JZ_LMSP_ZERO);
        float Iz = 0.5f * (lp + mp);
        return max(((JZ_ONE_P_D * Iz) / (1.0f + JZ_D * Iz)) - JZ_D0, 0.0f);
    }

    // Exact scalar inverse of LuminanceNitsToJz: Jz -> D65 equivalent luminance (nits)
    float JzToLuminanceNits(float Jz)
    {
        float JzClamped = clamp(Jz, 0.0f, 1.0f - JZ_D0);
        float Jzd = JzClamped + JZ_D0;
        float Iz = clamp(Jzd / max(JZ_ONE_P_D - JZ_D * Jzd, 1.0e-20f), JZ_LMSP_ZERO, 1.0f);
        return StabilizedPQInverseChannel(Iz, JZ_INV_P, JZ_LMSP_ZERO) / D65_UNIT_LM_AVG;
    }

    float3 LMSNitsToJzAzBz(float3 lmsNits)
    {
        // Sign-preserving Safdar (2017) PQ curve: identical to exact Safdar for all real colors
        // (Rec.709, P3, Rec.2020, spectral locus all have LMS > 0) while keeping imaginary /
        // extreme negative scRGB LMS values invertible without NaN or zero-clipping.
        float3 lmsp = SignedJzPQForward3(lmsNits);
        float3 izab = mul(M_LMSP_TO_IZAZBZ, lmsp);
        float Iz = clamp(izab.x, -1.0f, 1.0f);
        float Jz = ((JZ_ONE_P_D * Iz) / (1.0f + JZ_D * Iz)) - JZ_D0;
        return float3(clamp(Jz, -0.28f, 1.0f - JZ_D0), izab.y, izab.z);
    }

    float3 JzAzBzToLMSNits(float3 jab, bool clampToCone)
    {
        float minJz = clampToCone ? 0.0f : -0.28f;
        float Jz = clamp(jab.x, minJz, 1.0f - JZ_D0);
        float Jzd = Jz + JZ_D0;
        float minIz = clampToCone ? JZ_LMSP_ZERO : -1.0f;
        float Iz = clamp(Jzd / max(JZ_ONE_P_D - JZ_D * Jzd, 1.0e-20f), minIz, 1.0f);

        float3 chromaVec = mul(M_IZAZBZ_TO_LMSP, float3(0.0f, jab.y, jab.z));
        float3 lmsp = Iz + chromaVec;
        float3 lmsOut;

        if (clampToCone)
        {
            // Hue-preserving cone-response validity clamp to [c1^p, 1.0] (exact h_z & Jz preserved)
            float minLmsp = min(lmsp.x, min(lmsp.y, lmsp.z));
            float maxLmsp = max(lmsp.x, max(lmsp.y, lmsp.z));
            if (minLmsp < JZ_LMSP_ZERO || maxLmsp > 1.0f)
            {
                float availLow  = max(Iz - JZ_LMSP_ZERO, 0.0f);
                float availHigh = max(1.0f - Iz, 0.0f);
                float scale = 1.0f;
                if (chromaVec.x < 0.0f) scale = min(scale, availLow  / (-chromaVec.x));
                if (chromaVec.y < 0.0f) scale = min(scale, availLow  / (-chromaVec.y));
                if (chromaVec.z < 0.0f) scale = min(scale, availLow  / (-chromaVec.z));
                if (chromaVec.x > 0.0f) scale = min(scale, availHigh / ( chromaVec.x));
                if (chromaVec.y > 0.0f) scale = min(scale, availHigh / ( chromaVec.y));
                if (chromaVec.z > 0.0f) scale = min(scale, availHigh / ( chromaVec.z));
                lmsp = clamp(Iz + chromaVec * saturate(scale), JZ_LMSP_ZERO, 1.0f);
            }
            lmsOut = StabilizedPQInverse3(lmsp, JZ_INV_P, JZ_LMSP_ZERO);
        }
        else
        {
            // Unrestricted signed inverse: preserves negative LMS for imaginary / extreme scRGB
            lmsOut = SignedJzPQInverse3(clamp(lmsp, -1.0f, 1.0f));
        }

        return lmsOut;
    }

    float3 BackBufferToJzAzBz(float3 rawRgb)
    {
        rawRgb = SanitizeRGB(rawRgb);
        int cs = GetActiveColorSpace();
        float3 lmsNits;
        if (cs == 2)
        {
            // scRGB FP16: linear Rec.709 primaries where 1.0 = 80 nits; negative values encode wide gamut
            float3 nits709 = rawRgb * 80.0f;
            lmsNits = mul(M_REC709_TO_LMS, nits709);
        }
        else if (cs == 3)
        {
            // HDR10 ST 2084 PQ / Rec.2020
            float3 nits2020 = StabilizedPQInverse3(saturate(rawRgb), PQ_INV_M2, PQ_ZERO_FLOOR);
            lmsNits = mul(M_REC2020_TO_LMS, nits2020);
        }
        else if (cs == 4)
        {
            // HDR10 ITU-R BT.2100 HLG / Rec.2020
            float3 nits2020 = HLGToLinearNits(saturate(rawRgb));
            lmsNits = mul(M_REC2020_TO_LMS, nits2020);
        }
        else
        {
            // SDR sRGB / Rec.709 mapped to absolute SDR reference white nits
            float3 nits709 = SDRToLinear(saturate(rawRgb)) * SDRWhiteNits;
            lmsNits = mul(M_REC709_TO_LMS, nits709);
        }
        return LMSNitsToJzAzBz(lmsNits);
    }

    // Evaluates (Iz + s * chromaVec) -> linear Target RGB nits and checks cone & target gamut bounds
    bool EvalScaledChromaToRGB(
        float Iz, float3 chromaVec, float s, float3x3 matLMStoRGB, float maxNits, float tol,
        out float3 outRgb)
    {
        float3 lmsp = Iz + s * chromaVec;
        float coneTol = tol * 1.0e-8f;
        bool inCone = (min(lmsp.x, min(lmsp.y, lmsp.z)) >= (JZ_LMSP_ZERO - coneTol))
                   && (max(lmsp.x, max(lmsp.y, lmsp.z)) <= (1.0f + coneTol));
        outRgb = float3(0.0f, 0.0f, 0.0f);
        bool ok = false;
        if (inCone)
        {
            float3 lmspClamped = clamp(lmsp, JZ_LMSP_ZERO, 1.0f);
            float3 lms = StabilizedPQInverse3(lmspClamped, JZ_INV_P, JZ_LMSP_ZERO);
            outRgb = mul(matLMStoRGB, lms);
            ok = (min(outRgb.r, min(outRgb.g, outRgb.b)) >= -tol)
              && (max(outRgb.r, max(outRgb.g, outRgb.b)) <= (maxNits + tol));
        }
        return ok;
    }

    // Exact constant-(Jz, hue h_z) gamut mapping directly inside JzAzBz.
    //   softRollOff = true  (Mode 0): Adaptive C1 hyperbolic chroma roll-off starting at
    //                                 max(GamutSoftKnee, rho_orig) — unchanged in-gamut saturated
    //                                 colors (including pure primaries) pass through with 0 LSB shift,
    //                                 while filter-induced excursions roll off smoothly.
    //   softRollOff = false (Mode 1): Exact projection onto target gamut boundary;
    //                                 all in-gamut colors pass through untouched with zero bisection.
    float3 GamutMapJzAzBzToTargetRGB(float3 jab, float3 jabOrig, float3x3 matLMStoRGB, float maxNits, bool softRollOff)
    {
        float Jz = clamp(jab.x, 0.0f, 1.0f - JZ_D0);
        float Jzd = Jz + JZ_D0;
        float Iz = clamp(Jzd / max(JZ_ONE_P_D - JZ_D * Jzd, 1.0e-20f), JZ_LMSP_ZERO, 1.0f);
        float3 chromaVec = mul(M_IZAZBZ_TO_LMSP, float3(0.0f, jab.y, jab.z));
        float gamTol = maxNits * 5.0e-5f;

        float3 rgb0;
        bool ok0 = EvalScaledChromaToRGB(Iz, chromaVec, 0.0f, matLMStoRGB, maxNits, gamTol, rgb0);
        float3 finalRgb = clamp(rgb0, 0.0f, maxNits);

        float maxChroma = max(abs(chromaVec.x), max(abs(chromaVec.y), abs(chromaVec.z)));
        if (ok0 && maxChroma >= 1.0e-12f)
        {
            if (softRollOff)
            {
                float knee = clamp(GamutSoftKnee, 0.50f, 0.98f);
                float invKnee = 1.0f / knee;

                float3 rgbKnee;
                if (EvalScaledChromaToRGB(Iz, chromaVec, invKnee, matLMStoRGB, maxNits, 0.0f, rgbKnee))
                {
                    // Strictly inside untouched core (rho <= knee): zero bisection, exact pass-through
                    float3 rgb1;
                    EvalScaledChromaToRGB(Iz, chromaVec, 1.0f, matLMStoRGB, maxNits, gamTol, rgb1);
                    finalRgb = clamp(rgb1, 0.0f, maxNits);
                }
                else
                {
                    // Adapt effective knee to the original pixel's gamut radius so unchanged in-gamut
                    // saturated colors map identically to themselves (zero shift at FilterBlend > 0)
                    float JzO = clamp(jabOrig.x, 0.0f, 1.0f - JZ_D0);
                    float JzdO = JzO + JZ_D0;
                    float IzO = clamp(JzdO / max(JZ_ONE_P_D - JZ_D * JzdO, 1.0e-20f), JZ_LMSP_ZERO, 1.0f);
                    float3 chromaOrig = mul(M_IZAZBZ_TO_LMSP, float3(0.0f, jabOrig.y, jabOrig.z));

                    float sEdgeOrig = invKnee;
                    float3 rgbTmp;
                    if (!EvalScaledChromaToRGB(IzO, chromaOrig, invKnee, matLMStoRGB, maxNits, 0.0f, rgbTmp))
                    {
                        if (!EvalScaledChromaToRGB(IzO, chromaOrig, 1.0f, matLMStoRGB, maxNits, 0.0f, rgbTmp))
                        {
                            sEdgeOrig = 1.0f;
                        }
                        else
                        {
                            float loO = 1.0f;
                            float hiO = invKnee;
                            [loop]
                            for (int iterO = 0; iterO < 12; ++iterO)
                            {
                                float midO = 0.5f * (loO + hiO);
                                if (EvalScaledChromaToRGB(IzO, chromaOrig, midO, matLMStoRGB, maxNits, 0.0f, rgbTmp))
                                    loO = midO;
                                else
                                    hiO = midO;
                            }
                            sEdgeOrig = loO;
                        }
                    }

                    float3 rgb1;
                    bool ok1 = EvalScaledChromaToRGB(Iz, chromaVec, 1.0f, matLMStoRGB, maxNits, gamTol, rgb1);
                    if (sEdgeOrig <= 1.0f + 1.0e-5f)
                    {
                        // Original pixel was already on/outside gamut boundary -> preserve in-gamut or project cleanly
                        if (ok1)
                        {
                            finalRgb = clamp(rgb1, 0.0f, maxNits);
                        }
                        else
                        {
                            float lo = 0.0f;
                            float hi = 1.0f;
                            [loop]
                            for (int iter = 0; iter < 12; ++iter)
                            {
                                float mid = 0.5f * (lo + hi);
                                if (EvalScaledChromaToRGB(Iz, chromaVec, mid, matLMStoRGB, maxNits, 0.0f, rgbTmp))
                                    lo = mid;
                                else
                                    hi = mid;
                            }
                            float3 rgbMapped;
                            EvalScaledChromaToRGB(Iz, chromaVec, lo, matLMStoRGB, maxNits, 0.0f, rgbMapped);
                            finalRgb = clamp(rgbMapped, 0.0f, maxNits);
                        }
                    }
                    else if (ok1 && EvalScaledChromaToRGB(Iz, chromaVec, sEdgeOrig, matLMStoRGB, maxNits, gamTol, rgbTmp))
                    {
                        // Candidate radius <= max(knee, rho_orig): exact untouched pass-through!
                        finalRgb = clamp(rgb1, 0.0f, maxNits);
                    }
                    else
                    {
                        float effKnee = clamp(1.0f / sEdgeOrig, knee, 1.0f);
                        float lo = ok1 ? 1.0f : 0.0f;
                        float hi = ok1 ? sEdgeOrig : 1.0f;

                        [loop]
                        for (int iter = 0; iter < 12; ++iter)
                        {
                            float mid = 0.5f * (lo + hi);
                            if (EvalScaledChromaToRGB(Iz, chromaVec, mid, matLMStoRGB, maxNits, 0.0f, rgbTmp))
                                lo = mid;
                            else
                                hi = mid;
                        }

                        // C1 hyperbolic shoulder from effKnee -> 1.0
                        float sEdge = lo;
                        float rho = 1.0f / max(sEdge, 1.0e-6f);
                        float excess = max(rho - effKnee, 0.0f);
                        float margin = max(1.0f - effKnee, 1.0e-6f);
                        float rhoOut = effKnee + margin * (excess / (margin + excess));
                        float sFinal = min(rhoOut * sEdge, 1.0f);

                        float3 rgbMapped;
                        EvalScaledChromaToRGB(Iz, chromaVec, sFinal, matLMStoRGB, maxNits, 0.0f, rgbMapped);
                        finalRgb = clamp(rgbMapped, 0.0f, maxNits);
                    }
                }
            }
            else
            {
                float3 rgb1;
                if (EvalScaledChromaToRGB(Iz, chromaVec, 1.0f, matLMStoRGB, maxNits, gamTol, rgb1))
                {
                    finalRgb = clamp(rgb1, 0.0f, maxNits);
                }
                else
                {
                    float lo = 0.0f;
                    float hi = 1.0f;

                    [loop]
                    for (int iter = 0; iter < 12; ++iter)
                    {
                        float mid = 0.5f * (lo + hi);
                        float3 rgbMid;
                        if (EvalScaledChromaToRGB(Iz, chromaVec, mid, matLMStoRGB, maxNits, 0.0f, rgbMid))
                            lo = mid;
                        else
                            hi = mid;
                    }

                    float3 rgbMapped;
                    EvalScaledChromaToRGB(Iz, chromaVec, lo, matLMStoRGB, maxNits, 0.0f, rgbMapped);
                    finalRgb = clamp(rgbMapped, 0.0f, maxNits);
                }
            }
        }
        return finalRgb;
    }

    // Hue-preserving gamut compression toward luminance axis (avoids per-channel clipping hue twist)
    float3 GamutProtectLinearRGB(float3 rgb, float3 lumaWeights, float maxVal)
    {
        float Y = dot(rgb, lumaWeights);
        float3 res;
        if (Y <= 0.0f)
        {
            res = float3(0.0f, 0.0f, 0.0f);
        }
        else if (Y >= maxVal)
        {
            res = float3(maxVal, maxVal, maxVal);
        }
        else
        {
            float minC = min(rgb.r, min(rgb.g, rgb.b));
            if (minC < 0.0f)
            {
                float t = Y / max(Y - minC, 1.0e-20f);
                rgb = lerp(float3(Y, Y, Y), rgb, saturate(t));
            }

            float maxC = max(rgb.r, max(rgb.g, rgb.b));
            if (maxC > maxVal)
            {
                float t = (maxVal - Y) / max(maxC - Y, 1.0e-20f);
                rgb = lerp(float3(Y, Y, Y), rgb, saturate(t));
            }
            res = clamp(rgb, 0.0f, maxVal);
        }
        return res;
    }

    // Encodes linear target-gamut nits (Rec.709 for SDR, Rec.2020 for HDR10/HLG/scRGB-mapped)
    // into the active backbuffer encoding. For unrestricted scRGB, passes linear Rec.709 nits directly.
    float3 EncodeLinearNitsToBackBuffer(float3 linNits, int cs, bool isRec2020Space)
    {
        float3 outRgb;
        if (cs == 2)
        {
            float3 nits709 = linNits;
            if (isRec2020Space)
                nits709 = mul(M_REC2020_TO_REC709, linNits);
            outRgb = nits709 * (1.0f / 80.0f);
        }
        else if (cs == 3)
        {
            outRgb = saturate(StabilizedPQForward3(clamp(linNits, 0.0f, 10000.0f), PQ_M2, PQ_ZERO_FLOOR));
        }
        else if (cs == 4)
        {
            float Lw = max(HLGPeakNits, 100.0f);
            outRgb = saturate(LinearNitsToHLG(clamp(linNits, 0.0f, Lw)));
        }
        else
        {
            float whiteNits = max(SDRWhiteNits, 1.0f);
            outRgb = saturate(LinearToSDR(clamp(linNits, 0.0f, whiteNits) / whiteNits));
        }
        return outRgb;
    }

    // Unified single-call-site color reconstruction & gamut mapping (fast compile across all APIs).
    // Supports both ColorReconstructionMode (JzAzBz Perceptual / Linear Luminance Ratio) and all
    // 4 GamutMappingMode options (Soft Roll-Off, Hard Boundary, Linear RGB, Unrestricted Wide Gamut).
    float3 ReconstructColorToBackBuffer(float3 origRawRgb, float3 jabOrig, float JzNew)
    {
        int cs = GetActiveColorSpace();
        float whiteNits = max(SDRWhiteNits, 1.0f);
        float Lw = max(HLGPeakNits, 100.0f);

        // Select target gamut parameters ONCE (Rec.709 for SDR; Rec.2020 for scRGB/HDR10/HLG)
        bool useRec2020Target = (cs != 1);
        float3x3 matLMStoTarget;
        float3 lumaTarget;
        if (useRec2020Target)
        {
            matLMStoTarget = M_LMS_TO_REC2020;
            lumaTarget     = LUMA_REC2020;
        }
        else
        {
            matLMStoTarget = M_LMS_TO_REC709;
            lumaTarget     = LUMA_REC709;
        }
        float maxNits = (cs == 1) ? whiteNits : ((cs == 4) ? Lw : 10000.0f);

        bool needJzGamutMap = false;
        float3 jabForMap = float3(JzNew, 0.0f, 0.0f);
        float3 targetLinNits = float3(0.0f, 0.0f, 0.0f);
        bool outputIsRec2020 = useRec2020Target;

        if (ColorReconstructionMode == 1)
        {
            // Linear Luminance Ratio mode: scale original linear RGB by exact Y_new / Y_orig from Jz
            float3 cleanRaw = SanitizeRGB(origRawRgb);
            float yOrig = JzToLuminanceNits(jabOrig.x);
            float yNew  = JzToLuminanceNits(JzNew);
            bool liftFromBlack = (yOrig <= 1.0e-7f);
            float lumaRatio = liftFromBlack ? 0.0f : (yNew / yOrig);

            float3 origLinNits;
            if (cs == 2)
                origLinNits = cleanRaw * 80.0f;
            else if (cs == 3)
                origLinNits = StabilizedPQInverse3(saturate(cleanRaw), PQ_INV_M2, PQ_ZERO_FLOOR);
            else if (cs == 4)
                origLinNits = HLGToLinearNits(saturate(cleanRaw));
            else
                origLinNits = SDRToLinear(saturate(cleanRaw)) * whiteNits;

            float3 scaledNits;
            if (liftFromBlack)
                scaledNits = float3(yNew, yNew, yNew);
            else
                scaledNits = origLinNits * lumaRatio;

            if (GamutMappingMode == 3)
            {
                // Unrestricted Wide Gamut: preserve scRGB beyond Rec.2020; container-clamp SDR/HDR10
                targetLinNits = scaledNits;
                outputIsRec2020 = (cs == 3 || cs == 4);
            }
            else
            {
                // Convert scRGB to Rec.2020 for target-gamut bounds check
                float3 checkNits = scaledNits;
                if (cs == 2)
                    checkNits = mul(M_REC709_TO_REC2020, scaledNits);
                bool outOfGamut = (min(checkNits.r, min(checkNits.g, checkNits.b)) < 0.0f)
                               || (max(checkNits.r, max(checkNits.g, checkNits.b)) > maxNits);
                if (!outOfGamut)
                {
                    targetLinNits = scaledNits;
                    outputIsRec2020 = (cs == 3 || cs == 4);
                }
                else if (GamutMappingMode == 2)
                {
                    targetLinNits = GamutProtectLinearRGB(checkNits, lumaTarget, maxNits);
                }
                else
                {
                    float3 lmsScaled;
                    if (cs == 1 || cs == 2)
                        lmsScaled = mul(M_REC709_TO_LMS, scaledNits);
                    else
                        lmsScaled = mul(M_REC2020_TO_LMS, scaledNits);
                    jabForMap = LMSNitsToJzAzBz(lmsScaled);
                    needJzGamutMap = true;
                }
            }
        }
        else
        {
            // JzAzBz Perceptual mode with proportional chroma scaling (smoothly down to 0.0 near black)
            float chromaScale = 1.0f;
            if (ChromaPreservation > 0.0f && jabOrig.x > 1.0e-7f && JzNew >= 0.0f)
            {
                float ratio = clamp(JzNew / jabOrig.x, 0.0f, 2.5f);
                chromaScale = lerp(1.0f, ratio, ChromaPreservation);
            }
            jabForMap = float3(JzNew, jabOrig.y * chromaScale, jabOrig.z * chromaScale);

            if (GamutMappingMode == 0 || GamutMappingMode == 1)
            {
                needJzGamutMap = true;
            }
            else if (GamutMappingMode == 2)
            {
                float3 lmsNits = JzAzBzToLMSNits(jabForMap, true);
                targetLinNits = GamutProtectLinearRGB(mul(matLMStoTarget, lmsNits), lumaTarget, maxNits);
            }
            else
            {
                // Mode 3: Unrestricted Wide Gamut (beyond Rec.2020 & signed negative LMS on scRGB)
                if (cs == 2)
                {
                    float3 lmsNits = JzAzBzToLMSNits(jabForMap, false);
                    targetLinNits = mul(M_LMS_TO_REC709, lmsNits);
                    outputIsRec2020 = false;
                }
                else
                {
                    float3 lmsNits = JzAzBzToLMSNits(jabForMap, true);
                    targetLinNits = clamp(mul(matLMStoTarget, lmsNits), 0.0f, maxNits);
                }
            }
        }

        // Single call site to constant-(Jz, hue h_z) JzAzBz gamut mapper
        if (needJzGamutMap)
        {
            targetLinNits = GamutMapJzAzBzToTargetRGB(jabForMap, jabOrig, matLMStoTarget, maxNits, GamutMappingMode == 0);
            outputIsRec2020 = useRec2020Target;
        }

        return EncodeLinearNitsToBackBuffer(targetLinNits, cs, outputIsRec2020);
    }

    // ---------------------------------------------------------------------------------------------
    // FLLF Parameter Calibration & Delta-Residual Remapping Function
    // ---------------------------------------------------------------------------------------------

    // Active upper Jz ceiling for intensity slices (exact per-frame peak in HDR when Auto-Detect ON)
    float GetActiveJzCeiling()
    {
        int cs = GetActiveColorSpace();
        float ceilingJz;
        if (cs == 1)
        {
            ceilingJz = max(LuminanceNitsToJz(SDRWhiteNits), 0.05f);
        }
        else if (AutoDetectHDRPeak)
        {
            float detected = _FLLF_FETCH(SampPeak4, int2(0, 0), int2(1, 1)).r * 1.005f;
            float floorJz = max(LuminanceNitsToJz(HDRPaperWhiteNits), 0.10f);
            ceilingJz = clamp(max(detected, floorJz), floorJz, 1.0f - JZ_D0);
        }
        else if (cs == 4)
        {
            ceilingJz = max(LuminanceNitsToJz(max(HLGPeakNits, HDRPeakNits)), 0.10f);
        }
        else
        {
            ceilingJz = max(LuminanceNitsToJz(HDRPeakNits), 0.10f);
        }
        return ceilingJz;
    }

    // Perceptual reference white Jz for consistent Sigma calibration between SDR & HDR
    float GetReferenceWhiteJz()
    {
        int cs = GetActiveColorSpace();
        float refNits = (cs == 1) ? SDRWhiteNits : HDRPaperWhiteNits;
        return max(LuminanceNitsToJz(refNits), 0.05f);
    }

    void GetRemapParams(out float jzMax, out float sigmaJz, out float detailParam, out float betaHi, out float betaSh)
    {
        jzMax       = GetActiveJzCeiling();
        sigmaJz     = max(EdgeSigma * GetReferenceWhiteJz(), 1.0e-4f);
        detailParam = DetailContrast;
        betaHi      = max(EdgeToneScale * exp2(-HighlightCompression), 0.05f);
        betaSh      = max(EdgeToneScale * exp2(-ShadowLift), 0.05f);
    }

    // Evaluates Delta_r(i; g) = r(i; g) - i in Jz space.
    // Guarantees Delta_r == 0.0 identically when DetailContrast == 0 and betaHi == betaSh == 1.
    float EvalDeltaRemap(float i, float g, float sigmaJz, float detailParam, float betaHi, float betaSh)
    {
        float d = i - g;
        // In Local Laplacian Filtering, the remapping is anchored at the local Gaussian reference g:
        //   d >= 0 (i >= g): a dark (shadow) pixel g surrounded by brighter pixels i -> controlled by betaSh.
        //                    Compressing this tail (betaSh < 1) reduces downward contrast and LIFTS shadows.
        //   d <  0 (i <  g): a bright (highlight) pixel g surrounded by darker pixels i -> controlled by betaHi.
        //                    Compressing this tail (betaHi < 1) reduces upward contrast and RECOVERS highlights.
        float beta = (d >= 0.0f) ? betaSh : betaHi;

        if (RemappingCurveMode == 0)
        {
            // Smooth C-infinity Aubry (2014) Gaussian-modulated detail + tanh-asymptotic edge slope
            float u = d / sigmaJz;
            float dr = 0.0f;
            if (detailParam != 0.0f)
            {
                float u2 = min(u * u, 180.0f);
                dr += detailParam * d * exp(-0.5f * u2);
            }
            if (beta != 1.0f)
            {
                dr += (beta - 1.0f) * (d - sigmaJz * SafeTanh(u));
            }
            return dr;
        }
        else
        {
            // Classic Paris et al. (2011) piecewise power-law detail + linear edge slope
            float ad = abs(d);
            float sgn = (d >= 0.0f) ? 1.0f : -1.0f;
            float r;
            if (ad <= sigmaJz)
            {
                float alpha = exp2(-detailParam);
                float normD = ad / sigmaJz;
                r = g + sgn * sigmaJz * SafePow(normD, alpha);
            }
            else
            {
                r = g + sgn * (beta * (ad - sigmaJz) + sigmaJz);
            }
            return r - i;
        }
    }

    float4 EvalDeltaRemapPack(float i, int packIdx, float jzStep, float sigmaJz, float detailParam, float betaHi, float betaSh)
    {
        float g0 = float(packIdx * 4 + 0) * jzStep;
        float g1 = float(packIdx * 4 + 1) * jzStep;
        float g2 = float(packIdx * 4 + 2) * jzStep;
        float g3 = float(packIdx * 4 + 3) * jzStep;
        return float4(
            EvalDeltaRemap(i, g0, sigmaJz, detailParam, betaHi, betaSh),
            EvalDeltaRemap(i, g1, sigmaJz, detailParam, betaHi, betaSh),
            EvalDeltaRemap(i, g2, sigmaJz, detailParam, betaHi, betaSh),
            EvalDeltaRemap(i, g3, sigmaJz, detailParam, betaHi, betaSh)
        );
    }

    // ---------------------------------------------------------------------------------------------
    // Exact Discrete Burt-Adelson 5-Tap Reduce & Parity-Aware Expand (IEEE 754 float32)
    // ---------------------------------------------------------------------------------------------
    // Weights: Reduce = [1, 4, 6, 4, 1] / 16 (exact dyadic floats: 0.0625, 0.25, 0.375)
    //          Expand = [1, 6, 1] / 8 on even pixels (0.125, 0.75, 0.125)
    //                   [1, 1] / 2    on odd pixels  (0.5, 0.5)

    float ReduceScalar5x5(sampler2D src, int2 dstPos, int2 srcSize)
    {
        int2 c = dstPos * 2;
        int2 maxCoord = srcSize - 1;
        float acc = 0.0f;
        [unroll]
        for (int dy = -2; dy <= 2; ++dy)
        {
            int sy = clamp(c.y + dy, 0, maxCoord.y);
            float rowAcc = 0.0f;
            [unroll]
            for (int dx = -2; dx <= 2; ++dx)
            {
                int sx = clamp(c.x + dx, 0, maxCoord.x);
                rowAcc += BURT_W5[dx + 2] * _FLLF_FETCH(src, int2(sx, sy), srcSize).r;
            }
            acc += BURT_W5[dy + 2] * rowAcc;
        }
        return acc;
    }

    float4 ReduceVec4_5x5(sampler2D src, int2 dstPos, int2 srcSize)
    {
        int2 c = dstPos * 2;
        int2 maxCoord = srcSize - 1;
        float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);
        [unroll]
        for (int dy = -2; dy <= 2; ++dy)
        {
            int sy = clamp(c.y + dy, 0, maxCoord.y);
            float4 rowAcc = float4(0.0f, 0.0f, 0.0f, 0.0f);
            [unroll]
            for (int dx = -2; dx <= 2; ++dx)
            {
                int sx = clamp(c.x + dx, 0, maxCoord.x);
                rowAcc += BURT_W5[dx + 2] * _FLLF_FETCH(src, int2(sx, sy), srcSize);
            }
            acc += BURT_W5[dy + 2] * rowAcc;
        }
        return acc;
    }

    float ExpandScalar(sampler2D coarse, int2 finePos, int2 coarseSize)
    {
        int2 k = int2(uint2(finePos) / 2u);
        int2 parity = finePos - k * 2;
        int2 maxC = coarseSize - 1;

        int x0 = clamp(k.x - 1, 0, maxC.x);
        int x1 = clamp(k.x,     0, maxC.x);
        int x2 = clamp(k.x + 1, 0, maxC.x);

        int y0 = clamp(k.y - 1, 0, maxC.y);
        int y1 = clamp(k.y,     0, maxC.y);
        int y2 = clamp(k.y + 1, 0, maxC.y);

        if (parity.y == 0)
        {
            if (parity.x == 0)
            {
                float r0 = 0.125f * _FLLF_FETCH(coarse, int2(x0, y0), coarseSize).r
                         + 0.750f * _FLLF_FETCH(coarse, int2(x1, y0), coarseSize).r
                         + 0.125f * _FLLF_FETCH(coarse, int2(x2, y0), coarseSize).r;
                float r1 = 0.125f * _FLLF_FETCH(coarse, int2(x0, y1), coarseSize).r
                         + 0.750f * _FLLF_FETCH(coarse, int2(x1, y1), coarseSize).r
                         + 0.125f * _FLLF_FETCH(coarse, int2(x2, y1), coarseSize).r;
                float r2 = 0.125f * _FLLF_FETCH(coarse, int2(x0, y2), coarseSize).r
                         + 0.750f * _FLLF_FETCH(coarse, int2(x1, y2), coarseSize).r
                         + 0.125f * _FLLF_FETCH(coarse, int2(x2, y2), coarseSize).r;
                return 0.125f * r0 + 0.750f * r1 + 0.125f * r2;
            }
            else
            {
                float r0 = 0.5f * (_FLLF_FETCH(coarse, int2(x1, y0), coarseSize).r
                                 + _FLLF_FETCH(coarse, int2(x2, y0), coarseSize).r);
                float r1 = 0.5f * (_FLLF_FETCH(coarse, int2(x1, y1), coarseSize).r
                                 + _FLLF_FETCH(coarse, int2(x2, y1), coarseSize).r);
                float r2 = 0.5f * (_FLLF_FETCH(coarse, int2(x1, y2), coarseSize).r
                                 + _FLLF_FETCH(coarse, int2(x2, y2), coarseSize).r);
                return 0.125f * r0 + 0.750f * r1 + 0.125f * r2;
            }
        }
        else
        {
            if (parity.x == 0)
            {
                float r1 = 0.125f * _FLLF_FETCH(coarse, int2(x0, y1), coarseSize).r
                         + 0.750f * _FLLF_FETCH(coarse, int2(x1, y1), coarseSize).r
                         + 0.125f * _FLLF_FETCH(coarse, int2(x2, y1), coarseSize).r;
                float r2 = 0.125f * _FLLF_FETCH(coarse, int2(x0, y2), coarseSize).r
                         + 0.750f * _FLLF_FETCH(coarse, int2(x1, y2), coarseSize).r
                         + 0.125f * _FLLF_FETCH(coarse, int2(x2, y2), coarseSize).r;
                return 0.5f * (r1 + r2);
            }
            else
            {
                float r1 = 0.5f * (_FLLF_FETCH(coarse, int2(x1, y1), coarseSize).r
                                 + _FLLF_FETCH(coarse, int2(x2, y1), coarseSize).r);
                float r2 = 0.5f * (_FLLF_FETCH(coarse, int2(x1, y2), coarseSize).r
                                 + _FLLF_FETCH(coarse, int2(x2, y2), coarseSize).r);
                return 0.5f * (r1 + r2);
            }
        }
    }

    float4 ExpandVec4(sampler2D coarse, int2 finePos, int2 coarseSize)
    {
        int2 k = int2(uint2(finePos) / 2u);
        int2 parity = finePos - k * 2;
        int2 maxC = coarseSize - 1;

        int x0 = clamp(k.x - 1, 0, maxC.x);
        int x1 = clamp(k.x,     0, maxC.x);
        int x2 = clamp(k.x + 1, 0, maxC.x);

        int y0 = clamp(k.y - 1, 0, maxC.y);
        int y1 = clamp(k.y,     0, maxC.y);
        int y2 = clamp(k.y + 1, 0, maxC.y);

        if (parity.y == 0)
        {
            if (parity.x == 0)
            {
                float4 r0 = 0.125f * _FLLF_FETCH(coarse, int2(x0, y0), coarseSize)
                          + 0.750f * _FLLF_FETCH(coarse, int2(x1, y0), coarseSize)
                          + 0.125f * _FLLF_FETCH(coarse, int2(x2, y0), coarseSize);
                float4 r1 = 0.125f * _FLLF_FETCH(coarse, int2(x0, y1), coarseSize)
                          + 0.750f * _FLLF_FETCH(coarse, int2(x1, y1), coarseSize)
                          + 0.125f * _FLLF_FETCH(coarse, int2(x2, y1), coarseSize);
                float4 r2 = 0.125f * _FLLF_FETCH(coarse, int2(x0, y2), coarseSize)
                          + 0.750f * _FLLF_FETCH(coarse, int2(x1, y2), coarseSize)
                          + 0.125f * _FLLF_FETCH(coarse, int2(x2, y2), coarseSize);
                return 0.125f * r0 + 0.750f * r1 + 0.125f * r2;
            }
            else
            {
                float4 r0 = 0.5f * (_FLLF_FETCH(coarse, int2(x1, y0), coarseSize)
                                  + _FLLF_FETCH(coarse, int2(x2, y0), coarseSize));
                float4 r1 = 0.5f * (_FLLF_FETCH(coarse, int2(x1, y1), coarseSize)
                                  + _FLLF_FETCH(coarse, int2(x2, y1), coarseSize));
                float4 r2 = 0.5f * (_FLLF_FETCH(coarse, int2(x1, y2), coarseSize)
                                  + _FLLF_FETCH(coarse, int2(x2, y2), coarseSize));
                return 0.125f * r0 + 0.750f * r1 + 0.125f * r2;
            }
        }
        else
        {
            if (parity.x == 0)
            {
                float4 r1 = 0.125f * _FLLF_FETCH(coarse, int2(x0, y1), coarseSize)
                          + 0.750f * _FLLF_FETCH(coarse, int2(x1, y1), coarseSize)
                          + 0.125f * _FLLF_FETCH(coarse, int2(x2, y1), coarseSize);
                float4 r2 = 0.125f * _FLLF_FETCH(coarse, int2(x0, y2), coarseSize)
                          + 0.750f * _FLLF_FETCH(coarse, int2(x1, y2), coarseSize)
                          + 0.125f * _FLLF_FETCH(coarse, int2(x2, y2), coarseSize);
                return 0.5f * (r1 + r2);
            }
            else
            {
                float4 r1 = 0.5f * (_FLLF_FETCH(coarse, int2(x1, y1), coarseSize)
                                  + _FLLF_FETCH(coarse, int2(x2, y1), coarseSize));
                float4 r2 = 0.5f * (_FLLF_FETCH(coarse, int2(x1, y2), coarseSize)
                                  + _FLLF_FETCH(coarse, int2(x2, y2), coarseSize));
                return 0.5f * (r1 + r2);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Intensity Slice Indexing & Catmull-Rom / Linear Interpolation Helpers
    // ---------------------------------------------------------------------------------------------

    float ExtractChannel(float4 v, int c)
    {
        float res;
        if (c == 0)
            res = v.x;
        else if (c == 1)
            res = v.y;
        else if (c == 2)
            res = v.z;
        else
            res = v.w;
        return res;
    }

    float CatmullRom(float p0, float p1, float p2, float p3, float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        return 0.5f * (
            (2.0f * p1)
            + (-p0 + p2) * t
            + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2
            + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3
        );
    }

    // Fetch a single Laplacian coefficient L_l[Delta_R_k] from only the needed 1 or 2 MRT packs
    float4 FetchLaplacianPack(
        int packIdx,
        sampler2D sFine0, sampler2D sFine1,
    #if _FLLF_MRT_COUNT >= 3
        sampler2D sFine2,
    #endif
    #if _FLLF_MRT_COUNT >= 4
        sampler2D sFine3,
    #endif
        sampler2D sCoarse0, sampler2D sCoarse1,
    #if _FLLF_MRT_COUNT >= 3
        sampler2D sCoarse2,
    #endif
    #if _FLLF_MRT_COUNT >= 4
        sampler2D sCoarse3,
    #endif
        int2 finePos, int2 fineSize, int2 coarseSize)
    {
        float4 res;
        if (packIdx == 0)
            res = _FLLF_FETCH(sFine0, finePos, fineSize) - ExpandVec4(sCoarse0, finePos, coarseSize);
    #if _FLLF_MRT_COUNT == 2
        else
            res = _FLLF_FETCH(sFine1, finePos, fineSize) - ExpandVec4(sCoarse1, finePos, coarseSize);
    #elif _FLLF_MRT_COUNT == 3
        else if (packIdx == 1)
            res = _FLLF_FETCH(sFine1, finePos, fineSize) - ExpandVec4(sCoarse1, finePos, coarseSize);
        else
            res = _FLLF_FETCH(sFine2, finePos, fineSize) - ExpandVec4(sCoarse2, finePos, coarseSize);
    #else
        else if (packIdx == 1)
            res = _FLLF_FETCH(sFine1, finePos, fineSize) - ExpandVec4(sCoarse1, finePos, coarseSize);
        else if (packIdx == 2)
            res = _FLLF_FETCH(sFine2, finePos, fineSize) - ExpandVec4(sCoarse2, finePos, coarseSize);
        else
            res = _FLLF_FETCH(sFine3, finePos, fineSize) - ExpandVec4(sCoarse3, finePos, coarseSize);
    #endif
        return res;
    }

    float InterpolateLaplacianBand(
        float g,
        sampler2D sFine0, sampler2D sFine1,
    #if _FLLF_MRT_COUNT >= 3
        sampler2D sFine2,
    #endif
    #if _FLLF_MRT_COUNT >= 4
        sampler2D sFine3,
    #endif
        sampler2D sCoarse0, sampler2D sCoarse1,
    #if _FLLF_MRT_COUNT >= 3
        sampler2D sCoarse2,
    #endif
    #if _FLLF_MRT_COUNT >= 4
        sampler2D sCoarse3,
    #endif
        int2 finePos, int2 fineSize, int2 coarseSize,
        float bandWeight)
    {
        float jzMax = GetActiveJzCeiling();
        float pos = saturate(g / jzMax) * float(_FLLF_SLICES - 1);
        int idx = min(int(floor(pos)), _FLLF_SLICES - 2);
        float t = pos - float(idx);

        if (SliceInterpolationMode == 1)
        {
            // Classic Aubry piecewise linear interpolation (touches at most 2 packs)
            int i1 = idx;
            int i2 = idx + 1;
            int pLow  = int(uint(i1) / 4u);
            int pHigh = int(uint(i2) / 4u);

            float4 packLow = FetchLaplacianPack(
                pLow,
                sFine0, sFine1,
            #if _FLLF_MRT_COUNT >= 3
                sFine2,
            #endif
            #if _FLLF_MRT_COUNT >= 4
                sFine3,
            #endif
                sCoarse0, sCoarse1,
            #if _FLLF_MRT_COUNT >= 3
                sCoarse2,
            #endif
            #if _FLLF_MRT_COUNT >= 4
                sCoarse3,
            #endif
                finePos, fineSize, coarseSize);

            float4 packHigh = packLow;
            [branch]
            if (pHigh != pLow)
            {
                packHigh = FetchLaplacianPack(
                    pHigh,
                    sFine0, sFine1,
                #if _FLLF_MRT_COUNT >= 3
                    sFine2,
                #endif
                #if _FLLF_MRT_COUNT >= 4
                    sFine3,
                #endif
                    sCoarse0, sCoarse1,
                #if _FLLF_MRT_COUNT >= 3
                    sCoarse2,
                #endif
                #if _FLLF_MRT_COUNT >= 4
                    sCoarse3,
                #endif
                    finePos, fineSize, coarseSize);
            }

            float v1 = ExtractChannel(packLow,  i1 - pLow  * 4);
            float v2 = ExtractChannel(packHigh, i2 - pHigh * 4);
            return lerp(v1, v2, t) * bandWeight;
        }
        else
        {
            // Catmull-Rom cubic interpolation (4 consecutive slices -> at most 2 MRT packs!)
            int i0 = max(idx - 1, 0);
            int i1 = idx;
            int i2 = idx + 1;
            int i3 = min(idx + 2, _FLLF_SLICES - 1);

            int p0 = int(uint(i0) / 4u);
            int p1 = int(uint(i1) / 4u);
            int p2 = int(uint(i2) / 4u);
            int p3 = int(uint(i3) / 4u);

            float4 packLow = FetchLaplacianPack(
                p0,
                sFine0, sFine1,
            #if _FLLF_MRT_COUNT >= 3
                sFine2,
            #endif
            #if _FLLF_MRT_COUNT >= 4
                sFine3,
            #endif
                sCoarse0, sCoarse1,
            #if _FLLF_MRT_COUNT >= 3
                sCoarse2,
            #endif
            #if _FLLF_MRT_COUNT >= 4
                sCoarse3,
            #endif
                finePos, fineSize, coarseSize);

            float4 packHigh = packLow;
            [branch]
            if (p3 != p0)
            {
                packHigh = FetchLaplacianPack(
                    p3,
                    sFine0, sFine1,
                #if _FLLF_MRT_COUNT >= 3
                    sFine2,
                #endif
                #if _FLLF_MRT_COUNT >= 4
                    sFine3,
                #endif
                    sCoarse0, sCoarse1,
                #if _FLLF_MRT_COUNT >= 3
                    sCoarse2,
                #endif
                #if _FLLF_MRT_COUNT >= 4
                    sCoarse3,
                #endif
                    finePos, fineSize, coarseSize);
            }

            float v0 = ExtractChannel(packLow,  i0 - p0 * 4);
            float v1 = ExtractChannel((p1 == p0) ? packLow : packHigh, i1 - p1 * 4);
            float v2 = ExtractChannel((p2 == p0) ? packLow : packHigh, i2 - p2 * 4);
            float v3 = ExtractChannel(packHigh, i3 - p3 * 4);
            return CatmullRom(v0, v1, v2, v3, t) * bandWeight;
        }
    }

    // Expand one MRT pack of G1_R at full resolution (used by fused L0 final pass)
    float4 ExpandG1Pack(int packIdx, int2 finePos)
    {
        const int2 coarseSize = int2(_FLLF_W1, _FLLF_H1);
        float4 res;
        if (packIdx == 0)
            res = ExpandVec4(SampG1_R_0, finePos, coarseSize);
    #if _FLLF_MRT_COUNT == 2
        else
            res = ExpandVec4(SampG1_R_1, finePos, coarseSize);
    #elif _FLLF_MRT_COUNT == 3
        else if (packIdx == 1)
            res = ExpandVec4(SampG1_R_1, finePos, coarseSize);
        else
            res = ExpandVec4(SampG1_R_2, finePos, coarseSize);
    #else
        else if (packIdx == 1)
            res = ExpandVec4(SampG1_R_1, finePos, coarseSize);
        else if (packIdx == 2)
            res = ExpandVec4(SampG1_R_2, finePos, coarseSize);
        else
            res = ExpandVec4(SampG1_R_3, finePos, coarseSize);
    #endif
        return res;
    }

    // Interleaved Gradient Noise (Jimenez 2014) -> zero-mean TPDF dithering in [-1, +1]
    float TPDFNoise(float2 pos)
    {
        float r0 = frac(52.9829189f * frac(dot(pos, float2(0.06711056f, 0.00583715f))));
        float r1 = frac(52.9829189f * frac(dot(pos + float2(17.0f, 31.0f), float2(0.06711056f, 0.00583715f))));
        return r0 - r1;
    }

    // ---------------------------------------------------------------------------------------------
    // Vertex Shader (Self-Contained Fullscreen Triangle)
    // ---------------------------------------------------------------------------------------------

    void PostProcessVS(uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD0)
    {
        texcoord.x = (id == 2) ? 2.0f : 0.0f;
        texcoord.y = (id == 1) ? 2.0f : 0.0f;
        position = float4(texcoord * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    }

    // ---------------------------------------------------------------------------------------------
    // Phase 1: BackBuffer -> Full-Precision JzAzBz (RGBA32F)
    // ---------------------------------------------------------------------------------------------

    float4 PS_ToJzAzBz(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W0, _FLLF_H0));
        float4 raw = _FLLF_FETCH(SampBackBuffer, p, int2(_FLLF_W0, _FLLF_H0));
        float3 jab = BackBufferToJzAzBz(raw.rgb);
        return float4(jab, raw.a);
    }

    // Exact 4-pass 8x8 full-frame HDR peak Jz reduction (instant 0-fetch bypass in SDR or manual mode)
    float ReduceMax8x8(sampler2D src, int2 dstPos, int2 srcSize)
    {
        int2 base = dstPos * 8;
        float m = 0.0f;
        [loop]
        for (int dy = 0; dy < 8; ++dy)
        {
            int sy = base.y + dy;
            if (sy < srcSize.y)
            {
                [loop]
                for (int dx = 0; dx < 8; ++dx)
                {
                    int sx = base.x + dx;
                    if (sx < srcSize.x)
                        m = max(m, _FLLF_FETCH(src, int2(sx, sy), srcSize).r);
                }
            }
        }
        return m;
    }

    float PS_Peak_1(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        float res = 0.0f;
        if (AutoDetectHDRPeak && GetActiveColorSpace() != 1)
        {
            int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_PW1, _FLLF_PH1));
            res = ReduceMax8x8(SampJzAzBz, p, int2(_FLLF_W0, _FLLF_H0));
        }
        return res;
    }

    float PS_Peak_2(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        float res = 0.0f;
        if (AutoDetectHDRPeak && GetActiveColorSpace() != 1)
        {
            int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_PW2, _FLLF_PH2));
            res = ReduceMax8x8(SampPeak1, p, int2(_FLLF_PW1, _FLLF_PH1));
        }
        return res;
    }

    float PS_Peak_3(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        float res = 0.0f;
        if (AutoDetectHDRPeak && GetActiveColorSpace() != 1)
        {
            int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_PW3, _FLLF_PH3));
            res = ReduceMax8x8(SampPeak2, p, int2(_FLLF_PW2, _FLLF_PH2));
        }
        return res;
    }

    float PS_Peak_4(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        float m = 0.0f;
        if (AutoDetectHDRPeak && GetActiveColorSpace() != 1)
        {
            const int2 srcSize = int2(_FLLF_PW3, _FLLF_PH3);
        #if _FLLF_PH3 == 1
            int sy = 0;
        #else
            [loop]
            for (int sy = 0; sy < _FLLF_PH3; ++sy)
        #endif
            {
            #if _FLLF_PW3 == 1
                int sx = 0;
            #else
                [loop]
                for (int sx = 0; sx < _FLLF_PW3; ++sx)
            #endif
                {
                    m = max(m, _FLLF_FETCH(SampPeak3, int2(sx, sy), srcSize).r);
                }
            }
        }
        return m;
    }

    // ---------------------------------------------------------------------------------------------
    // Phase 2: Scalar Jz Gaussian Pyramid (G1 .. G{L-2})
    // ---------------------------------------------------------------------------------------------

    float PS_ReduceJz_01(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W1, _FLLF_H1));
        return ReduceScalar5x5(SampJzAzBz, p, int2(_FLLF_W0, _FLLF_H0));
    }

    float PS_ReduceJz_12(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W2, _FLLF_H2));
        return ReduceScalar5x5(SampG1, p, int2(_FLLF_W1, _FLLF_H1));
    }

    float PS_ReduceJz_23(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W3, _FLLF_H3));
        return ReduceScalar5x5(SampG2, p, int2(_FLLF_W2, _FLLF_H2));
    }

#if _FLLF_LEVELS >= 6
    float PS_ReduceJz_34(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W4, _FLLF_H4));
        return ReduceScalar5x5(SampG3, p, int2(_FLLF_W3, _FLLF_H3));
    }
#endif

#if _FLLF_LEVELS >= 7
    float PS_ReduceJz_45(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W5, _FLLF_H5));
        return ReduceScalar5x5(SampG4, p, int2(_FLLF_W4, _FLLF_H4));
    }
#endif

#if _FLLF_LEVELS >= 8
    float PS_ReduceJz_56(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W6, _FLLF_H6));
        return ReduceScalar5x5(SampG5, p, int2(_FLLF_W5, _FLLF_H5));
    }
#endif

#if _FLLF_LEVELS >= 9
    float PS_ReduceJz_67(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W7, _FLLF_H7));
        return ReduceScalar5x5(SampG6, p, int2(_FLLF_W6, _FLLF_H6));
    }
#endif

    // ---------------------------------------------------------------------------------------------
    // Phase 3: Remapped Delta Gaussian Pyramid Packs G1[Delta_R] .. G{L-1}[Delta_R]
    // ---------------------------------------------------------------------------------------------

    // Level 0 -> 1 Horizontal Separable Pass (computes Delta_R on the fly across 5 taps -> 5x less ALU)
    void PS_ReduceR01_H(
        float4 vpos : SV_Position,
        float2 uv   : TEXCOORD0,
        out float4 o0 : SV_Target0,
        out float4 o1 : SV_Target1
    #if _FLLF_MRT_COUNT >= 3
        , out float4 o2 : SV_Target2
    #endif
    #if _FLLF_MRT_COUNT >= 4
        , out float4 o3 : SV_Target3
    #endif
    )
    {
        int2 dstSize = int2(_FLLF_W1, _FLLF_H0);
        int2 srcSize = int2(_FLLF_W0, _FLLF_H0);
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, dstSize);

        float jzMax, sigmaJz, detailParam, betaHi, betaSh;
        GetRemapParams(jzMax, sigmaJz, detailParam, betaHi, betaSh);
        float jzStep = jzMax / float(_FLLF_SLICES - 1);

        o0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #if _FLLF_MRT_COUNT >= 3
        o2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #endif
    #if _FLLF_MRT_COUNT >= 4
        o3 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #endif

        int cx = p.x * 2;
        int maxX = srcSize.x - 1;
        [unroll]
        for (int dx = -2; dx <= 2; ++dx)
        {
            int sx = clamp(cx + dx, 0, maxX);
            float jz = _FLLF_FETCH(SampJzAzBz, int2(sx, p.y), srcSize).r;
            float wt = BURT_W5[dx + 2];

            o0 += wt * EvalDeltaRemapPack(jz, 0, jzStep, sigmaJz, detailParam, betaHi, betaSh);
            o1 += wt * EvalDeltaRemapPack(jz, 1, jzStep, sigmaJz, detailParam, betaHi, betaSh);
        #if _FLLF_MRT_COUNT >= 3
            o2 += wt * EvalDeltaRemapPack(jz, 2, jzStep, sigmaJz, detailParam, betaHi, betaSh);
        #endif
        #if _FLLF_MRT_COUNT >= 4
            o3 += wt * EvalDeltaRemapPack(jz, 3, jzStep, sigmaJz, detailParam, betaHi, betaSh);
        #endif
        }
    }

    // Level 0 -> 1 Vertical Separable Pass (pure linear 5-tap average)
    void PS_ReduceR01_V(
        float4 vpos : SV_Position,
        float2 uv   : TEXCOORD0,
        out float4 o0 : SV_Target0,
        out float4 o1 : SV_Target1
    #if _FLLF_MRT_COUNT >= 3
        , out float4 o2 : SV_Target2
    #endif
    #if _FLLF_MRT_COUNT >= 4
        , out float4 o3 : SV_Target3
    #endif
    )
    {
        int2 dstSize = int2(_FLLF_W1, _FLLF_H1);
        int2 srcSize = int2(_FLLF_W1, _FLLF_H0);
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, dstSize);

        o0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #if _FLLF_MRT_COUNT >= 3
        o2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #endif
    #if _FLLF_MRT_COUNT >= 4
        o3 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #endif

        int cy = p.y * 2;
        int maxY = srcSize.y - 1;
        [unroll]
        for (int dy = -2; dy <= 2; ++dy)
        {
            int sy = clamp(cy + dy, 0, maxY);
            float wt = BURT_W5[dy + 2];
            o0 += wt * _FLLF_FETCH(SampH1_R_0, int2(p.x, sy), srcSize);
            o1 += wt * _FLLF_FETCH(SampH1_R_1, int2(p.x, sy), srcSize);
        #if _FLLF_MRT_COUNT >= 3
            o2 += wt * _FLLF_FETCH(SampH1_R_2, int2(p.x, sy), srcSize);
        #endif
        #if _FLLF_MRT_COUNT >= 4
            o3 += wt * _FLLF_FETCH(SampH1_R_3, int2(p.x, sy), srcSize);
        #endif
        }
    }

    // Macro to generate linear 5x5 MRT Reduce shaders for G{L}_R -> G{L+1}_R
#if _FLLF_MRT_COUNT == 2
    #define _FLLF_DEF_REDUCE_R(L_SRC, L_DST, WS, HS, WD, HD) \
        void PS_ReduceR_##L_SRC##L_DST(float4 vpos : SV_Position, float2 uv : TEXCOORD0, \
            out float4 o0 : SV_Target0, out float4 o1 : SV_Target1) \
        { \
            int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(WD, HD)); \
            o0 = ReduceVec4_5x5(SampG##L_SRC##_R_0, p, int2(WS, HS)); \
            o1 = ReduceVec4_5x5(SampG##L_SRC##_R_1, p, int2(WS, HS)); \
        }
#elif _FLLF_MRT_COUNT == 3
    #define _FLLF_DEF_REDUCE_R(L_SRC, L_DST, WS, HS, WD, HD) \
        void PS_ReduceR_##L_SRC##L_DST(float4 vpos : SV_Position, float2 uv : TEXCOORD0, \
            out float4 o0 : SV_Target0, out float4 o1 : SV_Target1, out float4 o2 : SV_Target2) \
        { \
            int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(WD, HD)); \
            o0 = ReduceVec4_5x5(SampG##L_SRC##_R_0, p, int2(WS, HS)); \
            o1 = ReduceVec4_5x5(SampG##L_SRC##_R_1, p, int2(WS, HS)); \
            o2 = ReduceVec4_5x5(SampG##L_SRC##_R_2, p, int2(WS, HS)); \
        }
#else
    #define _FLLF_DEF_REDUCE_R(L_SRC, L_DST, WS, HS, WD, HD) \
        void PS_ReduceR_##L_SRC##L_DST(float4 vpos : SV_Position, float2 uv : TEXCOORD0, \
            out float4 o0 : SV_Target0, out float4 o1 : SV_Target1, out float4 o2 : SV_Target2, out float4 o3 : SV_Target3) \
        { \
            int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(WD, HD)); \
            o0 = ReduceVec4_5x5(SampG##L_SRC##_R_0, p, int2(WS, HS)); \
            o1 = ReduceVec4_5x5(SampG##L_SRC##_R_1, p, int2(WS, HS)); \
            o2 = ReduceVec4_5x5(SampG##L_SRC##_R_2, p, int2(WS, HS)); \
            o3 = ReduceVec4_5x5(SampG##L_SRC##_R_3, p, int2(WS, HS)); \
        }
#endif

    _FLLF_DEF_REDUCE_R(1, 2, _FLLF_W1, _FLLF_H1, _FLLF_W2, _FLLF_H2)
    _FLLF_DEF_REDUCE_R(2, 3, _FLLF_W2, _FLLF_H2, _FLLF_W3, _FLLF_H3)
    _FLLF_DEF_REDUCE_R(3, 4, _FLLF_W3, _FLLF_H3, _FLLF_W4, _FLLF_H4)
#if _FLLF_LEVELS >= 6
    _FLLF_DEF_REDUCE_R(4, 5, _FLLF_W4, _FLLF_H4, _FLLF_W5, _FLLF_H5)
#endif
#if _FLLF_LEVELS >= 7
    _FLLF_DEF_REDUCE_R(5, 6, _FLLF_W5, _FLLF_H5, _FLLF_W6, _FLLF_H6)
#endif
#if _FLLF_LEVELS >= 8
    _FLLF_DEF_REDUCE_R(6, 7, _FLLF_W6, _FLLF_H6, _FLLF_W7, _FLLF_H7)
#endif
#if _FLLF_LEVELS >= 9
    _FLLF_DEF_REDUCE_R(7, 8, _FLLF_W7, _FLLF_H7, _FLLF_W8, _FLLF_H8)
#endif

    // ---------------------------------------------------------------------------------------------
    // Phase 4: Interpolate Output Laplacian Delta Bands L1 .. L{L-2}
    // ---------------------------------------------------------------------------------------------

#if _FLLF_MRT_COUNT == 2
    #define _FLLF_DEF_INTERP_L(LVL, NEXT, WF, HF, WC, HC, WEIGHT) \
        float PS_InterpL_##LVL(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target \
        { \
            int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(WF, HF)); \
            float g = _FLLF_FETCH(SampG##LVL, p, int2(WF, HF)).r; \
            return InterpolateLaplacianBand(g, \
                SampG##LVL##_R_0, SampG##LVL##_R_1, \
                SampG##NEXT##_R_0, SampG##NEXT##_R_1, \
                p, int2(WF, HF), int2(WC, HC), WEIGHT); \
        }
#elif _FLLF_MRT_COUNT == 3
    #define _FLLF_DEF_INTERP_L(LVL, NEXT, WF, HF, WC, HC, WEIGHT) \
        float PS_InterpL_##LVL(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target \
        { \
            int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(WF, HF)); \
            float g = _FLLF_FETCH(SampG##LVL, p, int2(WF, HF)).r; \
            return InterpolateLaplacianBand(g, \
                SampG##LVL##_R_0, SampG##LVL##_R_1, SampG##LVL##_R_2, \
                SampG##NEXT##_R_0, SampG##NEXT##_R_1, SampG##NEXT##_R_2, \
                p, int2(WF, HF), int2(WC, HC), WEIGHT); \
        }
#else
    #define _FLLF_DEF_INTERP_L(LVL, NEXT, WF, HF, WC, HC, WEIGHT) \
        float PS_InterpL_##LVL(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target \
        { \
            int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(WF, HF)); \
            float g = _FLLF_FETCH(SampG##LVL, p, int2(WF, HF)).r; \
            return InterpolateLaplacianBand(g, \
                SampG##LVL##_R_0, SampG##LVL##_R_1, SampG##LVL##_R_2, SampG##LVL##_R_3, \
                SampG##NEXT##_R_0, SampG##NEXT##_R_1, SampG##NEXT##_R_2, SampG##NEXT##_R_3, \
                p, int2(WF, HF), int2(WC, HC), WEIGHT); \
        }
#endif

    _FLLF_DEF_INTERP_L(1, 2, _FLLF_W1, _FLLF_H1, _FLLF_W2, _FLLF_H2, ScaleWeightFine)
    _FLLF_DEF_INTERP_L(2, 3, _FLLF_W2, _FLLF_H2, _FLLF_W3, _FLLF_H3, ScaleWeightMedium)
    _FLLF_DEF_INTERP_L(3, 4, _FLLF_W3, _FLLF_H3, _FLLF_W4, _FLLF_H4, ScaleWeightMedium)
#if _FLLF_LEVELS >= 6
    _FLLF_DEF_INTERP_L(4, 5, _FLLF_W4, _FLLF_H4, _FLLF_W5, _FLLF_H5, ScaleWeightMedium)
#endif
#if _FLLF_LEVELS >= 7
    _FLLF_DEF_INTERP_L(5, 6, _FLLF_W5, _FLLF_H5, _FLLF_W6, _FLLF_H6, ScaleWeightCoarse)
#endif
#if _FLLF_LEVELS >= 8
    _FLLF_DEF_INTERP_L(6, 7, _FLLF_W6, _FLLF_H6, _FLLF_W7, _FLLF_H7, ScaleWeightCoarse)
#endif
#if _FLLF_LEVELS >= 9
    _FLLF_DEF_INTERP_L(7, 8, _FLLF_W7, _FLLF_H7, _FLLF_W8, _FLLF_H8, ScaleWeightCoarse)
#endif

    // ---------------------------------------------------------------------------------------------
    // Phase 5: Bottom-Up Laplacian Pyramid Collapse (O_{L-2} -> ... -> O_1)
    // Reuses TexG{l} as O_{l} since G{l}[Jz] is no longer needed after Phase 4!
    // Coarsest residual Delta_O_{L-1} = 0 (exact Paris 2011 / Aubry 2014 global mean anchoring).
    // ---------------------------------------------------------------------------------------------

#if _FLLF_LEVELS >= 9
    float PS_Collapse_76(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W6, _FLLF_H6));
        return _FLLF_FETCH(SampL6, p, int2(_FLLF_W6, _FLLF_H6)).r + ExpandScalar(SampL7, p, int2(_FLLF_W7, _FLLF_H7));
    }
#endif

#if _FLLF_LEVELS >= 8
    float PS_Collapse_65(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W5, _FLLF_H5));
    #if _FLLF_LEVELS == 8
        return _FLLF_FETCH(SampL5, p, int2(_FLLF_W5, _FLLF_H5)).r + ExpandScalar(SampL6, p, int2(_FLLF_W6, _FLLF_H6));
    #else
        return _FLLF_FETCH(SampL5, p, int2(_FLLF_W5, _FLLF_H5)).r + ExpandScalar(SampG6, p, int2(_FLLF_W6, _FLLF_H6));
    #endif
    }
#endif

#if _FLLF_LEVELS >= 7
    float PS_Collapse_54(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W4, _FLLF_H4));
    #if _FLLF_LEVELS == 7
        return _FLLF_FETCH(SampL4, p, int2(_FLLF_W4, _FLLF_H4)).r + ExpandScalar(SampL5, p, int2(_FLLF_W5, _FLLF_H5));
    #else
        return _FLLF_FETCH(SampL4, p, int2(_FLLF_W4, _FLLF_H4)).r + ExpandScalar(SampG5, p, int2(_FLLF_W5, _FLLF_H5));
    #endif
    }
#endif

#if _FLLF_LEVELS >= 6
    float PS_Collapse_43(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W3, _FLLF_H3));
    #if _FLLF_LEVELS == 6
        return _FLLF_FETCH(SampL3, p, int2(_FLLF_W3, _FLLF_H3)).r + ExpandScalar(SampL4, p, int2(_FLLF_W4, _FLLF_H4));
    #else
        return _FLLF_FETCH(SampL3, p, int2(_FLLF_W3, _FLLF_H3)).r + ExpandScalar(SampG4, p, int2(_FLLF_W4, _FLLF_H4));
    #endif
    }
#endif

    float PS_Collapse_32(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W2, _FLLF_H2));
    #if _FLLF_LEVELS == 5
        return _FLLF_FETCH(SampL2, p, int2(_FLLF_W2, _FLLF_H2)).r + ExpandScalar(SampL3, p, int2(_FLLF_W3, _FLLF_H3));
    #else
        return _FLLF_FETCH(SampL2, p, int2(_FLLF_W2, _FLLF_H2)).r + ExpandScalar(SampG3, p, int2(_FLLF_W3, _FLLF_H3));
    #endif
    }

    float PS_Collapse_21(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W1, _FLLF_H1));
        return _FLLF_FETCH(SampL1, p, int2(_FLLF_W1, _FLLF_H1)).r + ExpandScalar(SampG2, p, int2(_FLLF_W2, _FLLF_H2));
    }

    // ---------------------------------------------------------------------------------------------
    // Phase 6: Fused Full-Res L0 Interpolation + Final Collapse + JzAzBz -> SDR/HDR Output
    // ---------------------------------------------------------------------------------------------

    float4 PS_Reconstruct(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _FLLF_PIXEL_COORD(vpos, uv, int2(_FLLF_W0, _FLLF_H0));
        float4 origPixel = _FLLF_FETCH(SampBackBuffer, p, int2(_FLLF_W0, _FLLF_H0));

        // Split-screen comparison: left side returns bit-exact untouched pixel
        if (SplitScreenCompare && (uv.x < SplitPosition))
        {
            if (abs(uv.x - SplitPosition) < (BUFFER_RCP_WIDTH * 1.5f))
                return float4(0.5f, 0.5f, 0.5f, origPixel.a);
            return origPixel;
        }

        // Exact neutral bypass: zero rounding error when filter has no active effect
        bool isNeutral = (FilterBlend <= 0.0f)
                      || (DetailContrast == 0.0f && EdgeToneScale == 1.0f
                          && HighlightCompression == 0.0f && ShadowLift == 0.0f);
        if (isNeutral && DiagnosticView == 0)
            return origPixel;

        float4 jabA = _FLLF_FETCH(SampJzAzBz, p, int2(_FLLF_W0, _FLLF_H0));
        float JzOrig = jabA.r;

        // Compute Level-0 Laplacian delta L0_out on the fly using ONLY the 1 or 2 needed MRT packs
        float jzMax, sigmaJz, detailParam, betaHi, betaSh;
        GetRemapParams(jzMax, sigmaJz, detailParam, betaHi, betaSh);
        float jzStep = jzMax / float(_FLLF_SLICES - 1);

        float pos = saturate(JzOrig / jzMax) * float(_FLLF_SLICES - 1);
        int idx = min(int(floor(pos)), _FLLF_SLICES - 2);
        float t = pos - float(idx);

        float L0_out;
        if (SliceInterpolationMode == 1)
        {
            int i1 = idx;
            int i2 = idx + 1;
            int pLow  = int(uint(i1) / 4u);
            int pHigh = int(uint(i2) / 4u);

            float4 expLow  = ExpandG1Pack(pLow, p);
            float4 expHigh = expLow;
            [branch]
            if (pHigh != pLow)
                expHigh = ExpandG1Pack(pHigh, p);

            float v1 = EvalDeltaRemap(JzOrig, float(i1) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel(expLow, i1 - pLow * 4);
            float v2 = EvalDeltaRemap(JzOrig, float(i2) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel(expHigh, i2 - pHigh * 4);

            L0_out = lerp(v1, v2, t) * ScaleWeightFine;
        }
        else
        {
            int i0 = max(idx - 1, 0);
            int i1 = idx;
            int i2 = idx + 1;
            int i3 = min(idx + 2, _FLLF_SLICES - 1);

            int p0 = int(uint(i0) / 4u);
            int p1 = int(uint(i1) / 4u);
            int p2 = int(uint(i2) / 4u);
            int p3 = int(uint(i3) / 4u);

            float4 expLow  = ExpandG1Pack(p0, p);
            float4 expHigh = expLow;
            [branch]
            if (p3 != p0)
                expHigh = ExpandG1Pack(p3, p);

            float v0 = EvalDeltaRemap(JzOrig, float(i0) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel(expLow, i0 - p0 * 4);
            float v1 = EvalDeltaRemap(JzOrig, float(i1) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel((p1 == p0) ? expLow : expHigh, i1 - p1 * 4);
            float v2 = EvalDeltaRemap(JzOrig, float(i2) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel((p2 == p0) ? expLow : expHigh, i2 - p2 * 4);
            float v3 = EvalDeltaRemap(JzOrig, float(i3) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel(expHigh, i3 - p3 * 4);

            L0_out = CatmullRom(v0, v1, v2, v3, t) * ScaleWeightFine;
        }

        // Complete collapse: Delta_Jz = L0_out + Expand(O1)
        float expandO1 = ExpandScalar(SampG1, p, int2(_FLLF_W1, _FLLF_H1));
        float deltaJz = (L0_out + expandO1) * FilterBlend;
        float minJz = (GamutMappingMode == 3 && GetActiveColorSpace() == 2) ? -0.28f : 0.0f;
        float JzNew = max(JzOrig + deltaJz, minJz);

        // Diagnostic visualization modes (pure D65 grayscale mapped to reference white nits)
        if (DiagnosticView == 1)
        {
            float visJz = saturate(JzNew / jzMax) * GetReferenceWhiteJz();
            float visNits = JzToLuminanceNits(visJz);
            float3 visRgb = EncodeLinearNitsToBackBuffer(float3(visNits, visNits, visNits), GetActiveColorSpace(), false);
            return float4(visRgb, origPixel.a);
        }
        if (DiagnosticView == 2)
        {
            float visDelta = saturate(0.5f + 5.0f * (deltaJz / GetReferenceWhiteJz())) * GetReferenceWhiteJz();
            float visNits = JzToLuminanceNits(visDelta);
            float3 visRgb = EncodeLinearNitsToBackBuffer(float3(visNits, visNits, visNits), GetActiveColorSpace(), false);
            return float4(visRgb, origPixel.a);
        }

        // Unified color reconstruction + adaptive gamut mapping (no 1e-9 threshold -> zero flat-UI speckle
        // and spatially uniform TPDF dithering across the entire active frame)
        float3 rgbOut = ReconstructColorToBackBuffer(origPixel.rgb, jabA.rgb, JzNew);

        // Optional TPDF 1-LSB dithering for 8-bit & 10-bit integer swapchains (auto-off for FP16 scRGB)
    #if defined(BUFFER_COLOR_BIT_DEPTH) && ((BUFFER_COLOR_BIT_DEPTH == 8) || (BUFFER_COLOR_BIT_DEPTH == 10))
        if (EnableOutputDithering && (GetActiveColorSpace() != 2))
        {
        #if BUFFER_COLOR_BIT_DEPTH == 10
            const float lsb = 1.0f / 1023.0f;
        #else
            const float lsb = 1.0f / 255.0f;
        #endif
            float dither = TPDFNoise(float2(p)) * lsb;
            rgbOut = saturate(rgbOut + dither);
        }
    #endif

        return float4(rgbOut, origPixel.a);
    }

    // ---------------------------------------------------------------------------------------------
    // Pass MRT Target Binding Helper Macro
    // ---------------------------------------------------------------------------------------------
#if _FLLF_MRT_COUNT == 2
    #define _FLLF_BIND_MRT(NAME) \
        RenderTarget0 = Tex##NAME##_0; \
        RenderTarget1 = Tex##NAME##_1;
#elif _FLLF_MRT_COUNT == 3
    #define _FLLF_BIND_MRT(NAME) \
        RenderTarget0 = Tex##NAME##_0; \
        RenderTarget1 = Tex##NAME##_1; \
        RenderTarget2 = Tex##NAME##_2;
#else
    #define _FLLF_BIND_MRT(NAME) \
        RenderTarget0 = Tex##NAME##_0; \
        RenderTarget1 = Tex##NAME##_1; \
        RenderTarget2 = Tex##NAME##_2; \
        RenderTarget3 = Tex##NAME##_3;
#endif

    // ---------------------------------------------------------------------------------------------
    // Technique Definition
    // ---------------------------------------------------------------------------------------------

    technique FastLocalLaplacian_JzAzBz <
        ui_label = "Fast Local Laplacian Filter (JzAzBz HDR/SDR)";
        ui_tooltip = "Maximum-precision 32-bit float Fast Local Laplacian Filter (Aubry et al. 2014)\n"
                     "operating in the Safdar et al. (2017) JzAzBz perceptual color space.\n"
                     "Supports SDR (sRGB/Rec.709) and HDR (scRGB, HDR10 PQ, HDR10 HLG).";
    >
    {
        // Phase 1: Entry to full-precision JzAzBz + per-frame HDR peak detection
        pass ToJzAzBz
        {
            VertexShader = PostProcessVS;
            PixelShader  = PS_ToJzAzBz;
            RenderTarget = TexJzAzBz;
        }
        pass Peak_1 { VertexShader = PostProcessVS; PixelShader = PS_Peak_1; RenderTarget = TexPeak1; }
        pass Peak_2 { VertexShader = PostProcessVS; PixelShader = PS_Peak_2; RenderTarget = TexPeak2; }
        pass Peak_3 { VertexShader = PostProcessVS; PixelShader = PS_Peak_3; RenderTarget = TexPeak3; }
        pass Peak_4 { VertexShader = PostProcessVS; PixelShader = PS_Peak_4; RenderTarget = TexPeak4; }

        // Phase 2: Scalar Jz Gaussian pyramid (G1 .. G{L-2})
        pass ReduceJz_01 { VertexShader = PostProcessVS; PixelShader = PS_ReduceJz_01; RenderTarget = TexG1; }
        pass ReduceJz_12 { VertexShader = PostProcessVS; PixelShader = PS_ReduceJz_12; RenderTarget = TexG2; }
        pass ReduceJz_23 { VertexShader = PostProcessVS; PixelShader = PS_ReduceJz_23; RenderTarget = TexG3; }
    #if _FLLF_LEVELS >= 6
        pass ReduceJz_34 { VertexShader = PostProcessVS; PixelShader = PS_ReduceJz_34; RenderTarget = TexG4; }
    #endif
    #if _FLLF_LEVELS >= 7
        pass ReduceJz_45 { VertexShader = PostProcessVS; PixelShader = PS_ReduceJz_45; RenderTarget = TexG5; }
    #endif
    #if _FLLF_LEVELS >= 8
        pass ReduceJz_56 { VertexShader = PostProcessVS; PixelShader = PS_ReduceJz_56; RenderTarget = TexG6; }
    #endif
    #if _FLLF_LEVELS >= 9
        pass ReduceJz_67 { VertexShader = PostProcessVS; PixelShader = PS_ReduceJz_67; RenderTarget = TexG7; }
    #endif

        // Phase 3: Remapped Delta-Residual Gaussian pyramids G1[Delta_R] .. G{L-1}[Delta_R]
        pass ReduceR01_H { VertexShader = PostProcessVS; PixelShader = PS_ReduceR01_H; _FLLF_BIND_MRT(H1_R) }
        pass ReduceR01_V { VertexShader = PostProcessVS; PixelShader = PS_ReduceR01_V; _FLLF_BIND_MRT(G1_R) }
        pass ReduceR_12  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_12;  _FLLF_BIND_MRT(G2_R) }
        pass ReduceR_23  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_23;  _FLLF_BIND_MRT(G3_R) }
        pass ReduceR_34  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_34;  _FLLF_BIND_MRT(G4_R) }
    #if _FLLF_LEVELS >= 6
        pass ReduceR_45  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_45;  _FLLF_BIND_MRT(G5_R) }
    #endif
    #if _FLLF_LEVELS >= 7
        pass ReduceR_56  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_56;  _FLLF_BIND_MRT(G6_R) }
    #endif
    #if _FLLF_LEVELS >= 8
        pass ReduceR_67  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_67;  _FLLF_BIND_MRT(G7_R) }
    #endif
    #if _FLLF_LEVELS >= 9
        pass ReduceR_78  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_78;  _FLLF_BIND_MRT(G8_R) }
    #endif

        // Phase 4: Intensity-slice interpolation of Laplacian delta bands L1 .. L{L-2}
        pass InterpL_1 { VertexShader = PostProcessVS; PixelShader = PS_InterpL_1; RenderTarget = TexL1; }
        pass InterpL_2 { VertexShader = PostProcessVS; PixelShader = PS_InterpL_2; RenderTarget = TexL2; }
        pass InterpL_3 { VertexShader = PostProcessVS; PixelShader = PS_InterpL_3; RenderTarget = TexL3; }
    #if _FLLF_LEVELS >= 6
        pass InterpL_4 { VertexShader = PostProcessVS; PixelShader = PS_InterpL_4; RenderTarget = TexL4; }
    #endif
    #if _FLLF_LEVELS >= 7
        pass InterpL_5 { VertexShader = PostProcessVS; PixelShader = PS_InterpL_5; RenderTarget = TexL5; }
    #endif
    #if _FLLF_LEVELS >= 8
        pass InterpL_6 { VertexShader = PostProcessVS; PixelShader = PS_InterpL_6; RenderTarget = TexL6; }
    #endif
    #if _FLLF_LEVELS >= 9
        pass InterpL_7 { VertexShader = PostProcessVS; PixelShader = PS_InterpL_7; RenderTarget = TexL7; }
    #endif

        // Phase 5: Bottom-up Laplacian pyramid collapse (reuses TexG{l} as O_{l})
    #if _FLLF_LEVELS >= 9
        pass Collapse_76 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_76; RenderTarget = TexG6; }
    #endif
    #if _FLLF_LEVELS >= 8
        pass Collapse_65 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_65; RenderTarget = TexG5; }
    #endif
    #if _FLLF_LEVELS >= 7
        pass Collapse_54 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_54; RenderTarget = TexG4; }
    #endif
    #if _FLLF_LEVELS >= 6
        pass Collapse_43 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_43; RenderTarget = TexG3; }
    #endif
        pass Collapse_32 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_32; RenderTarget = TexG2; }
        pass Collapse_21 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_21; RenderTarget = TexG1; }

        // Phase 6: Fused full-res L0 + final collapse + JzAzBz -> SDR/HDR output
        pass Reconstruct
        {
            VertexShader = PostProcessVS;
            PixelShader  = PS_Reconstruct;
            SRGBWriteEnable = false;
        }
    }
}
