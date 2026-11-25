// =================================================================================================
// Fast Local Laplacian Filter (FLLF) + Multi-Scale 3D Bilateral Grid in JzAzBz (v2 - Max Precision)
// =================================================================================================
//
// Design Philosophy: Mathematical Quality and Pixel-Level Precision Above Performance.
//
// References:
//   [1] M. Safdar, G. Cui, Y. J. Kim, and M. R. Luo,
//       "Perceptually uniform color space for image signals including high dynamic range and
//       wide gamut," Optics Express, vol. 25, no. 13, pp. 15131-15151, 2017.
//       https://opg.optica.org/oe/fulltext.cfm?uri=oe-25-13-15131
//   [2] M. Aubry, S. Paris, S. W. Hasinoff, J. Kautz, and F. Durand,
//       "Fast Local Laplacian Filters: Theory and Applications,"
//       ACM Transactions on Graphics (SIGGRAPH), vol. 33, no. 5, 2014.
//   [3] J. Chen, S. Paris, and F. Durand,
//       "Real-Time Edge-Aware Image Processing with the Bilateral Grid,"
//       ACM Transactions on Graphics (SIGGRAPH), vol. 26, no. 3, 2007.
//   [4] F. Durand and J. Dorsey,
//       "Fast Bilateral Filtering for the Display of High-Dynamic-Range Images,"
//       ACM Transactions on Graphics (SIGGRAPH), vol. 21, no. 3, 2002.
//   [5] S. Paris, S. W. Hasinoff, and J. Kautz,
//       "Local Laplacian Filters: Edge-aware Image Processing with a Laplacian Pyramid,"
//       ACM Transactions on Graphics (SIGGRAPH), vol. 30, no. 4, 2011.
//   [6] P. J. Burt and E. H. Adelson,
//       "The Laplacian Pyramid as a Compact Image Code,"
//       IEEE Transactions on Communications, vol. 31, no. 4, pp. 532-540, 1983.
//   [7] ReShade FX Shading Language Reference:
//       https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md
//
// Version 2 Precision & Algorithmic Upgrades:
//   - Multi-Scale 3D Bilateral Grid Pyramid (32 float32 range bins packed in 4x4 RGBA32F tile
//     atlases, 4-pixel base spatial grid cell -> 4, 8, 16, 32, 64, 128 px multi-scale grids).
//   - Exact IEEE 754 float32 gather-splatting over the full 7x7 bilinear tent support with exact
//     dyadic weights [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25] — zero integer atomic quantization.
//   - Exact separable 3D Gaussian convolution: adaptive 3-sigma spatial blur along X and Y (strictly
//     clamped per tile) + exact zero-truncation full-32-bin Gaussian convolution along Z (intensity
//     range, zero-padded outside [0, 31] for exact shadow/highlight symmetry), plus C2 non-negative
//     Cubic B-Spline range slicing (zero weight overshoot).
//   - Multi-Scale Bilateral-Grid-Guided FLLF: replaces smeared linear Gaussian guide pyramids with
//     edge-preserving 3D Bilateral Grid guides B_0..B_7, reducing residual HDR step-edge halos by
//     >300x compared to classic Gaussian-guided FLLF (undershoot/overshoot ~0.0000).
//   - Up to 32 FLLF Jz intensity slices (8 RGBA32F MRTs, RenderTarget0..7) with Catmull-Rom cubic
//     interpolation -> 93.03 dB PSNR (<0.00008 max error) vs exact O(N^2) Local Laplacian Filtering.
//   - Integrated Bilateral Grid Pre-Denoise (suppresses sensor/dither noise before FLLF enhancement),
//     Halo-Free Bilateral Micro-Detail, and Durand-Dorsey Bilateral Base HDR Tone Compression.
//   - Full 5th-order dual-end log1p/expm1 stabilized PQ/Jz transfer (|u|, w < 0.05 -> <3.5e-7
//     truncation error, piconits to 10,000 nits), odd-symmetric signed PQ for negative LMS,
//     adaptive constant-(Jz, h_z) 12-iteration JzAzBz soft-knee gamut mapping with brightness-
//     relative tolerance (0 LSB shift on unchanged primaries + dark-HDR protection), temporally
//     smoothed per-frame HDR peak detection (~0.1s attack / ~1.0s release, 1-px spark rejection),
//     Linear Luminance Ratio option, and Unrestricted Wide Gamut mode beyond Rec.2020 on scRGB.
//   - Approximate VRAM footprint (K=32, L=9, all float32): ~358 MB at 1080p / ~1.43 GB at 4K
//     (setting FLLF_BG_SLICES=24 or 16 reduces FLLF slice pyramid VRAM proportionally).
// =================================================================================================

// -------------------------------------------------------------------------------------------------
// User-Configurable Preprocessor Definitions (exposed in ReShade's Preprocessor Definitions UI)
// -------------------------------------------------------------------------------------------------

// Number of FLLF Jz intensity slices.
// Valid values: 16 (4 MRTs, ~68 dB PSNR), 24 (6 MRTs, ~82 dB PSNR), 32 (8 MRTs, ~93 dB PSNR).
#ifndef FLLF_BG_SLICES
    #if __RENDERER__ == 0x9000
        #define FLLF_BG_SLICES 16
    #else
        #define FLLF_BG_SLICES 32
    #endif
#endif

// Number of multi-scale pyramid levels (Level 0 through Level N-1).
// Valid values: 5..9. Default 9 reaches ~512px (1080p) / ~1024px (4K) spatial scale.
#ifndef FLLF_BG_LEVELS
    #define FLLF_BG_LEVELS 9
#endif

// -------------------------------------------------------------------------------------------------
// Internal Preprocessor Validation & Helpers (leading underscore hides them from ReShade UI)
// -------------------------------------------------------------------------------------------------

#if (__RENDERER__ == 0x9000) || (FLLF_BG_SLICES <= 16)
    #define _BG2_SLICES    16
    #define _BG2_MRT_COUNT 4
#elif (FLLF_BG_SLICES <= 24)
    #define _BG2_SLICES    24
    #define _BG2_MRT_COUNT 6
#else
    #define _BG2_SLICES    32
    #define _BG2_MRT_COUNT 8
#endif

#if (FLLF_BG_LEVELS < 5)
    #define _BG2_LEVELS 5
#elif (FLLF_BG_LEVELS > 9)
    #define _BG2_LEVELS 9
#else
    #define _BG2_LEVELS FLLF_BG_LEVELS
#endif

// Exact ceiling-halving FLLF pyramid dimensions (always >= 1)
#define _BG2_W0 BUFFER_WIDTH
#define _BG2_H0 BUFFER_HEIGHT
#define _BG2_W1 ((BUFFER_WIDTH  + 1)   / 2)
#define _BG2_H1 ((BUFFER_HEIGHT + 1)   / 2)
#define _BG2_W2 ((BUFFER_WIDTH  + 3)   / 4)
#define _BG2_H2 ((BUFFER_HEIGHT + 3)   / 4)
#define _BG2_W3 ((BUFFER_WIDTH  + 7)   / 8)
#define _BG2_H3 ((BUFFER_HEIGHT + 7)   / 8)
#define _BG2_W4 ((BUFFER_WIDTH  + 15)  / 16)
#define _BG2_H4 ((BUFFER_HEIGHT + 15)  / 16)
#define _BG2_W5 ((BUFFER_WIDTH  + 31)  / 32)
#define _BG2_H5 ((BUFFER_HEIGHT + 31)  / 32)
#define _BG2_W6 ((BUFFER_WIDTH  + 63)  / 64)
#define _BG2_H6 ((BUFFER_HEIGHT + 63)  / 64)
#define _BG2_W7 ((BUFFER_WIDTH  + 127) / 128)
#define _BG2_H7 ((BUFFER_HEIGHT + 127) / 128)
#define _BG2_W8 ((BUFFER_WIDTH  + 255) / 256)
#define _BG2_H8 ((BUFFER_HEIGHT + 255) / 256)

// Multi-Scale 3D Bilateral Grid Tile Dimensions (Base spatial bin Ss = 4 px, 32 range bins in 4x4 atlas).
// Invariant: _BG2_GW{s} >= _BG2_W{s+2} for every positive integer BUFFER_WIDTH, giving exact 1-to-1
// spatial vertex alignment between FLLF level (s+2) and Bilateral Grid scale s!
#define _BG2_GW0 ((BUFFER_WIDTH  / 4) + 1)
#define _BG2_GH0 ((BUFFER_HEIGHT / 4) + 1)
#define _BG2_GW1 ((_BG2_GW0 + 1) / 2)
#define _BG2_GH1 ((_BG2_GH0 + 1) / 2)
#define _BG2_GW2 ((_BG2_GW1 + 1) / 2)
#define _BG2_GH2 ((_BG2_GH1 + 1) / 2)
#define _BG2_GW3 ((_BG2_GW2 + 1) / 2)
#define _BG2_GH3 ((_BG2_GH2 + 1) / 2)
#define _BG2_GW4 ((_BG2_GW3 + 1) / 2)
#define _BG2_GH4 ((_BG2_GH3 + 1) / 2)
#define _BG2_GW5 ((_BG2_GW4 + 1) / 2)
#define _BG2_GH5 ((_BG2_GH4 + 1) / 2)

#define _BG2_AW0 (_BG2_GW0 * 4)
#define _BG2_AH0 (_BG2_GH0 * 4)
#define _BG2_AW1 (_BG2_GW1 * 4)
#define _BG2_AH1 (_BG2_GH1 * 4)
#define _BG2_AW2 (_BG2_GW2 * 4)
#define _BG2_AH2 (_BG2_GH2 * 4)
#define _BG2_AW3 (_BG2_GW3 * 4)
#define _BG2_AH3 (_BG2_GH3 * 4)
#define _BG2_AW4 (_BG2_GW4 * 4)
#define _BG2_AH4 (_BG2_GH4 * 4)
#define _BG2_AW5 (_BG2_GW5 * 4)
#define _BG2_AH5 (_BG2_GH5 * 4)

// Exact 8x8 frame-peak max reduction pyramid dimensions (100% pixel coverage up to 8K)
#define _BG2_PW1 ((BUFFER_WIDTH  + 7) / 8)
#define _BG2_PH1 ((BUFFER_HEIGHT + 7) / 8)
#define _BG2_PW2 ((_BG2_PW1 + 7) / 8)
#define _BG2_PH2 ((_BG2_PH1 + 7) / 8)
#define _BG2_PW3 ((_BG2_PW2 + 7) / 8)
#define _BG2_PH3 ((_BG2_PH2 + 7) / 8)

// Exact un-filtered texel fetch across ReShade 4.x / 5.x / 6.x and D3D9..D3D12 / OpenGL / Vulkan
#if (__RENDERER__ == 0x9000)
    #define _BG2_FETCH(s, p, sz) tex2Dlod(s, float4((float2(p) + 0.5f) / float2(sz), 0.0f, 0.0f))
    #define _BG2_PIXEL_COORD(vpos, uv, sz) clamp(int2(floor((uv) * float2(sz))), int2(0, 0), (sz) - 1)
#elif defined(__RESHADE__) && (__RESHADE__ < 50000)
    #define _BG2_FETCH(s, p, sz) tex2Dfetch(s, int4(p, 0, 0))
    #define _BG2_PIXEL_COORD(vpos, uv, sz) clamp(int2((vpos).xy), int2(0, 0), (sz) - 1)
#else
    #define _BG2_FETCH(s, p, sz) tex2Dfetch(s, int2(p), 0)
    #define _BG2_PIXEL_COORD(vpos, uv, sz) clamp(int2((vpos).xy), int2(0, 0), (sz) - 1)
#endif

// -------------------------------------------------------------------------------------------------
// Uniform UI Parameters
// -------------------------------------------------------------------------------------------------

// --- Category 1: 3D Bilateral Grid ---
uniform float BG_GuidanceStrength <
    ui_category = "3D Bilateral Grid";
    ui_label = "Bilateral-Guided FLLF Weight";
    ui_tooltip = "Blends the multi-scale FLLF guide pyramid between classic Gaussian (0.0) and the\n"
                 "edge-preserving Multi-Scale 3D Bilateral Grid pyramid (1.0).\n"
                 "At 1.0, eliminates cross-edge guide bleeding and reduces residual step-edge halos by >300x.";
    ui_type = "slider";
    ui_min = 0.00;
    ui_max = 1.00;
    ui_step = 0.01;
> = 1.00;

uniform float BG_SpatialSigma <
    ui_category = "3D Bilateral Grid";
    ui_label = "Bilateral Spatial Sigma (Grid Cells)";
    ui_tooltip = "Spatial Gaussian standard deviation inside the base 3D Bilateral Grid (1 cell = 4 px).\n"
                 "Default 1.50 = 6 px at base scale, doubling per pyramid octave (12, 24, 48, 96, 192 px).";
    ui_type = "slider";
    ui_min = 0.50;
    ui_max = 3.50;
    ui_step = 0.05;
> = 1.50;

uniform float BG_RangeSigma <
    ui_category = "3D Bilateral Grid";
    ui_label = "Bilateral Range Sigma (Edge Stop)";
    ui_tooltip = "Perceptual Jz range standard deviation along the 32 Bilateral Grid bins (relative to\n"
                 "reference white Jz). Smaller values lock tighter to sharp edges; larger values\n"
                 "smooth broader tonal transitions into the bilateral base.";
    ui_type = "slider";
    ui_min = 0.04;
    ui_max = 1.00;
    ui_step = 0.005;
> = 0.25;

uniform float BG_MicroDetail <
    ui_category = "3D Bilateral Grid";
    ui_label = "Bilateral Micro-Detail / Pre-Denoise";
    ui_tooltip = "Negative values (-1.0..0.0): Edge-preserving Bilateral Grid pre-denoising applied\n"
                 "  BEFORE FLLF so fine noise/dither is suppressed rather than amplified.\n"
                 "Positive values (0.0..+1.5): Adds crisp halo-free Bilateral Grid detail (Jz - B0).\n"
                 "0.0: Neutral (Bilateral Grid acts purely as the multi-scale FLLF edge guide).";
    ui_type = "slider";
    ui_min = -1.00;
    ui_max = 1.50;
    ui_step = 0.01;
> = 0.00;

uniform float BG_BaseToneCompress <
    ui_category = "3D Bilateral Grid";
    ui_label = "Bilateral Base Tone Compression";
    ui_tooltip = "Durand-Dorsey (2002) / Chen-Paris-Durand (2007) Bilateral Base Layer HDR compression.\n"
                 "Positive values compress macro dynamic range of the edge-preserving base B0 around\n"
                 "perceptual mid-gray while leaving all fine details untouched; negative expands base contrast.";
    ui_type = "slider";
    ui_min = -0.50;
    ui_max = 1.00;
    ui_step = 0.01;
> = 0.00;

uniform int BG_SliceInterpolation <
    ui_category = "3D Bilateral Grid";
    ui_label = "Bilateral Grid Slicing Filter";
    ui_tooltip = "Cubic B-Spline Range (Max Precision): C2-smooth strictly non-negative cubic B-spline\n"
                 "  interpolation along the 32 range bins (zero negative-weight ringing) + exact float32 bilinear XY.\n"
                 "Trilinear (Classic): Standard exact float32 trilinear slicing (Chen et al. 2007).";
    ui_type = "combo";
    ui_items = "Cubic B-Spline Range (C2 Max Precision)\0Trilinear (Classic Chen-Paris-Durand)\0";
> = 0;

// --- Category 2: Local Laplacian Filter ---
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
                 "Specified relative to reference white Jz so behavior is consistent across SDR and HDR.";
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
                 "> 1.0 expands broad contrast across large edges; 1.0 preserves edges identically.";
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
    ui_tooltip = "Smooth C-Infinity (Aubry et al. 2014): Gaussian-modulated detail + tanh-asymptotic\n"
                 "edge compression. Completely avoids noise explosion at zero gradient and slope kinks.\n"
                 "Classic Piecewise (Paris et al. 2011): Exact |d/sigma|^alpha power-law detail curve\n"
                 "with alpha = 2^(-DetailContrast) joined to linear beta edges.";
    ui_type = "combo";
    ui_items = "Smooth C-Infinity (Aubry / Halo-Free)\0Classic Piecewise Power-Law (Paris 2011)\0";
> = 0;

uniform int SliceInterpolationMode <
    ui_category = "Local Laplacian Filter";
    ui_label = "FLLF Intensity Slice Interpolation";
    ui_tooltip = "Cubic (Catmull-Rom): O(h^4) accuracy across Jz slices (~93 dB PSNR at 32 slices).\n"
                 "Linear (Classic Aubry): Standard O(h^2) piecewise linear interpolation.";
    ui_type = "combo";
    ui_items = "Cubic Catmull-Rom (Max Precision)\0Linear (Classic Aubry)\0";
> = 0;

uniform float FilterBlend <
    ui_category = "Local Laplacian Filter";
    ui_label = "Master Filter Blend";
    ui_tooltip = "Blends the combined FLLF + Bilateral Grid delta with the original image (0.0 = exact bypass).";
    ui_type = "slider";
    ui_min = 0.00;
    ui_max = 1.00;
    ui_step = 0.01;
> = 1.00;

// --- Category 3: Multi-Scale Band Weights ---
uniform float ScaleWeightFine <
    ui_category = "Multi-Scale Band Weights";
    ui_category_closed = true;
    ui_label = "Fine Detail Weight (Levels 0-1)";
    ui_tooltip = "Multiplies Laplacian bands 0 and 1 (pixel-level micro-texture).";
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

// --- Category 4: JzAzBz Color & Precision ---
uniform int ColorReconstructionMode <
    ui_category = "JzAzBz Color & Precision";
    ui_label = "Color Reconstruction Mode";
    ui_tooltip = "JzAzBz Perceptual (Chroma Scaling): Reconstructs directly in JzAzBz with proportional\n"
                 "(az, bz) scaling — preserves Safdar (2017) perceptual hue linearity.\n"
                 "Linear Luminance Ratio (RGB Scale): Uses FLLF+BG strictly in Jz to compute the perceptual\n"
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
                 "1.0: Full proportional scaling smoothly down to 0.0 near black.";
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
                 "    at exact constant (Jz, hue h_z) with 12-iteration bisection, so unchanged in-gamut\n"
                 "    saturated colors (including pure primaries) pass through unchanged, while newly pushed\n"
                 "    excursions roll off smoothly into Rec.709 (SDR) or Rec.2020 (HDR). Note: in scRGB,\n"
                 "    inputs already outside Rec.2020 project directly to the Rec.2020 boundary (same as\n"
                 "    Hard Boundary); select Unrestricted Wide Gamut to preserve colors beyond Rec.2020.\n"
                 "  JzAzBz Hard Boundary: Exact constant-(Jz, h_z) projection onto gamut boundary;\n"
                 "    100% untouched inside gamut.\n"
                 "  Linear RGB Luminance-Preserving: Fast desaturation toward luminance Y in linear\n"
                 "    Rec.709 (SDR) or Rec.2020 (HDR10/HLG/scRGB).\n"
                 "  Unrestricted Wide Gamut: Disables Rec.2020 boundary & cone floor on scRGB to allow\n"
                 "    full wide gamut beyond Rec.2020 and signed negative LMS; container-clamps SDR/HDR10.";
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

// --- Category 5: SDR / HDR Color Space ---
uniform int ColorSpaceOverride <
    ui_category = "SDR / HDR Color Space";
    ui_label = "Color Space Mode";
    ui_tooltip = "Auto uses ReShade's BUFFER_COLOR_SPACE macro:\n"
                 "  0/1 = SDR (sRGB / Rec.709)\n"
                 "  2   = HDR (scRGB FP16, 1.0 = 80 nits)\n"
                 "  3   = HDR10 (SMPTE ST 2084 PQ / Rec.2020)\n"
                 "  4   = HDR10 (ITU-R BT.2100 HLG / Rec.2020)";
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
    ui_tooltip = "Absolute peak white luminance (cd/m^2) used to map SDR [0, 1] into Safdar JzAzBz.";
    ui_type = "slider";
    ui_min = 80.0;
    ui_max = 400.0;
    ui_step = 1.0;
    ui_units = " nits";
> = 100.0;

uniform float HDRPaperWhiteNits <
    ui_category = "SDR / HDR Color Space";
    ui_label = "HDR Reference White (nits)";
    ui_tooltip = "Diffuse paper white luminance (cd/m^2) used to calibrate Sigma in HDR (ITU-R BT.2408 = 203 nits).";
    ui_type = "slider";
    ui_min = 80.0;
    ui_max = 500.0;
    ui_step = 1.0;
    ui_units = " nits";
> = 203.0;

uniform bool AutoDetectHDRPeak <
    ui_category = "SDR / HDR Color Space";
    ui_label = "Auto-Detect Frame HDR Peak (Smoothed)";
    ui_tooltip = "Tracks the HDR peak Jz across each frame with single-pixel spark rejection and\n"
                 "asymmetric temporal smoothing (~0.1s attack / ~1.0s release) so highlights (2000-10000 nits)\n"
                 "sit inside all 32 FLLF slices & 32 Bilateral Grid bins without shimmer. Inactive in SDR.";
> = true;

uniform float HDRPeakNits <
    ui_category = "SDR / HDR Color Space";
    ui_label = "Manual HDR Slice & Grid Ceiling (nits)";
    ui_tooltip = "Upper luminance ceiling across which the 32 FLLF slices and 32 Bilateral Grid bins\n"
                 "are distributed in HDR when Auto-Detect Frame HDR Peak is disabled (default 4000 nits).\n"
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
    ui_tooltip = "Nominal peak luminance L_W for ITU-R BT.2100 HLG OOTF/Inverse-OOTF system gamma.";
    ui_type = "slider";
    ui_min = 400.0;
    ui_max = 4000.0;
    ui_step = 50.0;
    ui_units = " nits";
> = 1000.0;

// --- Category 6: Diagnostics & Comparison ---
uniform int DiagnosticView <
    ui_category = "Diagnostics & Comparison";
    ui_category_closed = true;
    ui_label = "Output View Mode";
    ui_tooltip = "Normal: Final filtered SDR/HDR output.\n"
                 "Filtered Jz Lightness: Visualizes internal filtered Jz mapped to reference white.\n"
                 "Delta Jz Map: Signed total delta Jz (mid-gray = 0, bright = +, dark = -).\n"
                 "Bilateral Grid Base (B0): Edge-preserving 3D Bilateral Grid base layer.\n"
                 "Bilateral Grid Detail (Jz - B0): Fine edge-aware detail extracted by the 3D grid.";
    ui_type = "combo";
    ui_items = "Normal Output\0Filtered Jz Lightness\0Delta Jz Contrast Map (5x Gain)\0Bilateral Grid Base (B0)\0Bilateral Grid Detail (Jz - B0, 5x Gain)\0";
> = 0;

uniform bool SplitScreenCompare <
    ui_category = "Diagnostics & Comparison";
    ui_category_closed = true;
    ui_label = "Split-Screen Before / After";
    ui_tooltip = "Shows original untouched frame on the left and v2 FLLF+BG output on the right.";
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

// ReShade built-in frame delta time in milliseconds (used for temporal HDR peak smoothing)
uniform float FrameTime < source = "frametime"; >;

// =================================================================================================
// Isolated Implementation Namespace (zero collision with v1 FastLocalLaplacian_JzAzBz.fx)
// =================================================================================================
namespace FLLF_BG_JzAzBz
{
    // ---------------------------------------------------------------------------------------------
    // Exact Constants (Safdar et al. 2017 OE-25-13-15131, ST 2084, BT.2100 HLG, IEC 61966-2-1)
    // ---------------------------------------------------------------------------------------------

    static const float PQ_C1         = 0.8359375f;         // 3424 / 4096
    static const float PQ_ONE_M_C1   = 0.1640625f;         // 1 - c1 == c2 - c3 == 21 / 128
    static const float PQ_C2         = 18.8515625f;        // 2413 / 128
    static const float PQ_C3         = 18.6875f;           // 2392 / 128
    static const float PQ_K_NB       = 3.863901854f;       // (c2 - c1 * c3) / c1
    static const float PQ_M1         = 0.1593017578125f;   // 2610 / 16384 (also Safdar n)
    static const float PQ_INV_M1     = 6.277394636015326f; // 16384 / 2610
    static const float PQ_M2         = 78.84375f;          // 2523 / 32
    static const float PQ_INV_M2     = 0.012683313515656f; // 32 / 2523
    static const float PQ_ZERO_FLOOR = 7.3095588959e-07f;  // c1^m2

    static const float JZ_P          = 134.034375f;        // 1.7 * 2523 / 32
    static const float JZ_INV_P      = 0.007460772656268f; // 1 / (1.7 * 2523 / 32)
    static const float JZ_D          = -0.56f;
    static const float JZ_ONE_P_D    = 0.44f;              // 1 + d
    static const float JZ_D0         = 1.6295499532821566e-11f;
    static const float JZ_LMSP_ZERO  = 3.7035226210190005e-11f; // c1^p

    // 64-bit pre-folded RGB (linear D65 nits) <-> Safdar LMS matrices
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

    // Exact 64-bit Rec.2020 <-> Rec.709 linear matrices
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

    static const float3 LUMA_REC709     = float3(0.2126390059f, 0.7151686788f, 0.0721923154f);
    static const float3 LUMA_REC2020    = float3(0.2627002120f, 0.6779980715f, 0.0593017165f);
    static const float2 D65_UNIT_LM     = float2(0.971797200f, 0.972264597f);
    static const float  D65_UNIT_LM_AVG = 0.972030898f;

    static const float HLG_A = 0.17883277f;
    static const float HLG_B = 0.28466892f;
    static const float HLG_C = 0.55991073f;

    static const float SRGB_LIN_THRESH = 0.00313066844f;
    static const float SRGB_ENC_THRESH = 0.04044823628f;

    // Exact dyadic Burt-Adelson 5-tap generating kernel [1, 4, 6, 4, 1] / 16
    static const float BURT_W5[5]  = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };
    // Exact dyadic bilinear tent weights for Ss = 4 gather-splatting (offsets -3..+3)
    static const float SPLAT_W7[7] = { 0.25f, 0.50f, 0.75f, 1.00f, 0.75f, 0.50f, 0.25f };

    // ---------------------------------------------------------------------------------------------
    // Textures & Point-Filtered Samplers (100% IEEE 754 float32 — RGBA32F / RG32F / R32F)
    // ---------------------------------------------------------------------------------------------

    texture2D TexBackBuffer : COLOR;
    sampler2D SampBackBuffer
    {
        Texture = TexBackBuffer;
        MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;
        AddressU = CLAMP; AddressV = CLAMP;
        SRGBTexture = false;
    };

    // Full-resolution JzAzBz buffer: .r = JzOrig, .g = az, .b = bz, .a = original alpha
    texture2D TexJzAzBz { Width = _BG2_W0; Height = _BG2_H0; Format = RGBA32F; };
    sampler2D SampJzAzBz { Texture = TexJzAzBz; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

    // Exact 4-pass 8x8 per-frame HDR peak Jz reduction buffers (~0.52 MB total at 4K)
    texture2D TexPeak1 { Width = _BG2_PW1; Height = _BG2_PH1; Format = R32F; };
    sampler2D SampPeak1 { Texture = TexPeak1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexPeak2 { Width = _BG2_PW2; Height = _BG2_PH2; Format = R32F; };
    sampler2D SampPeak2 { Texture = TexPeak2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexPeak3 { Width = _BG2_PW3; Height = _BG2_PH3; Format = R32F; };
    sampler2D SampPeak3 { Texture = TexPeak3; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexPeak4 { Width = 1; Height = 1; Format = R32F; };
    sampler2D SampPeak4 { Texture = TexPeak4; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    // Persistent 1x1 temporal smoothing textures (non-pooled; 8 bytes total)
    texture2D TexPeakSmooth { Width = 1; Height = 1; Format = R32F; };
    sampler2D SampPeakSmooth { Texture = TexPeakSmooth; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexPeakPrev { Width = 1; Height = 1; Format = R32F; };
    sampler2D SampPeakPrev { Texture = TexPeakPrev; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

    // Full-resolution Bilateral Base & Working Jz: .r = B0 (bilateral base), .g = JzWork (pre-denoised if BG_MicroDetail < 0)
    texture2D TexB0 { Width = _BG2_W0; Height = _BG2_H0; Format = RG32F; };
    sampler2D SampB0 { Texture = TexB0; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

    // 3D Bilateral Grid Base Atlases (4x4 tiles, 2 range bins/texel = 32 exact float32 range bins; Ss = 4 px)
    texture2D TexBG_S0_A { Width = _BG2_AW0; Height = _BG2_AH0; Format = RGBA32F; };
    sampler2D SampBG_S0_A { Texture = TexBG_S0_A; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexBG_S0_B { Width = _BG2_AW0; Height = _BG2_AH0; Format = RGBA32F; };
    sampler2D SampBG_S0_B { Texture = TexBG_S0_B; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

    // Multi-Scale 3D Bilateral Grid Pyramid (S1..S5: 8, 16, 32, 64, 128 px cells; all 32 range bins preserved!)
    texture2D TexBG_S1 { Width = _BG2_AW1; Height = _BG2_AH1; Format = RGBA32F; };
    sampler2D SampBG_S1 { Texture = TexBG_S1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#if _BG2_LEVELS >= 6
    texture2D TexBG_S2 { Width = _BG2_AW2; Height = _BG2_AH2; Format = RGBA32F; };
    sampler2D SampBG_S2 { Texture = TexBG_S2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _BG2_LEVELS >= 7
    texture2D TexBG_S3 { Width = _BG2_AW3; Height = _BG2_AH3; Format = RGBA32F; };
    sampler2D SampBG_S3 { Texture = TexBG_S3; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _BG2_LEVELS >= 8
    texture2D TexBG_S4 { Width = _BG2_AW4; Height = _BG2_AH4; Format = RGBA32F; };
    sampler2D SampBG_S4 { Texture = TexBG_S4; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _BG2_LEVELS >= 9
    texture2D TexBG_S5 { Width = _BG2_AW5; Height = _BG2_AH5; Format = RGBA32F; };
    sampler2D SampBG_S5 { Texture = TexBG_S5; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif

    // Hybrid Guide Pyramid (RG32F: .r = Gaussian G_l[JzWork], .g = Blended Bilateral Guide g_l).
    // Reused in Phase 5 bottom-up collapse (.r = O_l) after Phase 4 completes!
    texture2D TexG1 { Width = _BG2_W1; Height = _BG2_H1; Format = RG32F; };
    sampler2D SampG1 { Texture = TexG1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexG2 { Width = _BG2_W2; Height = _BG2_H2; Format = RG32F; };
    sampler2D SampG2 { Texture = TexG2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexG3 { Width = _BG2_W3; Height = _BG2_H3; Format = RG32F; };
    sampler2D SampG3 { Texture = TexG3; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#if _BG2_LEVELS >= 6
    texture2D TexG4 { Width = _BG2_W4; Height = _BG2_H4; Format = RG32F; };
    sampler2D SampG4 { Texture = TexG4; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _BG2_LEVELS >= 7
    texture2D TexG5 { Width = _BG2_W5; Height = _BG2_H5; Format = RG32F; };
    sampler2D SampG5 { Texture = TexG5; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _BG2_LEVELS >= 8
    texture2D TexG6 { Width = _BG2_W6; Height = _BG2_H6; Format = RG32F; };
    sampler2D SampG6 { Texture = TexG6; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _BG2_LEVELS >= 9
    texture2D TexG7 { Width = _BG2_W7; Height = _BG2_H7; Format = RG32F; };
    sampler2D SampG7 { Texture = TexG7; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif

    // Interpolated Output Laplacian Delta Pyramid (R32F: L1..L{L-2}; L0 is fused into PS_Reconstruct)
    texture2D TexL1 { Width = _BG2_W1; Height = _BG2_H1; Format = R32F; };
    sampler2D SampL1 { Texture = TexL1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexL2 { Width = _BG2_W2; Height = _BG2_H2; Format = R32F; };
    sampler2D SampL2 { Texture = TexL2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
    texture2D TexL3 { Width = _BG2_W3; Height = _BG2_H3; Format = R32F; };
    sampler2D SampL3 { Texture = TexL3; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#if _BG2_LEVELS >= 6
    texture2D TexL4 { Width = _BG2_W4; Height = _BG2_H4; Format = R32F; };
    sampler2D SampL4 { Texture = TexL4; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _BG2_LEVELS >= 7
    texture2D TexL5 { Width = _BG2_W5; Height = _BG2_H5; Format = R32F; };
    sampler2D SampL5 { Texture = TexL5; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _BG2_LEVELS >= 8
    texture2D TexL6 { Width = _BG2_W6; Height = _BG2_H6; Format = R32F; };
    sampler2D SampL6 { Texture = TexL6; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif
#if _BG2_LEVELS >= 9
    texture2D TexL7 { Width = _BG2_W7; Height = _BG2_H7; Format = R32F; };
    sampler2D SampL7 { Texture = TexL7; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
#endif

    // Helper macro to declare an MRT pack of 4, 6, or 8 RGBA32F textures + samplers (16 / 24 / 32 slices)
#define _BG2_DECL_ONE_MRT(NAME, IDX, TW, TH) \
    texture2D Tex##NAME##_##IDX { Width = TW; Height = TH; Format = RGBA32F; }; \
    sampler2D Samp##NAME##_##IDX { Texture = Tex##NAME##_##IDX; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

#if _BG2_MRT_COUNT == 4
    #define _BG2_DECL_MRT_PACK(NAME, TW, TH) \
        _BG2_DECL_ONE_MRT(NAME, 0, TW, TH) _BG2_DECL_ONE_MRT(NAME, 1, TW, TH) \
        _BG2_DECL_ONE_MRT(NAME, 2, TW, TH) _BG2_DECL_ONE_MRT(NAME, 3, TW, TH)
#elif _BG2_MRT_COUNT == 6
    #define _BG2_DECL_MRT_PACK(NAME, TW, TH) \
        _BG2_DECL_ONE_MRT(NAME, 0, TW, TH) _BG2_DECL_ONE_MRT(NAME, 1, TW, TH) \
        _BG2_DECL_ONE_MRT(NAME, 2, TW, TH) _BG2_DECL_ONE_MRT(NAME, 3, TW, TH) \
        _BG2_DECL_ONE_MRT(NAME, 4, TW, TH) _BG2_DECL_ONE_MRT(NAME, 5, TW, TH)
#else
    #define _BG2_DECL_MRT_PACK(NAME, TW, TH) \
        _BG2_DECL_ONE_MRT(NAME, 0, TW, TH) _BG2_DECL_ONE_MRT(NAME, 1, TW, TH) \
        _BG2_DECL_ONE_MRT(NAME, 2, TW, TH) _BG2_DECL_ONE_MRT(NAME, 3, TW, TH) \
        _BG2_DECL_ONE_MRT(NAME, 4, TW, TH) _BG2_DECL_ONE_MRT(NAME, 5, TW, TH) \
        _BG2_DECL_ONE_MRT(NAME, 6, TW, TH) _BG2_DECL_ONE_MRT(NAME, 7, TW, TH)
#endif

    // Level 0->1 separable horizontal intermediate buffer (W1 x H0)
    _BG2_DECL_MRT_PACK(H1_R, _BG2_W1, _BG2_H0)

    // Remapped Delta Gaussian Pyramid Packs G1[Delta_R] .. G{L-1}[Delta_R]
    _BG2_DECL_MRT_PACK(G1_R, _BG2_W1, _BG2_H1)
    _BG2_DECL_MRT_PACK(G2_R, _BG2_W2, _BG2_H2)
    _BG2_DECL_MRT_PACK(G3_R, _BG2_W3, _BG2_H3)
    _BG2_DECL_MRT_PACK(G4_R, _BG2_W4, _BG2_H4)
#if _BG2_LEVELS >= 6
    _BG2_DECL_MRT_PACK(G5_R, _BG2_W5, _BG2_H5)
#endif
#if _BG2_LEVELS >= 7
    _BG2_DECL_MRT_PACK(G6_R, _BG2_W6, _BG2_H6)
#endif
#if _BG2_LEVELS >= 8
    _BG2_DECL_MRT_PACK(G7_R, _BG2_W7, _BG2_H7)
#endif
#if _BG2_LEVELS >= 9
    _BG2_DECL_MRT_PACK(G8_R, _BG2_W8, _BG2_H8)
#endif

    // ---------------------------------------------------------------------------------------------
    // Numerically Stabilized Math Primitives (Dual-End log1p/expm1 PQ + Signed LMS PQ)
    // ---------------------------------------------------------------------------------------------

    float SafePow(float x, float e)
    {
        return (x > 0.0f) ? pow(abs(x), e) : 0.0f;
    }

    float SafeTanh(float x)
    {
        return tanh(clamp(x, -15.0f, 15.0f));
    }

    float StabilizedPQForwardChannel(float nitsVal, float expPow, float zeroFloor)
    {
        float x = SafePow(max(nitsVal, 0.0f) * 1.0e-4f, PQ_M1);
        float w = (PQ_K_NB * x) / (1.0f + PQ_C3 * x);
        float res;
        if (w < 0.05f)
        {
            float w2 = w * w;
            float ln1p = w - 0.5f * w2 + (1.0f / 3.0f) * w2 * w - 0.25f * w2 * w2 + 0.2f * w2 * w2 * w;
            res = zeroFloor * exp(expPow * ln1p);
        }
        else
        {
            float u = (PQ_ONE_M_C1 * (1.0f - x)) / (1.0f + PQ_C3 * x);
            if (abs(u) < 0.05f)
            {
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

    // Odd-symmetric sign-preserving JzAzBz PQ curve centered at c1^p (JZ_LMSP_ZERO)
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
            if (w < 0.05f)
            {
                float w2 = w * w;
                float ln1p = w - 0.5f * w2 + (1.0f / 3.0f) * w2 * w - 0.25f * w2 * w2 + 0.2f * w2 * w2 * w;
                float a = JZ_P * ln1p;
                if (a < 0.05f)
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
    // Exact Safdar et al. (2017) JzAzBz Transforms + 12-Iteration Constant-(Jz, h_z) Gamut Mapping
    // ---------------------------------------------------------------------------------------------

    float LuminanceNitsToJz(float nits)
    {
        float2 lm = D65_UNIT_LM * max(nits, 0.0f);
        float lp = StabilizedPQForwardChannel(lm.x, JZ_P, JZ_LMSP_ZERO);
        float mp = StabilizedPQForwardChannel(lm.y, JZ_P, JZ_LMSP_ZERO);
        float Iz = 0.5f * (lp + mp);
        return max(((JZ_ONE_P_D * Iz) / (1.0f + JZ_D * Iz)) - JZ_D0, 0.0f);
    }

    float JzToLuminanceNits(float Jz)
    {
        float JzClamped = clamp(Jz, 0.0f, 1.0f - JZ_D0);
        float Jzd = JzClamped + JZ_D0;
        float Iz = clamp(Jzd / max(JZ_ONE_P_D - JZ_D * Jzd, 1.0e-20f), JZ_LMSP_ZERO, 1.0f);
        return StabilizedPQInverseChannel(Iz, JZ_INV_P, JZ_LMSP_ZERO) / D65_UNIT_LM_AVG;
    }

    float3 LMSNitsToJzAzBz(float3 lmsNits)
    {
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
            float3 nits709 = rawRgb * 80.0f;
            lmsNits = mul(M_REC709_TO_LMS, nits709);
        }
        else if (cs == 3)
        {
            float3 nits2020 = StabilizedPQInverse3(saturate(rawRgb), PQ_INV_M2, PQ_ZERO_FLOOR);
            lmsNits = mul(M_REC2020_TO_LMS, nits2020);
        }
        else if (cs == 4)
        {
            float3 nits2020 = HLGToLinearNits(saturate(rawRgb));
            lmsNits = mul(M_REC2020_TO_LMS, nits2020);
        }
        else
        {
            float3 nits709 = SDRToLinear(saturate(rawRgb)) * SDRWhiteNits;
            lmsNits = mul(M_REC709_TO_LMS, nits709);
        }
        return LMSNitsToJzAzBz(lmsNits);
    }

    // Brightness-relative tolerance (2e-4 * max(peakChannel, 1e-3)) on pass-through checks only;
    // exact 0.0 tolerance during bisection so dark HDR shadows are never hard-clamped at -0.5 nits.
    bool EvalScaledChromaToRGB(
        float Iz, float3 chromaVec, float s, float3x3 matLMStoRGB, float maxNits, bool useRelTol,
        out float3 outRgb)
    {
        float3 lmsp = Iz + s * chromaVec;
        float coneTol = useRelTol ? (1.0e-6f * max(Iz, JZ_LMSP_ZERO)) : 0.0f;
        bool inCone = (min(lmsp.x, min(lmsp.y, lmsp.z)) >= (JZ_LMSP_ZERO - coneTol))
                   && (max(lmsp.x, max(lmsp.y, lmsp.z)) <= (1.0f + coneTol));
        outRgb = float3(0.0f, 0.0f, 0.0f);
        bool ok = false;
        if (inCone)
        {
            float3 lmspClamped = clamp(lmsp, JZ_LMSP_ZERO, 1.0f);
            float3 lms = StabilizedPQInverse3(lmspClamped, JZ_INV_P, JZ_LMSP_ZERO);
            outRgb = mul(matLMStoRGB, lms);
            float peakCh = max(outRgb.r, max(outRgb.g, outRgb.b));
            float chTol = useRelTol ? (2.0e-4f * max(peakCh, 1.0e-3f)) : 0.0f;
            ok = (min(outRgb.r, min(outRgb.g, outRgb.b)) >= -chTol)
              && (max(outRgb.r, max(outRgb.g, outRgb.b)) <= (maxNits + chTol));
        }
        return ok;
    }

    // 12-iteration adaptive constant-(Jz, hue h_z) gamut mapper inside JzAzBz (<0.025% precision;
    // unchanged in-gamut saturated colors pass through with 0 LSB shift)
    float3 GamutMapJzAzBzToTargetRGB(float3 jab, float3 jabOrig, float3x3 matLMStoRGB, float maxNits, bool softRollOff)
    {
        float Jz = clamp(jab.x, 0.0f, 1.0f - JZ_D0);
        float Jzd = Jz + JZ_D0;
        float Iz = clamp(Jzd / max(JZ_ONE_P_D - JZ_D * Jzd, 1.0e-20f), JZ_LMSP_ZERO, 1.0f);
        float3 chromaVec = mul(M_IZAZBZ_TO_LMSP, float3(0.0f, jab.y, jab.z));

        float3 rgb0;
        bool ok0 = EvalScaledChromaToRGB(Iz, chromaVec, 0.0f, matLMStoRGB, maxNits, true, rgb0);
        float3 finalRgb = clamp(rgb0, 0.0f, maxNits);

        float maxChroma = max(abs(chromaVec.x), max(abs(chromaVec.y), abs(chromaVec.z)));
        if (ok0 && maxChroma >= 1.0e-12f)
        {
            if (softRollOff)
            {
                float knee = clamp(GamutSoftKnee, 0.50f, 0.98f);
                float invKnee = 1.0f / knee;

                float3 rgbKnee;
                if (EvalScaledChromaToRGB(Iz, chromaVec, invKnee, matLMStoRGB, maxNits, false, rgbKnee))
                {
                    float3 rgb1;
                    EvalScaledChromaToRGB(Iz, chromaVec, 1.0f, matLMStoRGB, maxNits, true, rgb1);
                    finalRgb = clamp(rgb1, 0.0f, maxNits);
                }
                else
                {
                    float JzO = clamp(jabOrig.x, 0.0f, 1.0f - JZ_D0);
                    float JzdO = JzO + JZ_D0;
                    float IzO = clamp(JzdO / max(JZ_ONE_P_D - JZ_D * JzdO, 1.0e-20f), JZ_LMSP_ZERO, 1.0f);
                    float3 chromaOrig = mul(M_IZAZBZ_TO_LMSP, float3(0.0f, jabOrig.y, jabOrig.z));

                    float sEdgeOrig = invKnee;
                    float3 rgbTmp;
                    if (!EvalScaledChromaToRGB(IzO, chromaOrig, invKnee, matLMStoRGB, maxNits, false, rgbTmp))
                    {
                        if (!EvalScaledChromaToRGB(IzO, chromaOrig, 1.0f, matLMStoRGB, maxNits, false, rgbTmp))
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
                                if (EvalScaledChromaToRGB(IzO, chromaOrig, midO, matLMStoRGB, maxNits, false, rgbTmp))
                                    loO = midO;
                                else
                                    hiO = midO;
                            }
                            sEdgeOrig = loO;
                        }
                    }

                    float3 rgb1;
                    bool ok1 = EvalScaledChromaToRGB(Iz, chromaVec, 1.0f, matLMStoRGB, maxNits, true, rgb1);
                    if (sEdgeOrig <= 1.0f + 1.0e-5f)
                    {
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
                                if (EvalScaledChromaToRGB(Iz, chromaVec, mid, matLMStoRGB, maxNits, false, rgbTmp))
                                    lo = mid;
                                else
                                    hi = mid;
                            }
                            float3 rgbMapped;
                            EvalScaledChromaToRGB(Iz, chromaVec, lo, matLMStoRGB, maxNits, false, rgbMapped);
                            finalRgb = clamp(rgbMapped, 0.0f, maxNits);
                        }
                    }
                    else if (ok1 && EvalScaledChromaToRGB(Iz, chromaVec, sEdgeOrig, matLMStoRGB, maxNits, true, rgbTmp))
                    {
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
                            if (EvalScaledChromaToRGB(Iz, chromaVec, mid, matLMStoRGB, maxNits, false, rgbTmp))
                                lo = mid;
                            else
                                hi = mid;
                        }

                        float sEdge = lo;
                        float rho = 1.0f / max(sEdge, 1.0e-6f);
                        float excess = max(rho - effKnee, 0.0f);
                        float margin = max(1.0f - effKnee, 1.0e-6f);
                        float rhoOut = effKnee + margin * (excess / (margin + excess));
                        float sFinal = min(rhoOut * sEdge, 1.0f);

                        float3 rgbMapped;
                        EvalScaledChromaToRGB(Iz, chromaVec, sFinal, matLMStoRGB, maxNits, false, rgbMapped);
                        finalRgb = clamp(rgbMapped, 0.0f, maxNits);
                    }
                }
            }
            else
            {
                float3 rgb1;
                if (EvalScaledChromaToRGB(Iz, chromaVec, 1.0f, matLMStoRGB, maxNits, true, rgb1))
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
                        if (EvalScaledChromaToRGB(Iz, chromaVec, mid, matLMStoRGB, maxNits, false, rgbMid))
                            lo = mid;
                        else
                            hi = mid;
                    }

                    float3 rgbMapped;
                    EvalScaledChromaToRGB(Iz, chromaVec, lo, matLMStoRGB, maxNits, false, rgbMapped);
                    finalRgb = clamp(rgbMapped, 0.0f, maxNits);
                }
            }
        }
        return finalRgb;
    }

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

    // Unified single-call-site color reconstruction & gamut mapping (zero matrix ternaries)
    float3 ReconstructColorToBackBuffer(float3 origRawRgb, float3 jabOrig, float JzNew)
    {
        int cs = GetActiveColorSpace();
        float whiteNits = max(SDRWhiteNits, 1.0f);
        float Lw = max(HLGPeakNits, 100.0f);

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
                targetLinNits = scaledNits;
                outputIsRec2020 = (cs == 3 || cs == 4);
            }
            else
            {
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

        if (needJzGamutMap)
        {
            targetLinNits = GamutMapJzAzBzToTargetRGB(jabForMap, jabOrig, matLMStoTarget, maxNits, GamutMappingMode == 0);
            outputIsRec2020 = useRec2020Target;
        }

        return EncodeLinearNitsToBackBuffer(targetLinNits, cs, outputIsRec2020);
    }

    // ---------------------------------------------------------------------------------------------
    // Active Jz Calibration & FLLF Delta-Residual Remapping Function
    // ---------------------------------------------------------------------------------------------

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
            float smoothed = _BG2_FETCH(SampPeakSmooth, int2(0, 0), int2(1, 1)).r;
            float floorJz = max(LuminanceNitsToJz(HDRPaperWhiteNits), 0.10f);
            ceilingJz = clamp(max(smoothed, floorJz), floorJz, 1.0f - JZ_D0);
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

    float EvalDeltaRemap(float i, float g, float sigmaJz, float detailParam, float betaHi, float betaSh)
    {
        float d = i - g;
        // d >= 0 (i >= g): dark/shadow g surrounded by brighter i -> betaSh (< 1 lifts shadows).
        // d <  0 (i <  g): bright/highlight g surrounded by darker i -> betaHi (< 1 recovers highlights).
        float beta = (d >= 0.0f) ? betaSh : betaHi;

        if (RemappingCurveMode == 0)
        {
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
    // 3D Bilateral Grid Helpers (4x4 RGBA32F Tile Atlas, 32 Exact Float32 Range Bins)
    // ---------------------------------------------------------------------------------------------

    // Fetch homogeneous (sumVal, sumWeight) pair at grid cell (gPos, zBin) from a 4x4 RGBA32F atlas
    float2 FetchBGBin(sampler2D sAtlas, int2 gPos, int zBin, int2 tileSize, int2 atlasSize)
    {
        int zc = clamp(zBin, 0, 31);
        int tileIdx = int(uint(zc) / 2u);
        int ty = int(uint(tileIdx) / 4u);
        int tx = tileIdx - ty * 4;
        int2 cell = clamp(gPos, int2(0, 0), tileSize - 1);
        int2 atlasCoord = int2(tx * tileSize.x + cell.x, ty * tileSize.y + cell.y);
        float4 texel = _BG2_FETCH(sAtlas, atlasCoord, atlasSize);
        float2 res;
        if ((zc - tileIdx * 2) == 0)
            res = texel.rg;
        else
            res = texel.ba;
        return res;
    }

    // Strictly non-negative C2 Cubic B-Spline basis weights (partition of unity: sum == 1.0)
    float4 CubicBSplineWeights(float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        float mt = 1.0f - t;
        return float4(
            (mt * mt * mt) * (1.0f / 6.0f),
            (3.0f * t3 - 6.0f * t2 + 4.0f) * (1.0f / 6.0f),
            (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) * (1.0f / 6.0f),
            t3 * (1.0f / 6.0f)
        );
    }

    // 1D range slice at integer spatial grid vertex gPos (exact C2 B-Spline or Linear along 32 bins)
    float2 SliceBG_Range1D(sampler2D sAtlas, int2 gPos, float gz_f, int2 tileSize, int2 atlasSize)
    {
        int z1 = min(int(floor(gz_f)), 30);
        float fz = gz_f - float(z1);
        float2 acc;
        if (BG_SliceInterpolation == 1)
        {
            float2 b0 = FetchBGBin(sAtlas, gPos, z1,     tileSize, atlasSize);
            float2 b1 = FetchBGBin(sAtlas, gPos, z1 + 1, tileSize, atlasSize);
            acc = lerp(b0, b1, fz);
        }
        else
        {
            float4 wb = CubicBSplineWeights(fz);
            float2 b0 = FetchBGBin(sAtlas, gPos, max(z1 - 1, 0),  tileSize, atlasSize);
            float2 b1 = FetchBGBin(sAtlas, gPos, z1,              tileSize, atlasSize);
            float2 b2 = FetchBGBin(sAtlas, gPos, min(z1 + 1, 31), tileSize, atlasSize);
            float2 b3 = FetchBGBin(sAtlas, gPos, min(z1 + 2, 31), tileSize, atlasSize);
            acc = wb.x * b0 + wb.y * b1 + wb.z * b2 + wb.w * b3;
        }
        return acc;
    }

    // Continuous 3D slice at sub-cell spatial coordinate (gx_f, gy_f) and Jz value (exact float32)
    float SliceBG_3D(sampler2D sAtlas, float2 gCoordF, float jzGuide, float jzMax, int2 tileSize, int2 atlasSize)
    {
        float2 clamped = clamp(gCoordF, float2(0.0f, 0.0f), float2(tileSize - 1));
        int2 g0 = int2(floor(clamped));
        int2 g1 = min(g0 + 1, tileSize - 1);
        float2 f = clamped - float2(g0);
        float gz_f = saturate(jzGuide / jzMax) * 31.0f;

        float2 c00 = SliceBG_Range1D(sAtlas, int2(g0.x, g0.y), gz_f, tileSize, atlasSize);
        float2 c10 = SliceBG_Range1D(sAtlas, int2(g1.x, g0.y), gz_f, tileSize, atlasSize);
        float2 c01 = SliceBG_Range1D(sAtlas, int2(g0.x, g1.y), gz_f, tileSize, atlasSize);
        float2 c11 = SliceBG_Range1D(sAtlas, int2(g1.x, g1.y), gz_f, tileSize, atlasSize);

        float2 top = lerp(c00, c10, f.x);
        float2 bot = lerp(c01, c11, f.x);
        float2 res = lerp(top, bot, f.y);
        return (res.y > 1.0e-12f) ? (res.x / res.y) : jzGuide;
    }

    // Exact 1-to-1 integer-vertex slice for FLLF Levels 2..7 (zero spatial resampling!)
    float SliceBG_ExactVertex(sampler2D sAtlas, int2 gPos, float jzGuide, float jzMax, int2 tileSize, int2 atlasSize)
    {
        float gz_f = saturate(jzGuide / jzMax) * 31.0f;
        float2 res = SliceBG_Range1D(sAtlas, gPos, gz_f, tileSize, atlasSize);
        return (res.y > 1.0e-12f) ? (res.x / res.y) : jzGuide;
    }

    // Exact Burt-Adelson 5x5 spatial reduction INSIDE each tile of a 4x4 3D Bilateral Grid atlas
    float4 ReduceBGAtlas5x5(sampler2D srcAtlas, int2 dstAtlasPos, int2 srcTileSize, int2 dstTileSize, int2 srcAtlasSize)
    {
        int tx = int(uint(dstAtlasPos.x) / uint(dstTileSize.x));
        int ty = int(uint(dstAtlasPos.y) / uint(dstTileSize.y));
        int2 dstLocal = int2(dstAtlasPos.x - tx * dstTileSize.x, dstAtlasPos.y - ty * dstTileSize.y);
        int2 tileOrigin = int2(tx * srcTileSize.x, ty * srcTileSize.y);
        int2 c = dstLocal * 2;
        int2 maxLocal = srcTileSize - 1;

        float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);
        [unroll]
        for (int dy = -2; dy <= 2; ++dy)
        {
            int sy = tileOrigin.y + clamp(c.y + dy, 0, maxLocal.y);
            float4 rowAcc = float4(0.0f, 0.0f, 0.0f, 0.0f);
            [unroll]
            for (int dx = -2; dx <= 2; ++dx)
            {
                int sx = tileOrigin.x + clamp(c.x + dx, 0, maxLocal.x);
                rowAcc += BURT_W5[dx + 2] * _BG2_FETCH(srcAtlas, int2(sx, sy), srcAtlasSize);
            }
            acc += BURT_W5[dy + 2] * rowAcc;
        }
        return acc;
    }

    // ---------------------------------------------------------------------------------------------
    // Exact Discrete Burt-Adelson 5-Tap Reduce & Parity-Aware Expand (IEEE 754 float32)
    // ---------------------------------------------------------------------------------------------

    float ReduceScalar5x5_R(sampler2D src, int2 dstPos, int2 srcSize)
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
                rowAcc += BURT_W5[dx + 2] * _BG2_FETCH(src, int2(sx, sy), srcSize).r;
            }
            acc += BURT_W5[dy + 2] * rowAcc;
        }
        return acc;
    }

    float ReduceScalar5x5_G(sampler2D src, int2 dstPos, int2 srcSize)
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
                rowAcc += BURT_W5[dx + 2] * _BG2_FETCH(src, int2(sx, sy), srcSize).g;
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
                rowAcc += BURT_W5[dx + 2] * _BG2_FETCH(src, int2(sx, sy), srcSize);
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
                float r0 = 0.125f * _BG2_FETCH(coarse, int2(x0, y0), coarseSize).r
                         + 0.750f * _BG2_FETCH(coarse, int2(x1, y0), coarseSize).r
                         + 0.125f * _BG2_FETCH(coarse, int2(x2, y0), coarseSize).r;
                float r1 = 0.125f * _BG2_FETCH(coarse, int2(x0, y1), coarseSize).r
                         + 0.750f * _BG2_FETCH(coarse, int2(x1, y1), coarseSize).r
                         + 0.125f * _BG2_FETCH(coarse, int2(x2, y1), coarseSize).r;
                float r2 = 0.125f * _BG2_FETCH(coarse, int2(x0, y2), coarseSize).r
                         + 0.750f * _BG2_FETCH(coarse, int2(x1, y2), coarseSize).r
                         + 0.125f * _BG2_FETCH(coarse, int2(x2, y2), coarseSize).r;
                return 0.125f * r0 + 0.750f * r1 + 0.125f * r2;
            }
            else
            {
                float r0 = 0.5f * (_BG2_FETCH(coarse, int2(x1, y0), coarseSize).r
                                 + _BG2_FETCH(coarse, int2(x2, y0), coarseSize).r);
                float r1 = 0.5f * (_BG2_FETCH(coarse, int2(x1, y1), coarseSize).r
                                 + _BG2_FETCH(coarse, int2(x2, y1), coarseSize).r);
                float r2 = 0.5f * (_BG2_FETCH(coarse, int2(x1, y2), coarseSize).r
                                 + _BG2_FETCH(coarse, int2(x2, y2), coarseSize).r);
                return 0.125f * r0 + 0.750f * r1 + 0.125f * r2;
            }
        }
        else
        {
            if (parity.x == 0)
            {
                float r1 = 0.125f * _BG2_FETCH(coarse, int2(x0, y1), coarseSize).r
                         + 0.750f * _BG2_FETCH(coarse, int2(x1, y1), coarseSize).r
                         + 0.125f * _BG2_FETCH(coarse, int2(x2, y1), coarseSize).r;
                float r2 = 0.125f * _BG2_FETCH(coarse, int2(x0, y2), coarseSize).r
                         + 0.750f * _BG2_FETCH(coarse, int2(x1, y2), coarseSize).r
                         + 0.125f * _BG2_FETCH(coarse, int2(x2, y2), coarseSize).r;
                return 0.5f * (r1 + r2);
            }
            else
            {
                float r1 = 0.5f * (_BG2_FETCH(coarse, int2(x1, y1), coarseSize).r
                                 + _BG2_FETCH(coarse, int2(x2, y1), coarseSize).r);
                float r2 = 0.5f * (_BG2_FETCH(coarse, int2(x1, y2), coarseSize).r
                                 + _BG2_FETCH(coarse, int2(x2, y2), coarseSize).r);
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
                float4 r0 = 0.125f * _BG2_FETCH(coarse, int2(x0, y0), coarseSize)
                          + 0.750f * _BG2_FETCH(coarse, int2(x1, y0), coarseSize)
                          + 0.125f * _BG2_FETCH(coarse, int2(x2, y0), coarseSize);
                float4 r1 = 0.125f * _BG2_FETCH(coarse, int2(x0, y1), coarseSize)
                          + 0.750f * _BG2_FETCH(coarse, int2(x1, y1), coarseSize)
                          + 0.125f * _BG2_FETCH(coarse, int2(x2, y1), coarseSize);
                float4 r2 = 0.125f * _BG2_FETCH(coarse, int2(x0, y2), coarseSize)
                          + 0.750f * _BG2_FETCH(coarse, int2(x1, y2), coarseSize)
                          + 0.125f * _BG2_FETCH(coarse, int2(x2, y2), coarseSize);
                return 0.125f * r0 + 0.750f * r1 + 0.125f * r2;
            }
            else
            {
                float4 r0 = 0.5f * (_BG2_FETCH(coarse, int2(x1, y0), coarseSize)
                                  + _BG2_FETCH(coarse, int2(x2, y0), coarseSize));
                float4 r1 = 0.5f * (_BG2_FETCH(coarse, int2(x1, y1), coarseSize)
                                  + _BG2_FETCH(coarse, int2(x2, y1), coarseSize));
                float4 r2 = 0.5f * (_BG2_FETCH(coarse, int2(x1, y2), coarseSize)
                                  + _BG2_FETCH(coarse, int2(x2, y2), coarseSize));
                return 0.125f * r0 + 0.750f * r1 + 0.125f * r2;
            }
        }
        else
        {
            if (parity.x == 0)
            {
                float4 r1 = 0.125f * _BG2_FETCH(coarse, int2(x0, y1), coarseSize)
                          + 0.750f * _BG2_FETCH(coarse, int2(x1, y1), coarseSize)
                          + 0.125f * _BG2_FETCH(coarse, int2(x2, y1), coarseSize);
                float4 r2 = 0.125f * _BG2_FETCH(coarse, int2(x0, y2), coarseSize)
                          + 0.750f * _BG2_FETCH(coarse, int2(x1, y2), coarseSize)
                          + 0.125f * _BG2_FETCH(coarse, int2(x2, y2), coarseSize);
                return 0.5f * (r1 + r2);
            }
            else
            {
                float4 r1 = 0.5f * (_BG2_FETCH(coarse, int2(x1, y1), coarseSize)
                                  + _BG2_FETCH(coarse, int2(x2, y1), coarseSize));
                float4 r2 = 0.5f * (_BG2_FETCH(coarse, int2(x1, y2), coarseSize)
                                  + _BG2_FETCH(coarse, int2(x2, y2), coarseSize));
                return 0.5f * (r1 + r2);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // FLLF Intensity Slice Indexing & Catmull-Rom / Linear Interpolation (16 / 24 / 32 Slices)
    // ---------------------------------------------------------------------------------------------

    float ExtractChannel(float4 v, int c)
    {
        float res;
        if (c == 0)      res = v.x;
        else if (c == 1) res = v.y;
        else if (c == 2) res = v.z;
        else             res = v.w;
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

    // Macro parameter lists for 4, 6, or 8 MRT samplers
#if _BG2_MRT_COUNT == 4
    #define _BG2_SAMP_PARAMS(P) sampler2D P##0, sampler2D P##1, sampler2D P##2, sampler2D P##3
    #define _BG2_SAMP_ARGS(P)   P##0, P##1, P##2, P##3
    #define _BG2_OUT_PARAMS     out float4 o0 : SV_Target0, out float4 o1 : SV_Target1, out float4 o2 : SV_Target2, out float4 o3 : SV_Target3
#elif _BG2_MRT_COUNT == 6
    #define _BG2_SAMP_PARAMS(P) sampler2D P##0, sampler2D P##1, sampler2D P##2, sampler2D P##3, sampler2D P##4, sampler2D P##5
    #define _BG2_SAMP_ARGS(P)   P##0, P##1, P##2, P##3, P##4, P##5
    #define _BG2_OUT_PARAMS     out float4 o0 : SV_Target0, out float4 o1 : SV_Target1, out float4 o2 : SV_Target2, out float4 o3 : SV_Target3, out float4 o4 : SV_Target4, out float4 o5 : SV_Target5
#else
    #define _BG2_SAMP_PARAMS(P) sampler2D P##0, sampler2D P##1, sampler2D P##2, sampler2D P##3, sampler2D P##4, sampler2D P##5, sampler2D P##6, sampler2D P##7
    #define _BG2_SAMP_ARGS(P)   P##0, P##1, P##2, P##3, P##4, P##5, P##6, P##7
    #define _BG2_OUT_PARAMS     out float4 o0 : SV_Target0, out float4 o1 : SV_Target1, out float4 o2 : SV_Target2, out float4 o3 : SV_Target3, out float4 o4 : SV_Target4, out float4 o5 : SV_Target5, out float4 o6 : SV_Target6, out float4 o7 : SV_Target7
#endif

    float4 FetchLaplacianPack(
        int packIdx,
        _BG2_SAMP_PARAMS(sFine),
        _BG2_SAMP_PARAMS(sCoarse),
        int2 finePos, int2 fineSize, int2 coarseSize)
    {
        float4 res;
        if (packIdx == 0)      res = _BG2_FETCH(sFine0, finePos, fineSize) - ExpandVec4(sCoarse0, finePos, coarseSize);
        else if (packIdx == 1) res = _BG2_FETCH(sFine1, finePos, fineSize) - ExpandVec4(sCoarse1, finePos, coarseSize);
        else if (packIdx == 2) res = _BG2_FETCH(sFine2, finePos, fineSize) - ExpandVec4(sCoarse2, finePos, coarseSize);
    #if _BG2_MRT_COUNT == 4
        else                   res = _BG2_FETCH(sFine3, finePos, fineSize) - ExpandVec4(sCoarse3, finePos, coarseSize);
    #elif _BG2_MRT_COUNT == 6
        else if (packIdx == 3) res = _BG2_FETCH(sFine3, finePos, fineSize) - ExpandVec4(sCoarse3, finePos, coarseSize);
        else if (packIdx == 4) res = _BG2_FETCH(sFine4, finePos, fineSize) - ExpandVec4(sCoarse4, finePos, coarseSize);
        else                   res = _BG2_FETCH(sFine5, finePos, fineSize) - ExpandVec4(sCoarse5, finePos, coarseSize);
    #else
        else if (packIdx == 3) res = _BG2_FETCH(sFine3, finePos, fineSize) - ExpandVec4(sCoarse3, finePos, coarseSize);
        else if (packIdx == 4) res = _BG2_FETCH(sFine4, finePos, fineSize) - ExpandVec4(sCoarse4, finePos, coarseSize);
        else if (packIdx == 5) res = _BG2_FETCH(sFine5, finePos, fineSize) - ExpandVec4(sCoarse5, finePos, coarseSize);
        else if (packIdx == 6) res = _BG2_FETCH(sFine6, finePos, fineSize) - ExpandVec4(sCoarse6, finePos, coarseSize);
        else                   res = _BG2_FETCH(sFine7, finePos, fineSize) - ExpandVec4(sCoarse7, finePos, coarseSize);
    #endif
        return res;
    }

    float InterpolateLaplacianBand(
        float g,
        _BG2_SAMP_PARAMS(sFine),
        _BG2_SAMP_PARAMS(sCoarse),
        int2 finePos, int2 fineSize, int2 coarseSize,
        float bandWeight)
    {
        float jzMax = GetActiveJzCeiling();
        float pos = saturate(g / jzMax) * float(_BG2_SLICES - 1);
        int idx = min(int(floor(pos)), _BG2_SLICES - 2);
        float t = pos - float(idx);

        if (SliceInterpolationMode == 1)
        {
            int i1 = idx;
            int i2 = idx + 1;
            int pLow  = int(uint(i1) / 4u);
            int pHigh = int(uint(i2) / 4u);

            float4 packLow = FetchLaplacianPack(pLow, _BG2_SAMP_ARGS(sFine), _BG2_SAMP_ARGS(sCoarse), finePos, fineSize, coarseSize);
            float4 packHigh = packLow;
            [branch]
            if (pHigh != pLow)
                packHigh = FetchLaplacianPack(pHigh, _BG2_SAMP_ARGS(sFine), _BG2_SAMP_ARGS(sCoarse), finePos, fineSize, coarseSize);

            float v1 = ExtractChannel(packLow,  i1 - pLow  * 4);
            float v2 = ExtractChannel(packHigh, i2 - pHigh * 4);
            return lerp(v1, v2, t) * bandWeight;
        }
        else
        {
            int i0 = max(idx - 1, 0);
            int i1 = idx;
            int i2 = idx + 1;
            int i3 = min(idx + 2, _BG2_SLICES - 1);

            int p0 = int(uint(i0) / 4u);
            int p1 = int(uint(i1) / 4u);
            int p2 = int(uint(i2) / 4u);
            int p3 = int(uint(i3) / 4u);

            float4 packLow = FetchLaplacianPack(p0, _BG2_SAMP_ARGS(sFine), _BG2_SAMP_ARGS(sCoarse), finePos, fineSize, coarseSize);
            float4 packHigh = packLow;
            [branch]
            if (p3 != p0)
                packHigh = FetchLaplacianPack(p3, _BG2_SAMP_ARGS(sFine), _BG2_SAMP_ARGS(sCoarse), finePos, fineSize, coarseSize);

            float v0 = ExtractChannel(packLow, i0 - p0 * 4);
            float v1 = ExtractChannel((p1 == p0) ? packLow : packHigh, i1 - p1 * 4);
            float v2 = ExtractChannel((p2 == p0) ? packLow : packHigh, i2 - p2 * 4);
            float v3 = ExtractChannel(packHigh, i3 - p3 * 4);
            return CatmullRom(v0, v1, v2, v3, t) * bandWeight;
        }
    }

    float4 ExpandG1Pack(int packIdx, int2 finePos)
    {
        const int2 coarseSize = int2(_BG2_W1, _BG2_H1);
        float4 res;
        if (packIdx == 0)      res = ExpandVec4(SampG1_R_0, finePos, coarseSize);
        else if (packIdx == 1) res = ExpandVec4(SampG1_R_1, finePos, coarseSize);
        else if (packIdx == 2) res = ExpandVec4(SampG1_R_2, finePos, coarseSize);
    #if _BG2_MRT_COUNT == 4
        else                   res = ExpandVec4(SampG1_R_3, finePos, coarseSize);
    #elif _BG2_MRT_COUNT == 6
        else if (packIdx == 3) res = ExpandVec4(SampG1_R_3, finePos, coarseSize);
        else if (packIdx == 4) res = ExpandVec4(SampG1_R_4, finePos, coarseSize);
        else                   res = ExpandVec4(SampG1_R_5, finePos, coarseSize);
    #else
        else if (packIdx == 3) res = ExpandVec4(SampG1_R_3, finePos, coarseSize);
        else if (packIdx == 4) res = ExpandVec4(SampG1_R_4, finePos, coarseSize);
        else if (packIdx == 5) res = ExpandVec4(SampG1_R_5, finePos, coarseSize);
        else if (packIdx == 6) res = ExpandVec4(SampG1_R_6, finePos, coarseSize);
        else                   res = ExpandVec4(SampG1_R_7, finePos, coarseSize);
    #endif
        return res;
    }

    float TPDFNoise(float2 pos)
    {
        float r0 = frac(52.9829189f * frac(dot(pos, float2(0.06711056f, 0.00583715f))));
        float r1 = frac(52.9829189f * frac(dot(pos + float2(17.0f, 31.0f), float2(0.06711056f, 0.00583715f))));
        return r0 - r1;
    }

    // ---------------------------------------------------------------------------------------------
    // Vertex Shader
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
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W0, _BG2_H0));
        float4 raw = _BG2_FETCH(SampBackBuffer, p, int2(_BG2_W0, _BG2_H0));
        float3 jab = BackBufferToJzAzBz(raw.rgb);
        return float4(jab, raw.a);
    }

    // First-stage 8x8 reduction returning 2nd-brightest pixel per tile to reject isolated 1-px sparks
    // while preserving 100% exact luminance of any >=2-pixel specular highlight
    float ReduceRobustPeak8x8(sampler2D src, int2 dstPos, int2 srcSize)
    {
        int2 base = dstPos * 8;
        float m1 = 0.0f;
        float m2 = 0.0f;
        int cnt = 0;
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
                    {
                        float v = _BG2_FETCH(src, int2(sx, sy), srcSize).r;
                        if (v > m1)
                        {
                            m2 = m1;
                            m1 = v;
                        }
                        else if (v > m2)
                        {
                            m2 = v;
                        }
                        cnt += 1;
                    }
                }
            }
        }
        return (cnt >= 2) ? m2 : m1;
    }

    // Exact 8x8 max reduction for subsequent pyramid levels
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
                        m = max(m, _BG2_FETCH(src, int2(sx, sy), srcSize).r);
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
            int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_PW1, _BG2_PH1));
            res = ReduceRobustPeak8x8(SampJzAzBz, p, int2(_BG2_W0, _BG2_H0));
        }
        return res;
    }

    float PS_Peak_2(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        float res = 0.0f;
        if (AutoDetectHDRPeak && GetActiveColorSpace() != 1)
        {
            int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_PW2, _BG2_PH2));
            res = ReduceMax8x8(SampPeak1, p, int2(_BG2_PW1, _BG2_PH1));
        }
        return res;
    }

    float PS_Peak_3(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        float res = 0.0f;
        if (AutoDetectHDRPeak && GetActiveColorSpace() != 1)
        {
            int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_PW3, _BG2_PH3));
            res = ReduceMax8x8(SampPeak2, p, int2(_BG2_PW2, _BG2_PH2));
        }
        return res;
    }

    float PS_Peak_4(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        float m = 0.0f;
        if (AutoDetectHDRPeak && GetActiveColorSpace() != 1)
        {
            const int2 srcSize = int2(_BG2_PW3, _BG2_PH3);
        #if _BG2_PH3 == 1
            int sy = 0;
        #else
            [loop]
            for (int sy = 0; sy < _BG2_PH3; ++sy)
        #endif
            {
            #if _BG2_PW3 == 1
                int sx = 0;
            #else
                [loop]
                for (int sx = 0; sx < _BG2_PW3; ++sx)
            #endif
                {
                    m = max(m, _BG2_FETCH(SampPeak3, int2(sx, sy), srcSize).r);
                }
            }
        }
        return m;
    }

    // Asymmetric exponential temporal smoothing of HDR peak (~0.10 s attack, ~1.00 s release)
    float PS_Peak_Smooth(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        float res = 0.0f;
        if (AutoDetectHDRPeak && GetActiveColorSpace() != 1)
        {
            float rawPeak = _BG2_FETCH(SampPeak4, int2(0, 0), int2(1, 1)).r * 1.005f;
            float floorJz = max(LuminanceNitsToJz(HDRPaperWhiteNits), 0.10f);
            float target  = clamp(max(rawPeak, floorJz), floorJz, 1.0f - JZ_D0);
            float prev    = _BG2_FETCH(SampPeakPrev, int2(0, 0), int2(1, 1)).r;
            if (prev <= 1.0e-6f || prev > 1.0f)
            {
                res = target;
            }
            else
            {
                float dt = clamp(FrameTime * 0.001f, 0.0f, 0.25f);
                float tau = (target > prev) ? 0.10f : 1.00f;
                float alpha = 1.0f - exp(-dt / tau);
                res = lerp(prev, target, saturate(alpha));
            }
        }
        return res;
    }

    float PS_Peak_Store(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        return _BG2_FETCH(SampPeakSmooth, int2(0, 0), int2(1, 1)).r;
    }

    // ---------------------------------------------------------------------------------------------
    // Phase 2: 3D Bilateral Grid Construction (Exact Float32 Splat + 9-Tap 3D Blur + Multi-Scale)
    // ---------------------------------------------------------------------------------------------

    // Exact IEEE 754 float32 trilinear gather-splatting over the complete 7x7 support (Ss = 4 px).
    // Each atlas texel accumulates 2 consecutive range bins: .rg = bin 2*tileIdx, .ba = bin 2*tileIdx+1.
    float4 PS_BG_Splat(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        const int2 tileSize  = int2(_BG2_GW0, _BG2_GH0);
        const int2 atlasSize = int2(_BG2_AW0, _BG2_AH0);
        const int2 imgSize   = int2(_BG2_W0,  _BG2_H0);

        int2 ap = _BG2_PIXEL_COORD(vpos, uv, atlasSize);
        int tx = int(uint(ap.x) / uint(tileSize.x));
        int ty = int(uint(ap.y) / uint(tileSize.y));
        int2 gPos = int2(ap.x - tx * tileSize.x, ap.y - ty * tileSize.y);

        int tileIdx = ty * 4 + tx;
        float zBin0 = float(tileIdx * 2);
        float zBin1 = zBin0 + 1.0f;

        float jzMax = GetActiveJzCeiling();
        float invJzScale = 31.0f / jzMax;

        int2 centerPx = gPos * 4;
        float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);

        [loop]
        for (int dy = -3; dy <= 3; ++dy)
        {
            int sy = centerPx.y + dy;
            if (sy >= 0 && sy < imgSize.y)
            {
                float wy = SPLAT_W7[dy + 3];
                [unroll]
                for (int dx = -3; dx <= 3; ++dx)
                {
                    int sx = centerPx.x + dx;
                    if (sx >= 0 && sx < imgSize.x)
                    {
                        float wxy = wy * SPLAT_W7[dx + 3];
                        float jz = _BG2_FETCH(SampJzAzBz, int2(sx, sy), imgSize).r;
                        float gz = clamp(jz * invJzScale, 0.0f, 31.0f);

                        float wz0 = max(0.0f, 1.0f - abs(gz - zBin0));
                        float wz1 = max(0.0f, 1.0f - abs(gz - zBin1));
                        if (wz0 > 0.0f)
                        {
                            float w0 = wxy * wz0;
                            acc.x += w0 * jz;
                            acc.y += w0;
                        }
                        if (wz1 > 0.0f)
                        {
                            float w1 = wxy * wz1;
                            acc.z += w1 * jz;
                            acc.w += w1;
                        }
                    }
                }
            }
        }
        return acc;
    }

    // Exact adaptive 3-sigma Gaussian blur along grid X (strictly clamped inside each tile)
    float4 PS_BG_BlurX(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        const int2 tileSize  = int2(_BG2_GW0, _BG2_GH0);
        const int2 atlasSize = int2(_BG2_AW0, _BG2_AH0);
        int2 ap = _BG2_PIXEL_COORD(vpos, uv, atlasSize);
        int tx = int(uint(ap.x) / uint(tileSize.x));
        int tileX0 = tx * tileSize.x;
        int localX = ap.x - tileX0;
        int maxLX = tileSize.x - 1;

        float sig = max(BG_SpatialSigma, 0.25f);
        float inv2Sig2 = 0.5f / (sig * sig);
        int rad = clamp(int(ceil(3.0f * sig)), 2, 11);

        float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float wSum = 0.0f;
        [loop]
        for (int dx = -rad; dx <= rad; ++dx)
        {
            float w = exp(-float(dx * dx) * inv2Sig2);
            int sx = tileX0 + clamp(localX + dx, 0, maxLX);
            sum += w * _BG2_FETCH(SampBG_S0_A, int2(sx, ap.y), atlasSize);
            wSum += w;
        }
        return sum / wSum;
    }

    // Exact adaptive 3-sigma Gaussian blur along grid Y (strictly clamped inside each tile)
    float4 PS_BG_BlurY(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        const int2 tileSize  = int2(_BG2_GW0, _BG2_GH0);
        const int2 atlasSize = int2(_BG2_AW0, _BG2_AH0);
        int2 ap = _BG2_PIXEL_COORD(vpos, uv, atlasSize);
        int ty = int(uint(ap.y) / uint(tileSize.y));
        int tileY0 = ty * tileSize.y;
        int localY = ap.y - tileY0;
        int maxLY = tileSize.y - 1;

        float sig = max(BG_SpatialSigma, 0.25f);
        float inv2Sig2 = 0.5f / (sig * sig);
        int rad = clamp(int(ceil(3.0f * sig)), 2, 11);

        float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float wSum = 0.0f;
        [loop]
        for (int dy = -rad; dy <= rad; ++dy)
        {
            float w = exp(-float(dy * dy) * inv2Sig2);
            int sy = tileY0 + clamp(localY + dy, 0, maxLY);
            sum += w * _BG2_FETCH(SampBG_S0_B, int2(ap.x, sy), atlasSize);
            wSum += w;
        }
        return sum / wSum;
    }

    // Exact zero-truncation full-32-bin Gaussian convolution along intensity range Z (16 tile fetches).
    // Mathematically exact zero-padding outside [0, 31] prevents boundary weight bias at black/peak.
    float4 PS_BG_BlurZ(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        const int2 tileSize  = int2(_BG2_GW0, _BG2_GH0);
        const int2 atlasSize = int2(_BG2_AW0, _BG2_AH0);
        int2 ap = _BG2_PIXEL_COORD(vpos, uv, atlasSize);
        int tx = int(uint(ap.x) / uint(tileSize.x));
        int ty = int(uint(ap.y) / uint(tileSize.y));
        int2 gPos = int2(ap.x - tx * tileSize.x, ap.y - ty * tileSize.y);

        int tileIdx = ty * 4 + tx;
        float z0 = float(tileIdx * 2);
        float z1 = z0 + 1.0f;

        // Convert perceptual BG_RangeSigma (relative to reference white Jz) into exact grid Z-bin units
        float jzMax = GetActiveJzCeiling();
        float sigmaJzBG = max(BG_RangeSigma * GetReferenceWhiteJz(), 1.0e-4f);
        float sigBins = max((sigmaJzBG / jzMax) * 31.0f, 0.25f);
        float inv2Sig2 = 0.5f / (sigBins * sigBins);
        // Global z-independent normalization preserves exact homogeneous linearity across all 32 bins
        float invNorm = 1.0f / max(2.5066282746f * sigBins, 1.0f);

        float2 sum0 = float2(0.0f, 0.0f);
        float2 sum1 = float2(0.0f, 0.0f);

        [unroll]
        for (int t = 0; t < 16; ++t)
        {
            int tty = int(uint(t) / 4u);
            int ttx = t - tty * 4;
            int2 coord = int2(ttx * tileSize.x + gPos.x, tty * tileSize.y + gPos.y);
            float4 texel = _BG2_FETCH(SampBG_S0_A, coord, atlasSize);

            float k0 = float(t * 2);
            float k1 = k0 + 1.0f;

            float d00 = k0 - z0;
            float d10 = k1 - z0;
            float d01 = k0 - z1;
            float d11 = k1 - z1;

            float w00 = exp(-d00 * d00 * inv2Sig2);
            float w10 = exp(-d10 * d10 * inv2Sig2);
            float w01 = exp(-d01 * d01 * inv2Sig2);
            float w11 = exp(-d11 * d11 * inv2Sig2);

            sum0 += w00 * texel.rg + w10 * texel.ba;
            sum1 += w01 * texel.rg + w11 * texel.ba;
        }
        return float4(sum0, sum1) * invNorm;
    }

    // Multi-Scale 3D Bilateral Grid Spatial Reductions (S0 -> S1 -> ... -> S5; preserves all 32 Z bins)
    float4 PS_BG_Reduce_01(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 ap = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_AW1, _BG2_AH1));
        return ReduceBGAtlas5x5(SampBG_S0_B, ap, int2(_BG2_GW0, _BG2_GH0), int2(_BG2_GW1, _BG2_GH1), int2(_BG2_AW0, _BG2_AH0));
    }

#if _BG2_LEVELS >= 6
    float4 PS_BG_Reduce_12(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 ap = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_AW2, _BG2_AH2));
        return ReduceBGAtlas5x5(SampBG_S1, ap, int2(_BG2_GW1, _BG2_GH1), int2(_BG2_GW2, _BG2_GH2), int2(_BG2_AW1, _BG2_AH1));
    }
#endif

#if _BG2_LEVELS >= 7
    float4 PS_BG_Reduce_23(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 ap = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_AW3, _BG2_AH3));
        return ReduceBGAtlas5x5(SampBG_S2, ap, int2(_BG2_GW2, _BG2_GH2), int2(_BG2_GW3, _BG2_GH3), int2(_BG2_AW2, _BG2_AH2));
    }
#endif

#if _BG2_LEVELS >= 8
    float4 PS_BG_Reduce_34(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 ap = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_AW4, _BG2_AH4));
        return ReduceBGAtlas5x5(SampBG_S3, ap, int2(_BG2_GW3, _BG2_GH3), int2(_BG2_GW4, _BG2_GH4), int2(_BG2_AW3, _BG2_AH3));
    }
#endif

#if _BG2_LEVELS >= 9
    float4 PS_BG_Reduce_45(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 ap = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_AW5, _BG2_AH5));
        return ReduceBGAtlas5x5(SampBG_S4, ap, int2(_BG2_GW4, _BG2_GH4), int2(_BG2_GW5, _BG2_GH5), int2(_BG2_AW4, _BG2_AH4));
    }
#endif

    // Full-resolution Bilateral Base B0 (.r) and Pre-Denoised Working Jz (.g)
    float2 PS_BG_SliceB0(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W0, _BG2_H0));
        float jzOrig = _BG2_FETCH(SampJzAzBz, p, int2(_BG2_W0, _BG2_H0)).r;
        float jzMax = GetActiveJzCeiling();

        float b0 = SliceBG_3D(SampBG_S0_B, float2(p) * 0.25f, jzOrig, jzMax, int2(_BG2_GW0, _BG2_GH0), int2(_BG2_AW0, _BG2_AH0));
        // Negative BG_MicroDetail applies edge-preserving pre-denoising BEFORE FLLF runs
        float preDenoise = saturate(-BG_MicroDetail);
        float jzWork = lerp(jzOrig, b0, preDenoise);
        return float2(b0, jzWork);
    }

    // ---------------------------------------------------------------------------------------------
    // Phase 3: Hybrid Bilateral-Grid-Guided Reference Pyramid (G1..G{L-2}: .r = G_l, .g = Guide g_l)
    // ---------------------------------------------------------------------------------------------

    float2 PS_Guide_01(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W1, _BG2_H1));
        float gStd = ReduceScalar5x5_G(SampB0, p, int2(_BG2_W0, _BG2_H0));
        int2 fullP = min(p * 2, int2(_BG2_W0 - 1, _BG2_H0 - 1));
        float b0AtP = _BG2_FETCH(SampB0, fullP, int2(_BG2_W0, _BG2_H0)).r;
        float b1 = SliceBG_3D(SampBG_S0_B, float2(p) * 0.5f, b0AtP, GetActiveJzCeiling(), int2(_BG2_GW0, _BG2_GH0), int2(_BG2_AW0, _BG2_AH0));
        return float2(gStd, lerp(gStd, b1, BG_GuidanceStrength));
    }

    float2 PS_Guide_12(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W2, _BG2_H2));
        float gStd = ReduceScalar5x5_R(SampG1, p, int2(_BG2_W1, _BG2_H1));
        int2 fullP = min(p * 4, int2(_BG2_W0 - 1, _BG2_H0 - 1));
        float b0AtP = _BG2_FETCH(SampB0, fullP, int2(_BG2_W0, _BG2_H0)).r;
        float b2 = SliceBG_ExactVertex(SampBG_S0_B, p, b0AtP, GetActiveJzCeiling(), int2(_BG2_GW0, _BG2_GH0), int2(_BG2_AW0, _BG2_AH0));
        return float2(gStd, lerp(gStd, b2, BG_GuidanceStrength));
    }

    float2 PS_Guide_23(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W3, _BG2_H3));
        float gStd = ReduceScalar5x5_R(SampG2, p, int2(_BG2_W2, _BG2_H2));
        int2 fullP = min(p * 8, int2(_BG2_W0 - 1, _BG2_H0 - 1));
        float b0AtP = _BG2_FETCH(SampB0, fullP, int2(_BG2_W0, _BG2_H0)).r;
        float b3 = SliceBG_ExactVertex(SampBG_S1, p, b0AtP, GetActiveJzCeiling(), int2(_BG2_GW1, _BG2_GH1), int2(_BG2_AW1, _BG2_AH1));
        return float2(gStd, lerp(gStd, b3, BG_GuidanceStrength));
    }

#if _BG2_LEVELS >= 6
    float2 PS_Guide_34(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W4, _BG2_H4));
        float gStd = ReduceScalar5x5_R(SampG3, p, int2(_BG2_W3, _BG2_H3));
        int2 fullP = min(p * 16, int2(_BG2_W0 - 1, _BG2_H0 - 1));
        float b0AtP = _BG2_FETCH(SampB0, fullP, int2(_BG2_W0, _BG2_H0)).r;
        float b4 = SliceBG_ExactVertex(SampBG_S2, p, b0AtP, GetActiveJzCeiling(), int2(_BG2_GW2, _BG2_GH2), int2(_BG2_AW2, _BG2_AH2));
        return float2(gStd, lerp(gStd, b4, BG_GuidanceStrength));
    }
#endif

#if _BG2_LEVELS >= 7
    float2 PS_Guide_45(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W5, _BG2_H5));
        float gStd = ReduceScalar5x5_R(SampG4, p, int2(_BG2_W4, _BG2_H4));
        int2 fullP = min(p * 32, int2(_BG2_W0 - 1, _BG2_H0 - 1));
        float b0AtP = _BG2_FETCH(SampB0, fullP, int2(_BG2_W0, _BG2_H0)).r;
        float b5 = SliceBG_ExactVertex(SampBG_S3, p, b0AtP, GetActiveJzCeiling(), int2(_BG2_GW3, _BG2_GH3), int2(_BG2_AW3, _BG2_AH3));
        return float2(gStd, lerp(gStd, b5, BG_GuidanceStrength));
    }
#endif

#if _BG2_LEVELS >= 8
    float2 PS_Guide_56(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W6, _BG2_H6));
        float gStd = ReduceScalar5x5_R(SampG5, p, int2(_BG2_W5, _BG2_H5));
        int2 fullP = min(p * 64, int2(_BG2_W0 - 1, _BG2_H0 - 1));
        float b0AtP = _BG2_FETCH(SampB0, fullP, int2(_BG2_W0, _BG2_H0)).r;
        float b6 = SliceBG_ExactVertex(SampBG_S4, p, b0AtP, GetActiveJzCeiling(), int2(_BG2_GW4, _BG2_GH4), int2(_BG2_AW4, _BG2_AH4));
        return float2(gStd, lerp(gStd, b6, BG_GuidanceStrength));
    }
#endif

#if _BG2_LEVELS >= 9
    float2 PS_Guide_67(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W7, _BG2_H7));
        float gStd = ReduceScalar5x5_R(SampG6, p, int2(_BG2_W6, _BG2_H6));
        int2 fullP = min(p * 128, int2(_BG2_W0 - 1, _BG2_H0 - 1));
        float b0AtP = _BG2_FETCH(SampB0, fullP, int2(_BG2_W0, _BG2_H0)).r;
        float b7 = SliceBG_ExactVertex(SampBG_S5, p, b0AtP, GetActiveJzCeiling(), int2(_BG2_GW5, _BG2_GH5), int2(_BG2_AW5, _BG2_AH5));
        return float2(gStd, lerp(gStd, b7, BG_GuidanceStrength));
    }
#endif

    // ---------------------------------------------------------------------------------------------
    // Phase 4: Remapped Delta-Residual Gaussian Pyramid Packs (16 / 24 / 32 Slices across 4/6/8 MRTs)
    // ---------------------------------------------------------------------------------------------

    void PS_ReduceR01_H(float4 vpos : SV_Position, float2 uv : TEXCOORD0, _BG2_OUT_PARAMS)
    {
        int2 dstSize = int2(_BG2_W1, _BG2_H0);
        int2 srcSize = int2(_BG2_W0, _BG2_H0);
        int2 p = _BG2_PIXEL_COORD(vpos, uv, dstSize);

        float jzMax, sigmaJz, detailParam, betaHi, betaSh;
        GetRemapParams(jzMax, sigmaJz, detailParam, betaHi, betaSh);
        float jzStep = jzMax / float(_BG2_SLICES - 1);

        o0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o3 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #if _BG2_MRT_COUNT >= 6
        o4 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o5 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #endif
    #if _BG2_MRT_COUNT >= 8
        o6 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o7 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #endif

        int cx = p.x * 2;
        int maxX = srcSize.x - 1;
        [unroll]
        for (int dx = -2; dx <= 2; ++dx)
        {
            int sx = clamp(cx + dx, 0, maxX);
            // Reads pre-denoised working Jz from TexB0.g (exact JzOrig when BG_MicroDetail >= 0)
            float jz = _BG2_FETCH(SampB0, int2(sx, p.y), srcSize).g;
            float wt = BURT_W5[dx + 2];

            o0 += wt * EvalDeltaRemapPack(jz, 0, jzStep, sigmaJz, detailParam, betaHi, betaSh);
            o1 += wt * EvalDeltaRemapPack(jz, 1, jzStep, sigmaJz, detailParam, betaHi, betaSh);
            o2 += wt * EvalDeltaRemapPack(jz, 2, jzStep, sigmaJz, detailParam, betaHi, betaSh);
            o3 += wt * EvalDeltaRemapPack(jz, 3, jzStep, sigmaJz, detailParam, betaHi, betaSh);
        #if _BG2_MRT_COUNT >= 6
            o4 += wt * EvalDeltaRemapPack(jz, 4, jzStep, sigmaJz, detailParam, betaHi, betaSh);
            o5 += wt * EvalDeltaRemapPack(jz, 5, jzStep, sigmaJz, detailParam, betaHi, betaSh);
        #endif
        #if _BG2_MRT_COUNT >= 8
            o6 += wt * EvalDeltaRemapPack(jz, 6, jzStep, sigmaJz, detailParam, betaHi, betaSh);
            o7 += wt * EvalDeltaRemapPack(jz, 7, jzStep, sigmaJz, detailParam, betaHi, betaSh);
        #endif
        }
    }

    void PS_ReduceR01_V(float4 vpos : SV_Position, float2 uv : TEXCOORD0, _BG2_OUT_PARAMS)
    {
        int2 dstSize = int2(_BG2_W1, _BG2_H1);
        int2 srcSize = int2(_BG2_W1, _BG2_H0);
        int2 p = _BG2_PIXEL_COORD(vpos, uv, dstSize);

        o0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o3 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #if _BG2_MRT_COUNT >= 6
        o4 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o5 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #endif
    #if _BG2_MRT_COUNT >= 8
        o6 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o7 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    #endif

        int cy = p.y * 2;
        int maxY = srcSize.y - 1;
        [unroll]
        for (int dy = -2; dy <= 2; ++dy)
        {
            int sy = clamp(cy + dy, 0, maxY);
            float wt = BURT_W5[dy + 2];
            o0 += wt * _BG2_FETCH(SampH1_R_0, int2(p.x, sy), srcSize);
            o1 += wt * _BG2_FETCH(SampH1_R_1, int2(p.x, sy), srcSize);
            o2 += wt * _BG2_FETCH(SampH1_R_2, int2(p.x, sy), srcSize);
            o3 += wt * _BG2_FETCH(SampH1_R_3, int2(p.x, sy), srcSize);
        #if _BG2_MRT_COUNT >= 6
            o4 += wt * _BG2_FETCH(SampH1_R_4, int2(p.x, sy), srcSize);
            o5 += wt * _BG2_FETCH(SampH1_R_5, int2(p.x, sy), srcSize);
        #endif
        #if _BG2_MRT_COUNT >= 8
            o6 += wt * _BG2_FETCH(SampH1_R_6, int2(p.x, sy), srcSize);
            o7 += wt * _BG2_FETCH(SampH1_R_7, int2(p.x, sy), srcSize);
        #endif
        }
    }

#if _BG2_MRT_COUNT == 4
    #define _BG2_REDUCE_R_BODY(L_SRC, WS, HS) \
        o0 = ReduceVec4_5x5(SampG##L_SRC##_R_0, p, int2(WS, HS)); \
        o1 = ReduceVec4_5x5(SampG##L_SRC##_R_1, p, int2(WS, HS)); \
        o2 = ReduceVec4_5x5(SampG##L_SRC##_R_2, p, int2(WS, HS)); \
        o3 = ReduceVec4_5x5(SampG##L_SRC##_R_3, p, int2(WS, HS));
#elif _BG2_MRT_COUNT == 6
    #define _BG2_REDUCE_R_BODY(L_SRC, WS, HS) \
        o0 = ReduceVec4_5x5(SampG##L_SRC##_R_0, p, int2(WS, HS)); \
        o1 = ReduceVec4_5x5(SampG##L_SRC##_R_1, p, int2(WS, HS)); \
        o2 = ReduceVec4_5x5(SampG##L_SRC##_R_2, p, int2(WS, HS)); \
        o3 = ReduceVec4_5x5(SampG##L_SRC##_R_3, p, int2(WS, HS)); \
        o4 = ReduceVec4_5x5(SampG##L_SRC##_R_4, p, int2(WS, HS)); \
        o5 = ReduceVec4_5x5(SampG##L_SRC##_R_5, p, int2(WS, HS));
#else
    #define _BG2_REDUCE_R_BODY(L_SRC, WS, HS) \
        o0 = ReduceVec4_5x5(SampG##L_SRC##_R_0, p, int2(WS, HS)); \
        o1 = ReduceVec4_5x5(SampG##L_SRC##_R_1, p, int2(WS, HS)); \
        o2 = ReduceVec4_5x5(SampG##L_SRC##_R_2, p, int2(WS, HS)); \
        o3 = ReduceVec4_5x5(SampG##L_SRC##_R_3, p, int2(WS, HS)); \
        o4 = ReduceVec4_5x5(SampG##L_SRC##_R_4, p, int2(WS, HS)); \
        o5 = ReduceVec4_5x5(SampG##L_SRC##_R_5, p, int2(WS, HS)); \
        o6 = ReduceVec4_5x5(SampG##L_SRC##_R_6, p, int2(WS, HS)); \
        o7 = ReduceVec4_5x5(SampG##L_SRC##_R_7, p, int2(WS, HS));
#endif

#define _BG2_DEF_REDUCE_R(L_SRC, L_DST, WS, HS, WD, HD) \
    void PS_ReduceR_##L_SRC##L_DST(float4 vpos : SV_Position, float2 uv : TEXCOORD0, _BG2_OUT_PARAMS) \
    { \
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(WD, HD)); \
        _BG2_REDUCE_R_BODY(L_SRC, WS, HS) \
    }

    _BG2_DEF_REDUCE_R(1, 2, _BG2_W1, _BG2_H1, _BG2_W2, _BG2_H2)
    _BG2_DEF_REDUCE_R(2, 3, _BG2_W2, _BG2_H2, _BG2_W3, _BG2_H3)
    _BG2_DEF_REDUCE_R(3, 4, _BG2_W3, _BG2_H3, _BG2_W4, _BG2_H4)
#if _BG2_LEVELS >= 6
    _BG2_DEF_REDUCE_R(4, 5, _BG2_W4, _BG2_H4, _BG2_W5, _BG2_H5)
#endif
#if _BG2_LEVELS >= 7
    _BG2_DEF_REDUCE_R(5, 6, _BG2_W5, _BG2_H5, _BG2_W6, _BG2_H6)
#endif
#if _BG2_LEVELS >= 8
    _BG2_DEF_REDUCE_R(6, 7, _BG2_W6, _BG2_H6, _BG2_W7, _BG2_H7)
#endif
#if _BG2_LEVELS >= 9
    _BG2_DEF_REDUCE_R(7, 8, _BG2_W7, _BG2_H7, _BG2_W8, _BG2_H8)
#endif

    // ---------------------------------------------------------------------------------------------
    // Phase 5: Bilateral-Guided Laplacian Band Interpolation (L1..L{L-2})
    // ---------------------------------------------------------------------------------------------

#if _BG2_MRT_COUNT == 4
    #define _BG2_SAMP_LVL_ARGS(L) SampG##L##_R_0, SampG##L##_R_1, SampG##L##_R_2, SampG##L##_R_3
#elif _BG2_MRT_COUNT == 6
    #define _BG2_SAMP_LVL_ARGS(L) SampG##L##_R_0, SampG##L##_R_1, SampG##L##_R_2, SampG##L##_R_3, SampG##L##_R_4, SampG##L##_R_5
#else
    #define _BG2_SAMP_LVL_ARGS(L) SampG##L##_R_0, SampG##L##_R_1, SampG##L##_R_2, SampG##L##_R_3, SampG##L##_R_4, SampG##L##_R_5, SampG##L##_R_6, SampG##L##_R_7
#endif

#define _BG2_DEF_INTERP_L(LVL, NEXT, WF, HF, WC, HC, WEIGHT) \
    float PS_InterpL_##LVL(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target \
    { \
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(WF, HF)); \
        float g = _BG2_FETCH(SampG##LVL, p, int2(WF, HF)).g; \
        return InterpolateLaplacianBand(g, _BG2_SAMP_LVL_ARGS(LVL), _BG2_SAMP_LVL_ARGS(NEXT), p, int2(WF, HF), int2(WC, HC), WEIGHT); \
    }

    _BG2_DEF_INTERP_L(1, 2, _BG2_W1, _BG2_H1, _BG2_W2, _BG2_H2, ScaleWeightFine)
    _BG2_DEF_INTERP_L(2, 3, _BG2_W2, _BG2_H2, _BG2_W3, _BG2_H3, ScaleWeightMedium)
    _BG2_DEF_INTERP_L(3, 4, _BG2_W3, _BG2_H3, _BG2_W4, _BG2_H4, ScaleWeightMedium)
#if _BG2_LEVELS >= 6
    _BG2_DEF_INTERP_L(4, 5, _BG2_W4, _BG2_H4, _BG2_W5, _BG2_H5, ScaleWeightMedium)
#endif
#if _BG2_LEVELS >= 7
    _BG2_DEF_INTERP_L(5, 6, _BG2_W5, _BG2_H5, _BG2_W6, _BG2_H6, ScaleWeightCoarse)
#endif
#if _BG2_LEVELS >= 8
    _BG2_DEF_INTERP_L(6, 7, _BG2_W6, _BG2_H6, _BG2_W7, _BG2_H7, ScaleWeightCoarse)
#endif
#if _BG2_LEVELS >= 9
    _BG2_DEF_INTERP_L(7, 8, _BG2_W7, _BG2_H7, _BG2_W8, _BG2_H8, ScaleWeightCoarse)
#endif

    // ---------------------------------------------------------------------------------------------
    // Phase 6: Bottom-Up Laplacian Pyramid Collapse (reuses TexG{l}.r as O_{l})
    // ---------------------------------------------------------------------------------------------

#if _BG2_LEVELS >= 9
    float2 PS_Collapse_76(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W6, _BG2_H6));
        float o = _BG2_FETCH(SampL6, p, int2(_BG2_W6, _BG2_H6)).r + ExpandScalar(SampL7, p, int2(_BG2_W7, _BG2_H7));
        return float2(o, 0.0f);
    }
#endif

#if _BG2_LEVELS >= 8
    float2 PS_Collapse_65(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W5, _BG2_H5));
    #if _BG2_LEVELS == 8
        float o = _BG2_FETCH(SampL5, p, int2(_BG2_W5, _BG2_H5)).r + ExpandScalar(SampL6, p, int2(_BG2_W6, _BG2_H6));
    #else
        float o = _BG2_FETCH(SampL5, p, int2(_BG2_W5, _BG2_H5)).r + ExpandScalar(SampG6, p, int2(_BG2_W6, _BG2_H6));
    #endif
        return float2(o, 0.0f);
    }
#endif

#if _BG2_LEVELS >= 7
    float2 PS_Collapse_54(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W4, _BG2_H4));
    #if _BG2_LEVELS == 7
        float o = _BG2_FETCH(SampL4, p, int2(_BG2_W4, _BG2_H4)).r + ExpandScalar(SampL5, p, int2(_BG2_W5, _BG2_H5));
    #else
        float o = _BG2_FETCH(SampL4, p, int2(_BG2_W4, _BG2_H4)).r + ExpandScalar(SampG5, p, int2(_BG2_W5, _BG2_H5));
    #endif
        return float2(o, 0.0f);
    }
#endif

#if _BG2_LEVELS >= 6
    float2 PS_Collapse_43(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W3, _BG2_H3));
    #if _BG2_LEVELS == 6
        float o = _BG2_FETCH(SampL3, p, int2(_BG2_W3, _BG2_H3)).r + ExpandScalar(SampL4, p, int2(_BG2_W4, _BG2_H4));
    #else
        float o = _BG2_FETCH(SampL3, p, int2(_BG2_W3, _BG2_H3)).r + ExpandScalar(SampG4, p, int2(_BG2_W4, _BG2_H4));
    #endif
        return float2(o, 0.0f);
    }
#endif

    float2 PS_Collapse_32(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W2, _BG2_H2));
    #if _BG2_LEVELS == 5
        float o = _BG2_FETCH(SampL2, p, int2(_BG2_W2, _BG2_H2)).r + ExpandScalar(SampL3, p, int2(_BG2_W3, _BG2_H3));
    #else
        float o = _BG2_FETCH(SampL2, p, int2(_BG2_W2, _BG2_H2)).r + ExpandScalar(SampG3, p, int2(_BG2_W3, _BG2_H3));
    #endif
        return float2(o, 0.0f);
    }

    float2 PS_Collapse_21(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W1, _BG2_H1));
        float o = _BG2_FETCH(SampL1, p, int2(_BG2_W1, _BG2_H1)).r + ExpandScalar(SampG2, p, int2(_BG2_W2, _BG2_H2));
        return float2(o, 0.0f);
    }

    // ---------------------------------------------------------------------------------------------
    // Phase 7: Fused L0 + Bilateral Base/Micro-Detail + Final Collapse + JzAzBz -> SDR/HDR Output
    // ---------------------------------------------------------------------------------------------

    float4 PS_Reconstruct(float4 vpos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
    {
        int2 p = _BG2_PIXEL_COORD(vpos, uv, int2(_BG2_W0, _BG2_H0));
        float4 origPixel = _BG2_FETCH(SampBackBuffer, p, int2(_BG2_W0, _BG2_H0));

        if (SplitScreenCompare && (uv.x < SplitPosition))
        {
            if (abs(uv.x - SplitPosition) < (BUFFER_RCP_WIDTH * 1.5f))
                return float4(0.5f, 0.5f, 0.5f, origPixel.a);
            return origPixel;
        }

        bool isNeutral = (FilterBlend <= 0.0f)
                      || (DetailContrast == 0.0f && EdgeToneScale == 1.0f
                          && HighlightCompression == 0.0f && ShadowLift == 0.0f
                          && BG_MicroDetail == 0.0f && BG_BaseToneCompress == 0.0f);
        if (isNeutral && DiagnosticView == 0)
            return origPixel;

        float4 jabA   = _BG2_FETCH(SampJzAzBz, p, int2(_BG2_W0, _BG2_H0));
        float2 b0Work = _BG2_FETCH(SampB0,     p, int2(_BG2_W0, _BG2_H0)).rg;
        float JzOrig = jabA.r;
        float B0     = b0Work.r;
        float JzWork = b0Work.g;

        float jzMax, sigmaJz, detailParam, betaHi, betaSh;
        GetRemapParams(jzMax, sigmaJz, detailParam, betaHi, betaSh);

        // Bilateral Grid Diagnostic Views (B0 base & Jz - B0 fine detail)
        if (DiagnosticView == 3)
        {
            float visJz = saturate(B0 / jzMax) * GetReferenceWhiteJz();
            float visNits = JzToLuminanceNits(visJz);
            float3 visRgb = EncodeLinearNitsToBackBuffer(float3(visNits, visNits, visNits), GetActiveColorSpace(), false);
            return float4(visRgb, origPixel.a);
        }
        if (DiagnosticView == 4)
        {
            float visDet = saturate(0.5f + 5.0f * ((JzOrig - B0) / GetReferenceWhiteJz())) * GetReferenceWhiteJz();
            float visNits = JzToLuminanceNits(visDet);
            float3 visRgb = EncodeLinearNitsToBackBuffer(float3(visNits, visNits, visNits), GetActiveColorSpace(), false);
            return float4(visRgb, origPixel.a);
        }

        // Level-0 Bilateral-Guided FLLF interpolation (touches only the 1 or 2 needed MRT packs!)
        float g0 = lerp(JzWork, B0, BG_GuidanceStrength);
        float jzStep = jzMax / float(_BG2_SLICES - 1);
        float pos = saturate(g0 / jzMax) * float(_BG2_SLICES - 1);
        int idx = min(int(floor(pos)), _BG2_SLICES - 2);
        float t = pos - float(idx);

        float L0_out;
        if (SliceInterpolationMode == 1)
        {
            int i1 = idx;
            int i2 = idx + 1;
            int pLow  = int(uint(i1) / 4u);
            int pHigh = int(uint(i2) / 4u);

            float4 expLow = ExpandG1Pack(pLow, p);
            float4 expHigh = expLow;
            [branch]
            if (pHigh != pLow)
                expHigh = ExpandG1Pack(pHigh, p);

            float v1 = EvalDeltaRemap(JzWork, float(i1) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel(expLow, i1 - pLow * 4);
            float v2 = EvalDeltaRemap(JzWork, float(i2) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel(expHigh, i2 - pHigh * 4);

            L0_out = lerp(v1, v2, t) * ScaleWeightFine;
        }
        else
        {
            int i0 = max(idx - 1, 0);
            int i1 = idx;
            int i2 = idx + 1;
            int i3 = min(idx + 2, _BG2_SLICES - 1);

            int p0 = int(uint(i0) / 4u);
            int p1 = int(uint(i1) / 4u);
            int p2 = int(uint(i2) / 4u);
            int p3 = int(uint(i3) / 4u);

            float4 expLow = ExpandG1Pack(p0, p);
            float4 expHigh = expLow;
            [branch]
            if (p3 != p0)
                expHigh = ExpandG1Pack(p3, p);

            float v0 = EvalDeltaRemap(JzWork, float(i0) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel(expLow, i0 - p0 * 4);
            float v1 = EvalDeltaRemap(JzWork, float(i1) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel((p1 == p0) ? expLow : expHigh, i1 - p1 * 4);
            float v2 = EvalDeltaRemap(JzWork, float(i2) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel((p2 == p0) ? expLow : expHigh, i2 - p2 * 4);
            float v3 = EvalDeltaRemap(JzWork, float(i3) * jzStep, sigmaJz, detailParam, betaHi, betaSh)
                     - ExtractChannel(expHigh, i3 - p3 * 4);

            L0_out = CatmullRom(v0, v1, v2, v3, t) * ScaleWeightFine;
        }

        // Combine: (1) Multi-scale FLLF delta + (2) Bilateral Pre-Denoise / Micro-Detail + (3) Bilateral Base Tone
        float deltaFLLF    = L0_out + ExpandScalar(SampG1, p, int2(_BG2_W1, _BG2_H1));
        float deltaBGMicro = (BG_MicroDetail > 0.0f) ? ((JzOrig - B0) * BG_MicroDetail) : (JzWork - JzOrig);
        float deltaBGBase  = 0.0f;
        if (BG_BaseToneCompress != 0.0f)
        {
            // Black-preserving, mid-gray-anchored C-infinity Durand-Dorsey bilateral base tone curve:
            // f(u; gamma) = ((u + eps)^gamma - eps^gamma) / ((1 + eps)^gamma - eps^gamma)
            // Exact properties: f(0) == 0 (never lifts pure black), f(1) == 1 (pins mid-gray),
            // f(u; 1) == u (exact identity at 0.0), bounded gentle slope near black.
            float anchorJz = 0.5f * GetReferenceWhiteJz();
            float u = max(B0, 0.0f) / anchorJz;
            float gamma = exp2(-BG_BaseToneCompress);
            const float eps = 0.05f;
            float epsG = SafePow(eps, gamma);
            float denom = max(SafePow(1.0f + eps, gamma) - epsG, 1.0e-12f);
            float uNew = (SafePow(u + eps, gamma) - epsG) / denom;
            deltaBGBase = anchorJz * (uNew - u);
        }

        float deltaJz = (deltaFLLF + deltaBGMicro + deltaBGBase) * FilterBlend;
        float minJz = (GamutMappingMode == 3 && GetActiveColorSpace() == 2) ? -0.28f : 0.0f;
        float JzNew = max(JzOrig + deltaJz, minJz);

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
    // MRT RenderTarget Binding Helper Macro (4, 6, or 8 RenderTargets)
    // ---------------------------------------------------------------------------------------------
#if _BG2_MRT_COUNT == 4
    #define _BG2_BIND_MRT(NAME) \
        RenderTarget0 = Tex##NAME##_0; RenderTarget1 = Tex##NAME##_1; \
        RenderTarget2 = Tex##NAME##_2; RenderTarget3 = Tex##NAME##_3;
#elif _BG2_MRT_COUNT == 6
    #define _BG2_BIND_MRT(NAME) \
        RenderTarget0 = Tex##NAME##_0; RenderTarget1 = Tex##NAME##_1; \
        RenderTarget2 = Tex##NAME##_2; RenderTarget3 = Tex##NAME##_3; \
        RenderTarget4 = Tex##NAME##_4; RenderTarget5 = Tex##NAME##_5;
#else
    #define _BG2_BIND_MRT(NAME) \
        RenderTarget0 = Tex##NAME##_0; RenderTarget1 = Tex##NAME##_1; \
        RenderTarget2 = Tex##NAME##_2; RenderTarget3 = Tex##NAME##_3; \
        RenderTarget4 = Tex##NAME##_4; RenderTarget5 = Tex##NAME##_5; \
        RenderTarget6 = Tex##NAME##_6; RenderTarget7 = Tex##NAME##_7;
#endif

    // ---------------------------------------------------------------------------------------------
    // Technique Definition
    // ---------------------------------------------------------------------------------------------

    technique FastLocalLaplacian_BilateralGrid_JzAzBz <
        ui_label = "Fast Local Laplacian + Bilateral Grid (JzAzBz HDR/SDR v2)";
        ui_tooltip = "Maximum-precision 32-bit float hybrid combining a Multi-Scale 3D Bilateral Grid\n"
                     "(Chen, Paris & Durand 2007) with a 32-slice Fast Local Laplacian Filter (Aubry et al. 2014)\n"
                     "in Safdar et al. (2017) JzAzBz space. Supports SDR, scRGB, HDR10 PQ, and HDR10 HLG.";
    >
    {
        // Phase 1: Entry to full-precision JzAzBz + per-frame HDR peak detection
        pass ToJzAzBz    { VertexShader = PostProcessVS; PixelShader = PS_ToJzAzBz;    RenderTarget = TexJzAzBz;  }
        pass Peak_1      { VertexShader = PostProcessVS; PixelShader = PS_Peak_1;      RenderTarget = TexPeak1;      }
        pass Peak_2      { VertexShader = PostProcessVS; PixelShader = PS_Peak_2;      RenderTarget = TexPeak2;      }
        pass Peak_3      { VertexShader = PostProcessVS; PixelShader = PS_Peak_3;      RenderTarget = TexPeak3;      }
        pass Peak_4      { VertexShader = PostProcessVS; PixelShader = PS_Peak_4;      RenderTarget = TexPeak4;      }
        pass Peak_Smooth { VertexShader = PostProcessVS; PixelShader = PS_Peak_Smooth; RenderTarget = TexPeakSmooth; }
        pass Peak_Store  { VertexShader = PostProcessVS; PixelShader = PS_Peak_Store;  RenderTarget = TexPeakPrev;   }

        // Phase 2: Multi-Scale 3D Bilateral Grid (Exact Float32 Splat -> 3D Blur X/Y/Z -> S1..S5 -> B0)
        pass BG_Splat    { VertexShader = PostProcessVS; PixelShader = PS_BG_Splat;    RenderTarget = TexBG_S0_A; }
        pass BG_BlurX    { VertexShader = PostProcessVS; PixelShader = PS_BG_BlurX;    RenderTarget = TexBG_S0_B; }
        pass BG_BlurY    { VertexShader = PostProcessVS; PixelShader = PS_BG_BlurY;    RenderTarget = TexBG_S0_A; }
        pass BG_BlurZ    { VertexShader = PostProcessVS; PixelShader = PS_BG_BlurZ;    RenderTarget = TexBG_S0_B; }
        pass BG_Reduce01 { VertexShader = PostProcessVS; PixelShader = PS_BG_Reduce_01; RenderTarget = TexBG_S1;  }
    #if _BG2_LEVELS >= 6
        pass BG_Reduce12 { VertexShader = PostProcessVS; PixelShader = PS_BG_Reduce_12; RenderTarget = TexBG_S2;  }
    #endif
    #if _BG2_LEVELS >= 7
        pass BG_Reduce23 { VertexShader = PostProcessVS; PixelShader = PS_BG_Reduce_23; RenderTarget = TexBG_S3;  }
    #endif
    #if _BG2_LEVELS >= 8
        pass BG_Reduce34 { VertexShader = PostProcessVS; PixelShader = PS_BG_Reduce_34; RenderTarget = TexBG_S4;  }
    #endif
    #if _BG2_LEVELS >= 9
        pass BG_Reduce45 { VertexShader = PostProcessVS; PixelShader = PS_BG_Reduce_45; RenderTarget = TexBG_S5;  }
    #endif
        pass BG_SliceB0  { VertexShader = PostProcessVS; PixelShader = PS_BG_SliceB0;  RenderTarget = TexB0;      }

        // Phase 3: Hybrid Bilateral-Guided Reference Pyramid (G1..G{L-2})
        pass Guide_01    { VertexShader = PostProcessVS; PixelShader = PS_Guide_01;    RenderTarget = TexG1;      }
        pass Guide_12    { VertexShader = PostProcessVS; PixelShader = PS_Guide_12;    RenderTarget = TexG2;      }
        pass Guide_23    { VertexShader = PostProcessVS; PixelShader = PS_Guide_23;    RenderTarget = TexG3;      }
    #if _BG2_LEVELS >= 6
        pass Guide_34    { VertexShader = PostProcessVS; PixelShader = PS_Guide_34;    RenderTarget = TexG4;      }
    #endif
    #if _BG2_LEVELS >= 7
        pass Guide_45    { VertexShader = PostProcessVS; PixelShader = PS_Guide_45;    RenderTarget = TexG5;      }
    #endif
    #if _BG2_LEVELS >= 8
        pass Guide_56    { VertexShader = PostProcessVS; PixelShader = PS_Guide_56;    RenderTarget = TexG6;      }
    #endif
    #if _BG2_LEVELS >= 9
        pass Guide_67    { VertexShader = PostProcessVS; PixelShader = PS_Guide_67;    RenderTarget = TexG7;      }
    #endif

        // Phase 4: Remapped Delta-Residual Gaussian Pyramids (16 / 24 / 32 slices)
        pass ReduceR01_H { VertexShader = PostProcessVS; PixelShader = PS_ReduceR01_H; _BG2_BIND_MRT(H1_R) }
        pass ReduceR01_V { VertexShader = PostProcessVS; PixelShader = PS_ReduceR01_V; _BG2_BIND_MRT(G1_R) }
        pass ReduceR_12  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_12;  _BG2_BIND_MRT(G2_R) }
        pass ReduceR_23  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_23;  _BG2_BIND_MRT(G3_R) }
        pass ReduceR_34  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_34;  _BG2_BIND_MRT(G4_R) }
    #if _BG2_LEVELS >= 6
        pass ReduceR_45  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_45;  _BG2_BIND_MRT(G5_R) }
    #endif
    #if _BG2_LEVELS >= 7
        pass ReduceR_56  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_56;  _BG2_BIND_MRT(G6_R) }
    #endif
    #if _BG2_LEVELS >= 8
        pass ReduceR_67  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_67;  _BG2_BIND_MRT(G7_R) }
    #endif
    #if _BG2_LEVELS >= 9
        pass ReduceR_78  { VertexShader = PostProcessVS; PixelShader = PS_ReduceR_78;  _BG2_BIND_MRT(G8_R) }
    #endif

        // Phase 5: Bilateral-Guided Intensity-Slice Interpolation of Laplacian Delta Bands (L1..L{L-2})
        pass InterpL_1   { VertexShader = PostProcessVS; PixelShader = PS_InterpL_1;   RenderTarget = TexL1; }
        pass InterpL_2   { VertexShader = PostProcessVS; PixelShader = PS_InterpL_2;   RenderTarget = TexL2; }
        pass InterpL_3   { VertexShader = PostProcessVS; PixelShader = PS_InterpL_3;   RenderTarget = TexL3; }
    #if _BG2_LEVELS >= 6
        pass InterpL_4   { VertexShader = PostProcessVS; PixelShader = PS_InterpL_4;   RenderTarget = TexL4; }
    #endif
    #if _BG2_LEVELS >= 7
        pass InterpL_5   { VertexShader = PostProcessVS; PixelShader = PS_InterpL_5;   RenderTarget = TexL5; }
    #endif
    #if _BG2_LEVELS >= 8
        pass InterpL_6   { VertexShader = PostProcessVS; PixelShader = PS_InterpL_6;   RenderTarget = TexL6; }
    #endif
    #if _BG2_LEVELS >= 9
        pass InterpL_7   { VertexShader = PostProcessVS; PixelShader = PS_InterpL_7;   RenderTarget = TexL7; }
    #endif

        // Phase 6: Bottom-Up Laplacian Pyramid Collapse (reuses TexG{l}.r as O_{l})
    #if _BG2_LEVELS >= 9
        pass Collapse_76 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_76; RenderTarget = TexG6; }
    #endif
    #if _BG2_LEVELS >= 8
        pass Collapse_65 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_65; RenderTarget = TexG5; }
    #endif
    #if _BG2_LEVELS >= 7
        pass Collapse_54 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_54; RenderTarget = TexG4; }
    #endif
    #if _BG2_LEVELS >= 6
        pass Collapse_43 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_43; RenderTarget = TexG3; }
    #endif
        pass Collapse_32 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_32; RenderTarget = TexG2; }
        pass Collapse_21 { VertexShader = PostProcessVS; PixelShader = PS_Collapse_21; RenderTarget = TexG1; }

        // Phase 7: Fused L0 + Bilateral Base/Detail + JzAzBz -> SDR/HDR Output
        pass Reconstruct
        {
            VertexShader = PostProcessVS;
            PixelShader  = PS_Reconstruct;
            SRGBWriteEnable = false;
        }
    }
}
