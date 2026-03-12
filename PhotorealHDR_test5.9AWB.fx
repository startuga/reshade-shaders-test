/**
 * ============================================================================
 * Photoreal HDR Color Grader (V5.9 - Mastering Edition)
 * Companion shader to Bilateral Contrast v8.4.5+
 *
 * Architecture:
 *   Pass 1 — Weighted Minkowski p-norm scene estimation (256×256 → mip chain)
 *            Area-average prefiltering via 2×2 bilinear box taps.
 *   Pass 2 — Temporal hysteresis (EMA smoothing into persistent 1×1 texture)
 *   Pass 3 — Full color grading pipeline with AWB readback
 *
 * V5.9 Features:
 * - Advanced Illuminant Estimator (Weighted Minkowski p=5 norm).
 *     - 2×2 bilinear area-average decimation (prevents shimmer from UI/foliage).
 *     - Saturation exclusion prevents colored objects from biasing the estimate.
 *     - Luminance gating excludes black-crush noise and specular peaks.
 *     - White-relative normalization prevents float16 overflow in the mip chain.
 *     - Per-tap independent weighting (one bad tap cannot contaminate others).
 * - Temporal Auto White Balance (Von Kries LMS adaptation with EMA hysteresis).
 * - Oklab Skin Tone Protection (Hemoglobin/Melanin hue isolation with early exit).
 * - Khronos PBR Neutral Tonemapper (parameterized peak for HDR, scRGB-safe).
 * - Filmic Contrast (signed-luminance safe for WCG/scRGB negatives).
 * - Intelligent Saturation (Oklab chroma with dark-pixel reliability ramp).
 * - Geometric-mean fallback for luminance normalization under extreme LMS ratios.
 * ============================================================================
 */

#include "ReShade.fxh"

// ============================================================================
// 1. Constants
// ============================================================================

// Preprocessor color space detection (ReShade built-in, 1 = sRGB)
#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

// IEEE 754 smallest normal float32 — used as a safe-division epsilon
static const float FLT_MIN = 1.175494351e-38;

// scRGB Reference White: 1.0 in scRGB = 80 cd/m² (nits)
static const float SCRGB_WHITE_NITS = 80.0;

// ---- Illuminant Estimation (Finlayson & Trezzi, 2004) ----
// p=5 is empirically optimal across Barnard, Gehler-Shi, and Ciurea datasets.
// Higher p weights brighter diffuse surfaces more heavily, which are more
// likely to reveal the illuminant and less likely to be deeply colored.
static const float AWB_P_NORM = 5.0;
static const float AWB_INV_P  = 1.0 / AWB_P_NORM;

// AWB Area-Average Decimation: quarter-texel offset for 2×2 box taps.
// Each AWB texel (256×256 target) spans 1/256 in UV. Taps at ±¼ of
// that span place 4 bilinear samples on non-overlapping quadrants.
// At 1080p: each offset ≈ 1.9 × 1.1 source pixels.
// At 4K:    each offset ≈ 3.8 × 2.1 source pixels.
// Combined with bilinear filtering (2×2 per tap), 4 taps cover
// roughly 20–60 source pixels per AWB texel (vs. 1 with POINT).
static const float AWB_QUARTER_TEXEL = 0.25 / 256.0;

// ---- sRGB Transfer Function Thresholds ----
static const float SRGB_THRESHOLD_EOTF = 0.04045;
static const float SRGB_THRESHOLD_OETF = 0.04045 / 12.92;

// ---- ST.2084 (PQ) EOTF Constants ----
static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;
static const float PQ_PEAK_LUMINANCE = 10000.0;

// ---- Chroma Reliability (aligned with Bilateral Contrast v8.4.5) ----
// Near-black pixels have numerically unstable chromaticity. The reliability
// ramp fades chroma modifications to identity below this threshold.
static const float CHROMA_STABILITY_THRESH = 1e-4;
static const float CHROMA_RELIABILITY_START = 5e-5;
static const float INV_CHROMA_RELIABILITY_SPAN =
    1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);

// ---- ITU-R Luma Coefficients ----
static const float3 Luma709  = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020 = float3(0.2627, 0.6780, 0.0593);

// ---- Color Space Matrices ----

// Rec.709 (sRGB) linear → LMS (Oklab-paired, Ottosson 2020)
static const float3x3 RGB709_to_LMS = float3x3(
    0.4122214708, 0.5363325363, 0.0514459929,
    0.2119034982, 0.6806995451, 0.1073969566,
    0.0883024619, 0.2817188376, 0.6299787005
);

// LMS → Rec.709 linear (inverse of above)
static const float3x3 LMS_to_RGB709 = float3x3(
     4.0767416621, -3.3077115913,  0.2309699292,
    -1.2684380046,  2.6097574011, -0.3413193965,
    -0.0041960863, -0.7034186147,  1.7076147010
);

// Rec.2020 linear → LMS (row-sum-normalized: D65 maps to equal LMS)
static const float3x3 RGB2020_to_LMS = float3x3(
    0.6167596970, 0.3601880240, 0.0230522790,
    0.2651316740, 0.6358515800, 0.0990167460,
    0.1001279150, 0.2038783840, 0.6959937010
);

// LMS → Rec.2020 linear (inverse of above)
static const float3x3 LMS_to_RGB2020 = float3x3(
     2.1398540771, -1.2462788877,  0.1064290765,
    -0.8846737634,  2.1631158093, -0.2784377818,
    -0.0486976682, -0.4543507342,  1.5030526721
);

// LMS' (cube-rooted) → Oklab L,a,b
static const float3x3 LMS_to_Oklab = float3x3(
    0.2104542553,  0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050,  0.4505937099,
    0.0259040371,  0.7827717662, -0.8086757660
);

// Oklab L,a,b → LMS' (for inverse cube to recover LMS)
static const float3x3 Oklab_to_LMSPrime = float3x3(
    1.0,  0.3963377774,  0.2158037573,
    1.0, -0.1055613458, -0.0638541728,
    1.0, -0.0894841775, -1.2914855480
);

// ============================================================================
// 2. Textures & Samplers
// ============================================================================

texture2D TextureBackBuffer : COLOR;

// Main pass: exact pixel fetch, no filtering
sampler2D SamplerBackBuffer
{
    Texture   = TextureBackBuffer;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = NONE;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

// AWB pass only: bilinear prefilter for area-average decimation.
// Same underlying texture, different filter state.
// Each bilinear tap blends a 2×2 source pixel neighborhood.
// Combined with the 2×2 tap pattern in PS_AWBDownsample, this
// covers ~16 source pixels per AWB texel (vs. 1 with POINT).
//
// Note: Bilinear interpolation occurs in the encoded domain (gamma/PQ),
// not linear. For AWB purposes (statistical scene average across 65K
// samples), the gamma-space interpolation bias is negligible. Decoding
// each corner texel individually would require 16 fetches per AWB texel.
sampler2D SamplerBackBufferLinear
{
    Texture   = TextureBackBuffer;
    MagFilter = LINEAR;
    MinFilter = LINEAR;
    MipFilter = NONE;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

// 256×256 scene estimation texture.
// Pass 1 writes white-relative p-norm weighted colors to mip 0.
// Hardware generates mips 1–8. Mip 8 (1×1) is the global weighted average.
texture2D TexSceneAvg
{
    Width     = 256;
    Height    = 256;
    Format    = RGBA16F;
    MipLevels = 9;
};
sampler2D SamplerSceneAvg
{
    Texture   = TexSceneAvg;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = LINEAR;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

// 1×1 persistent texture for temporal AWB state.
// RGBA32F prevents precision drift during long exponential moving average runs.
texture2D TexAWB_Temporal
{
    Width  = 1;
    Height = 1;
    Format = RGBA32F;
};
sampler2D SamplerAWB_Temporal
{
    Texture   = TexAWB_Temporal;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = NONE;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

// ============================================================================
// 3. UI Parameters
// ============================================================================

// ---- Tone & Exposure ----

uniform float fExposure <
    ui_type = "slider";
    ui_min = -3.00; ui_max = 3.00; ui_step = 0.01;
    ui_label = "Exposure (EV)";
    ui_category = "Tone & Exposure";
> = 0.00;

uniform float fBlackPoint <
    ui_type = "slider";
    ui_min = 0.000; ui_max = 0.050; ui_step = 0.001;
    ui_label = "Dehaze / Black Point";
    ui_tooltip =
        "Subtracts a percentage of reference white from the luminance range.\n"
        "Removes atmospheric haze and lifted blacks.\n"
        "Uses a C1-continuous parabolic toe to avoid hard contours.\n"
        "0.003 = 0.3%% of reference white.";
    ui_category = "Tone & Exposure";
> = 0.003;

uniform float fContrast <
    ui_type = "slider";
    ui_min = 0.80; ui_max = 1.50; ui_step = 0.001;
    ui_label = "Filmic Contrast";
    ui_tooltip =
        "Luminance-based power curve pivoted at 18%% grey.\n"
        "Preserves chromaticity via scalar ratio (no per-channel distortion).\n"
        "Handles scRGB negative luminance via absolute value.";
    ui_category = "Tone & Exposure";
> = 1.03;

// ---- Color Balance ----

uniform float fTemperature <
    ui_type = "slider";
    ui_min = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label = "Manual Temp (LMS)";
    ui_tooltip =
        "Negative = Cooler (removes warmth).\n"
        "Positive = Warmer.\n"
        "Uses exponential gain — always positive, no channel collapse.";
    ui_category = "Color Balance";
> = 0.00;

uniform float fTint <
    ui_type = "slider";
    ui_min = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label = "Manual Tint (LMS)";
    ui_tooltip =
        "Negative = Greener.\n"
        "Positive = More magenta.";
    ui_category = "Color Balance";
> = 0.00;

uniform float fSaturation <
    ui_type = "slider";
    ui_min = 0.00; ui_max = 2.00; ui_step = 0.01;
    ui_label = "Intelligent Saturation";
    ui_tooltip =
        "Oklab chroma adjustment with vibrance-style vivid protection.\n"
        "Above 1.0: protects already-saturated colors from over-boosting.\n"
        "Below 1.0: uniform chroma reduction.\n"
        "Near-black pixels fade to neutral (aligned with Bilateral Contrast).";
    ui_category = "Color Balance";
> = 1.08;

// ---- Dynamic Adaptation (AWB) ----

uniform bool bEnableDynamicAWB <
    ui_label = "Enable Dynamic Auto White Balance";
    ui_tooltip =
        "Weighted Minkowski p-norm illuminant estimator (p=5).\n"
        "Automatically neutralizes scene lighting via Von Kries adaptation.\n\n"
        "Area-average decimation: 2x2 bilinear box taps prevent\n"
        "thin UI elements and foliage from biasing the estimate.\n\n"
        "Luminance gating excludes shadows/noise and specular peaks.\n"
        "Saturation exclusion prevents colored objects from skewing.\n\n"
        "Limitations (Gray World family):\n"
        "Intentionally colored scenes (red tavern, underwater) will be\n"
        "partially neutralized. Reduce AWB Strength to preserve intent.";
    ui_category = "Dynamic Adaptation (AWB)";
> = true;

uniform float fAWBAdaptSpeed <
    ui_type = "slider";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.01;
    ui_label = "AWB Adaptation Speed";
    ui_tooltip =
        "Exponential moving average blend factor per frame.\n"
        "0.05 = Slow, cinematic (settles in ~1 sec at 60fps).\n"
        "0.20 = Fast, responsive (~0.25 sec).\n"
        "1.00 = Instant (may cause visible flicker).";
    ui_category = "Dynamic Adaptation (AWB)";
> = 0.05;

uniform float fAWBStrength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_label = "AWB Strength";
    ui_tooltip =
        "How much of the detected color cast to neutralize.\n"
        "0.0 = No correction (manual WB only).\n"
        "0.5 = Half correction (recommended for realism).\n"
        "1.0 = Full correction (strict Gray World neutral).";
    ui_category = "Dynamic Adaptation (AWB)";
> = 0.50;

uniform float fSkinToneProtection <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_label = "Skin Tone Protection (Oklab)";
    ui_tooltip =
        "Protects the Hemoglobin/Melanin hue range (~35-50 deg in Oklab).\n"
        "Reduces AWB correction strength for pixels detected as skin tones.\n"
        "Prevents human faces from looking gray, green, or lifeless.\n"
        "Only computes Oklab when > 0 (zero-cost early exit).";
    ui_category = "Dynamic Adaptation (AWB)";
> = 0.85;

// ---- Tone Mapping ----

uniform bool bEnableKhronosNeutral <
    ui_label = "Enable Khronos PBR Neutral Tonemapper";
    ui_tooltip =
        "Hue-preserving highlight compression with physical desaturation.\n"
        "Prevents hard clipping and hue rotation in extreme highlights.\n"
        "Parameterized peak for HDR displays.";
    ui_category = "Tone Mapping";
> = true;

uniform float fDisplayPeakNits <
    ui_type = "slider";
    ui_min = 80.0; ui_max = 4000.0; ui_step = 10.0;
    ui_label = "Display Peak Luminance (Nits)";
    ui_tooltip =
        "Maximum brightness your display can produce.\n"
        "SDR: ignored (locked to 1.0x reference white).\n"
        "HDR: sets the ceiling for highlight compression.";
    ui_category = "Tone Mapping";
> = 1000.0;

uniform float fCompressionStart <
    ui_type = "slider";
    ui_min = 0.50; ui_max = 0.95; ui_step = 0.01;
    ui_label = "Compression Start (%%)";
    ui_tooltip =
        "Percentage of display peak where highlight rolloff begins.\n"
        "0.80 = 1:1 color mapping up to 80%% of peak brightness.";
    ui_category = "Tone Mapping";
> = 0.80;

// ---- System ----

uniform int iColorSpaceOverride <
    ui_type = "combo";
    ui_label = "Color Space Override";
    ui_items = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0";
    ui_tooltip = "Must match Bilateral Contrast v8.4.5+.";
    ui_category = "System";
> = 0;

uniform float fWhitePoint <
    ui_type = "slider";
    ui_min = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_label = "Reference White (Nits)";
    ui_tooltip =
        "SDR is fixed at 80 nits (ignored).\n"
        "HDR: 203 = ITU-R BT.2408 reference diffuse white.\n"
        "Should match Bilateral Contrast Zone White Point.";
    ui_category = "System";
> = 203.0;

// ============================================================================
// 4. Math Utilities
// ============================================================================

// pow() that returns 0 for non-positive base (prevents NaN from negative
// base with fractional exponent, and preserves exact zero).
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

// Bit-level NaN/Inf detection (works on all GPU backends)
bool IsNanVal(float x) { return (asuint(x) & 0x7FFFFFFF) > 0x7F800000; }
bool IsInfVal(float x) { return (asuint(x) & 0x7FFFFFFF) == 0x7F800000; }
bool3 IsNan3(float3 v) { return bool3(IsNanVal(v.x), IsNanVal(v.y), IsNanVal(v.z)); }
bool3 IsInf3(float3 v) { return bool3(IsInfVal(v.x), IsInfVal(v.y), IsInfVal(v.z)); }

// ============================================================================
// 5. Transfer Functions (EOTF / OETF)
// ============================================================================

// sRGB Electro-Optical Transfer Function (display encoding → linear)
// Handles signed input for scRGB compatibility.
float3 sRGB_EOTF(float3 V)
{
    float3 abs_V  = abs(V);
    float3 lin_lo = abs_V / 12.92;
    float3 lin_hi = PowNonNegPreserveZero3((abs_V + 0.055) / 1.055, 2.4);

    float3 out_lin;
    out_lin.r = (abs_V.r <= SRGB_THRESHOLD_EOTF) ? lin_lo.r : lin_hi.r;
    out_lin.g = (abs_V.g <= SRGB_THRESHOLD_EOTF) ? lin_lo.g : lin_hi.g;
    out_lin.b = (abs_V.b <= SRGB_THRESHOLD_EOTF) ? lin_lo.b : lin_hi.b;

    return sign(V) * out_lin;
}

// sRGB Opto-Electronic Transfer Function (linear → display encoding)
float3 sRGB_OETF(float3 L)
{
    float3 abs_L  = abs(L);
    float3 enc_lo = abs_L * 12.92;
    float3 enc_hi = 1.055 * PowNonNegPreserveZero3(abs_L, 1.0 / 2.4) - 0.055;

    float3 out_enc;
    out_enc.r = (abs_L.r <= SRGB_THRESHOLD_OETF) ? enc_lo.r : enc_hi.r;
    out_enc.g = (abs_L.g <= SRGB_THRESHOLD_OETF) ? enc_lo.g : enc_hi.g;
    out_enc.b = (abs_L.b <= SRGB_THRESHOLD_OETF) ? enc_lo.b : enc_hi.b;

    return sign(L) * out_enc;
}

// ST.2084 (PQ) EOTF: PQ code values → absolute luminance (nits)
float3 PQ_EOTF(float3 N)
{
    N = saturate(N);
    float3 Np  = PowNonNegPreserveZero3(N, 1.0 / PQ_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    return PowNonNegPreserveZero3(num / den, 1.0 / PQ_M1) * PQ_PEAK_LUMINANCE;
}

// ST.2084 (PQ) Inverse EOTF: absolute luminance (nits) → PQ code values
float3 PQ_InverseEOTF(float3 L)
{
    L = clamp(L, 0.0, PQ_PEAK_LUMINANCE);
    float3 Lp  = PowNonNegPreserveZero3(L / PQ_PEAK_LUMINANCE, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    return saturate(PowNonNegPreserveZero3(num / den, PQ_M2));
}

// Unified decode: any supported space → linear nits
// SDR returns [0, 80] nits. scRGB returns unbounded signed nits.
// HDR10 returns [0, 10000] nits.
float3 DecodeToLinear(float3 encoded, int space)
{
    [branch] if (space == 3) return PQ_EOTF(encoded);
    [branch] if (space == 2) return encoded * SCRGB_WHITE_NITS;
    return sRGB_EOTF(encoded) * SCRGB_WHITE_NITS;
}

// Unified encode: linear nits → any supported space
float3 EncodeFromLinear(float3 lin, int space)
{
    [branch] if (space == 3) return PQ_InverseEOTF(lin);
    [branch] if (space == 2) return lin / SCRGB_WHITE_NITS;
    return sRGB_OETF(lin / SCRGB_WHITE_NITS);
}

// ============================================================================
// 6. AWB Pre-Passes
// ============================================================================

/**
 * Pass 1: Weighted Minkowski p-norm Scene Estimation
 *
 * Renders to TexSceneAvg (256×256 RGBA16F). Hardware mip generation
 * averages the result down to 1×1 at mip level 8.
 *
 * Area-average decimation:
 *   A 2×2 grid of bilinear-filtered taps replaces the single POINT sample.
 *   Each bilinear tap blends a 2×2 source neighborhood, and the 4 taps are
 *   offset to non-overlapping quadrants of the AWB texel footprint. This
 *   prevents thin bright elements (UI, crosshairs, foliage edges) from
 *   producing unrepresentative point samples that bias the global average.
 *
 * Per-tap independent weighting:
 *   Luminance gating and saturation exclusion are applied to each tap
 *   individually BEFORE accumulation. If one tap hits a specular highlight
 *   (weight → 0), it is zeroed out without contaminating the other three.
 *   Post-average weighting would blend the highlight into the color first,
 *   then try to suppress the already-contaminated result.
 *
 * White-relative normalization:
 *   DecodeToLinear returns values in nits (SDR white = 80, HDR up to 10000).
 *   pow(80, 5) = 3.3e9 overflows RGBA16F (max 65504). Dividing by whitePt
 *   first maps diffuse white to 1.0. After luminance gating (max ~2.5×),
 *   pow(2.5, 5) = 97.66 — safely within float16 range.
 *
 * The output is:
 *   RGB = sum_taps[ pow(normalizedColor, p) × weight ]
 *   A   = sum_taps[ weight ]
 */
void PS_AWBDownsample(
    float4 vpos     : SV_Position,
    float2 texcoord : TEXCOORD,
    out float4 avgColor : SV_Target)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;
    float whitePt = (space <= 1) ? SCRGB_WHITE_NITS : fWhitePoint;

    float3 accumulated = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;

    // 2×2 box tap pattern: offsets at ±¼ of the AWB texel in UV space.
    // The [unroll] hint guarantees the compiler unrolls to 4 inline samples,
    // eliminating loop overhead on all GPU architectures.
    [unroll]
    for (int y = -1; y <= 1; y += 2)
    {
        [unroll]
        for (int x = -1; x <= 1; x += 2)
        {
            float2 offset = float2(x, y) * AWB_QUARTER_TEXEL;
            float3 tap = DecodeToLinear(
                tex2D(SamplerBackBufferLinear, texcoord + offset).rgb,
                space
            );

            float luma = dot(tap, lumaCoeffs);

            // ---- Luminance Gating ----
            // Below 3% of white: chromaticity dominated by quantization noise.
            // Above 250% of white: specular/emissive carry object color, not
            // illuminant color. Smoothstep transitions avoid hard boundaries.
            float lumaLo = 0.03 * whitePt;
            float lumaHi = 2.50 * whitePt;
            float lumaMask = smoothstep(lumaLo * 0.5, lumaLo, luma)
                           * (1.0 - smoothstep(lumaHi, lumaHi * 1.5, luma));

            // ---- Saturation Exclusion ----
            // Highly saturated pixels are colored objects (red car, blue sky),
            // not neutral surfaces that reveal the illuminant. Suppression
            // begins at 60% HSV saturation, full exclusion at 90%.
            float maxC = max(tap.r, max(tap.g, tap.b));
            float minC = min(tap.r, min(tap.g, tap.b));
            float chroma_ratio = (maxC > 1e-6)
                ? ((maxC - minC) / maxC) : 0.0;
            float satMask = 1.0 - smoothstep(0.6, 0.9, chroma_ratio);

            float w = lumaMask * satMask;

            // ---- White-Relative Normalization + Minkowski p-norm ----
            // max(tap, 0) defends against scRGB negatives before fractional pow.
            float3 normalized = max(tap, 0.0) / max(whitePt, FLT_MIN);
            float3 powered = PowNonNegPreserveZero3(normalized, AWB_P_NORM);

            accumulated += powered * w;
            weightSum += w;
        }
    }

    // RGB = accumulated weighted p-norm; A = weight sum.
    // Hardware mip generation correctly averages both, preserving the ratio.
    avgColor = float4(accumulated, weightSum);
}

/**
 * Pass 2: Temporal Smoothing (Exponential Moving Average)
 *
 * Reads the 1×1 mip of the scene estimate (mip 8 of 256×256) and blends
 * it with the previous frame's stored value. This prevents the AWB from
 * flickering when the camera pans or scene lighting changes abruptly.
 *
 * The EMA operates on the raw weighted p-norm values (numerator and
 * denominator together). This is mathematically valid: each past frame
 * contributes with weight (1 - adaptSpeed)^age, producing a correctly
 * normalized time-weighted generalized mean at every frame.
 *
 * At fAWBAdaptSpeed = 0.05 and 60 fps:
 *   Time constant τ ≈ 20 frames ≈ 0.33 seconds
 *   95% settled in ~1.0 seconds
 *
 * On first frame (texture initialized to zero), the current value is
 * used directly to avoid lerping from black.
 */
void PS_AWBTemporal(
    float4 vpos     : SV_Position,
    float2 texcoord : TEXCOORD,
    out float4 result : SV_Target)
{
    float4 currentAvg  = tex2Dlod(SamplerSceneAvg, float4(0.5, 0.5, 0, 8));
    float4 previousAvg = tex2Dfetch(SamplerAWB_Temporal, int2(0, 0));

    // First frame detection: previousAvg.a (weight sum) is zero when
    // the texture has never been written. Skip blend to avoid lerping
    // from uninitialized data.
    result = (previousAvg.a < FLT_MIN)
        ? currentAvg
        : lerp(previousAvg, currentAvg, fAWBAdaptSpeed);
}

// ============================================================================
// 7. Color Processing Functions
// ============================================================================

/**
 * Black Point / Dehaze (C1-continuous parabolic toe)
 *
 * Subtracts bpNits from luminance via a scalar ratio applied to all channels.
 * Below luma = 2×bpNits, a parabolic toe smoothly transitions to zero,
 * providing C1 (value + derivative) continuity at the junction.
 *
 * Direct ratio form avoids luma² underflow for extremely small values.
 *
 * Continuity proof at junction (luma = 2×bpNits):
 *   Toe value:    (2bp) / (4bp) = 0.5
 *   Linear value: (2bp - bp) / (2bp) = 0.5  ✓
 *   Toe deriv:    1 / (4bp) = 1/(4bp)
 *   Linear deriv: d/dL[(L-bp)/L] = bp/L² = bp/(4bp²) = 1/(4bp)  ✓
 */
float ComputeBlackPointRatio(float luma, float bpNits)
{
    if (bpNits <= FLT_MIN || luma <= FLT_MIN)
        return 1.0;

    if (luma < 2.0 * bpNits)
        return luma / (4.0 * bpNits);

    return (luma - bpNits) / luma;
}

/**
 * LMS White Balance (Manual + Dynamic AWB + Skin Protection)
 *
 * Pipeline within this function:
 *   1. Manual temperature/tint → exponential LMS cone scaling.
 *   2. (Optional) Dynamic AWB:
 *      a. Read temporally-smoothed scene estimate from persistent texture.
 *      b. Invert the p-norm to recover illuminant chromaticity in nits.
 *      c. Compute Von Kries adaptation scales (clamped to ±3.5 stops).
 *   3. (Optional) Oklab skin tone protection:
 *      a. Convert current pixel to Oklab for hue analysis.
 *      b. Measure distance to skin hue anchor (0.75 rad ≈ 43°).
 *      c. Reduce AWB strength proportionally to skin mask.
 *   4. Luminance-preserving normalization for D65 neutrals.
 *      Falls back to geometric mean under extreme LMS ratios.
 */
float3 ApplyLMSWhiteBalance(
    float3 color,
    float temp, float tint,
    float3 lumaCoeffs,
    float3x3 to_LMS, float3x3 to_RGB,
    float whitePt)
{
    // ---- 1. Manual WB ----
    // 0.35 maps slider ±0.5 → ±0.175 stops. Gentle, camera-like response.
    // exp2() guarantees always-positive gains — no channel collapse.
    float3 wbStops = 0.35 * float3(
         temp + tint,   // L cone: warm + magenta
        -tint,          // M cone: green/magenta axis
        -temp + tint    // S cone: cool + magenta
    );
    float3 wbScale = exp2(wbStops);

    // ---- 2. Dynamic Auto White Balance ----
    [branch]
    if (bEnableDynamicAWB && fAWBStrength > 0.0)
    {
        // Read temporally-smoothed weighted p-norm from persistent 1×1 texture.
        // .rgb = sum of [pow(normalized_color, p) × weight] across frames
        // .a   = sum of weights across frames
        float4 rawAvg = tex2Dfetch(SamplerAWB_Temporal, int2(0, 0));
        float weightSum = max(rawAvg.a, FLT_MIN);

        // Invert the Minkowski norm: (mean of powers)^(1/p) × whitePt.
        // This recovers the estimated illuminant chromaticity in nits.
        // The whitePt multiplication reverses the normalization in Pass 1.
        float3 meanPowered = rawAvg.rgb / weightSum;
        float3 sceneAvg = PowNonNegPreserveZero3(meanPowered, AWB_INV_P) * whitePt;

        // Fallback: if the scene is entirely excluded by the gates (pitch
        // black, pure neon), default to neutral D65 → no AWB correction.
        if (weightSum < 1e-4)
            sceneAvg = float3(whitePt, whitePt, whitePt);

        // Von Kries target: equal-energy LMS at scene average luminance.
        // Row-sum-normalized matrices guarantee D65 maps to equal LMS,
        // so targetLMS = (avgLuma, avgLuma, avgLuma) without matrix multiply.
        float avgLuma = max(dot(sceneAvg, lumaCoeffs), FLT_MIN);
        float3 sceneLMS  = mul(to_LMS, sceneAvg);
        float3 targetLMS = float3(avgLuma, avgLuma, avgLuma);

        // Adaptation ratio with safety clamp (±3.5 stops ≈ [0.09, 11.0]).
        // Prevents mathematical explosion for extreme localized lighting
        // (e.g., single-color neon sign dominating after gate exclusion).
        float3 autoScale = clamp(
            targetLMS / max(sceneLMS, FLT_MIN),
            0.09, 11.0
        );

        // ---- 3. Skin Tone Protection (Oklab Hue Isolation) ----
        // Only pay for the Oklab round-trip when protection is active.
        float effectiveAWB = fAWBStrength;

        [branch]
        if (fSkinToneProtection > 0.0)
        {
            // Convert current pixel to Oklab for hue analysis
            float3 lms_pixel = mul(to_LMS, color);
            float3 lms_p = sign(lms_pixel)
                         * pow(max(abs(lms_pixel), FLT_MIN), 1.0 / 3.0);
            float3 lab = mul(LMS_to_Oklab, lms_p);

            // Skin hue anchor: 0.75 rad ≈ 43°.
            // This covers the hemoglobin/melanin spectral reflectance peak
            // across diverse skin tones (Fitzpatrick types I–VI).
            float hue = atan2(lab.z, lab.y);
            float skinDist = abs(hue - 0.75);

            // Smooth falloff: full protection at exact skin hue,
            // drops to zero at 0.35 rad (~20°) distance.
            float skinMask = 1.0 - smoothstep(0.0, 0.35, skinDist);

            // Ignore desaturated pixels: grays and deep shadows can
            // accidentally match any hue angle. Require minimum chroma.
            float chroma = sqrt(lab.y * lab.y + lab.z * lab.z);
            skinMask *= smoothstep(0.005, 0.02, chroma);

            // Reduce AWB strength where skin is detected
            effectiveAWB *= lerp(1.0, 1.0 - fSkinToneProtection, skinMask);
        }

        // Blend auto-correction into the WB scale
        wbScale *= lerp(float3(1.0, 1.0, 1.0), autoScale, effectiveAWB);
    }

    // ---- 4. Luminance-Preserving Normalization ----
    // Ensures D65 neutral tones maintain exact luminance after WB.
    // Converts the LMS scale to RGB and normalizes by the resulting luminance.
    float3 d65_wb_rgb = mul(to_RGB, wbScale);
    float lumaScale = dot(d65_wb_rgb, lumaCoeffs);

    // Safety: Extreme LMS ratios (e.g., single-channel scale > 5×) can
    // produce negative intermediate RGB channels due to the large negative
    // off-diagonal entries in LMS → RGB matrices. This makes the dot-product
    // luminance near zero or negative, causing normalization to explode.
    //
    // Fallback: geometric mean of LMS scales. This is always well-defined
    // for positive scales and provides approximate luminance preservation.
    // It only triggers under extreme auto-correction that is already
    // visually suspect (beyond ±3.5 stops in a single channel).
    if (lumaScale < 0.1)
        lumaScale = pow(wbScale.x * wbScale.y * wbScale.z, 1.0 / 3.0);

    wbScale /= max(lumaScale, FLT_MIN);

    // ---- Apply Von Kries Adaptation ----
    float3 lms = mul(to_LMS, color);
    lms *= wbScale;
    return mul(to_RGB, lms);
}

/**
 * Intelligent Saturation (Oklab Chroma with Vivid Protection)
 *
 * Protection metric uses RGB channel spread (fast, numerically stable).
 * Actual chroma scaling uses Oklab (perceptually uniform, hue-preserving).
 *
 * Three-tier early exit skips the Oklab round-trip when unnecessary:
 *   1. Slider at neutral (1.0) → no work needed.
 *   2. Pixel is near-black → chroma is numerically unreliable.
 *   3. Effective gain collapsed to 1.0 after protection + reliability.
 *
 * `space` parameter is needed (unlike WB) for gamut-specific tuning:
 *   Rec.2020 gets gentler boost (0.90× comp) and smaller residual (0.20)
 *   to avoid pushing colors toward the wider gamut boundary.
 */
float3 ApplyIntelligentSaturation(
    float3 color,
    float saturation,
    int space,
    float3 lumaCoeffs,
    float3x3 to_LMS, float3x3 to_RGB)
{
    // Exit 1: Slider at neutral
    if (abs(saturation - 1.0) < 1e-6)
        return color;

    float luma = dot(color, lumaCoeffs);

    // Dark-chroma reliability ramp (aligned with Bilateral Contrast).
    // Negative luma → reliability = 0 → bypass (correct for scRGB negatives).
    float ct = saturate(
        (luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN
    );
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);

    // Exit 2: Near-black pixel — chroma is numerically unstable
    if (chroma_reliability <= 0.0)
        return color;

    // Vivid-color protection from RGB channel spread.
    // abs(peak) handles scRGB negatives; signed max/min measure true spread.
    float peak  = max(abs(color.r), max(abs(color.g), abs(color.b)));
    float max_c = max(color.r, max(color.g, color.b));
    float min_c = min(color.r, min(color.g, color.b));

    float sat_current = 0.0;
    if (peak > 1e-6)
        sat_current = saturate((max_c - min_c) / peak);

    // Smoothstep makes protection onset gradual
    float protection = sat_current * sat_current * (3.0 - 2.0 * sat_current);

    float chroma_gain = saturation;
    if (saturation > 1.0)
    {
        float boost = saturation - 1.0;
        float space_comp      = (space >= 3) ? 0.90 : 1.0;
        float min_boost_share = (space >= 3) ? 0.20 : 0.25;
        chroma_gain = 1.0 + boost * space_comp
                    * lerp(1.0, min_boost_share, protection);
    }

    // Fade unreliable dark pixels toward identity
    chroma_gain = lerp(1.0, chroma_gain, chroma_reliability);

    // Exit 3: Effective gain collapsed to neutral
    if (abs(chroma_gain - 1.0) < 1e-6)
        return color;

    // ---- Oklab Round-Trip ----
    float3 lms   = mul(to_LMS, color);
    float3 lms_p = sign(lms) * pow(max(abs(lms), FLT_MIN), 1.0 / 3.0);
    float3 lab   = mul(LMS_to_Oklab, lms_p);

    lab.yz *= chroma_gain;

    float3 lms_p_out = mul(Oklab_to_LMSPrime, lab);
    float3 lms_out   = lms_p_out * lms_p_out * lms_p_out;

    return mul(to_RGB, lms_out);
}

/**
 * Khronos PBR Neutral Tonemapper (Generalized for HDR)
 *
 * Expects input normalized so 1.0 = diffuse white.
 * targetPeak = 1.0 for SDR, displayPeak/refWhite for HDR.
 *
 * Pipeline:
 *   1. Fresnel toe: subtle shadow shaping (parabolic for min channel < 0.08)
 *   2. Rational compression: smooth highlight rolloff asymptoting to targetPeak
 *   3. Physical desaturation: bright highlights trend toward neutral white
 *
 * ARCHITECTURAL DEVIATION FROM REFERENCE:
 *   Below the compression threshold, the reference applies the toe offset
 *   (subtract then re-add — algebraically identity for non-negative input).
 *   This implementation returns the original unmodified color to preserve
 *   negative scRGB wide-gamut values that max(color, 0) would crush.
 *   The toe only produces visible results through the compression rescaling.
 */
float3 ApplyKhronosPBRNeutral(float3 color, float targetPeak, float compressionStart)
{
    float3 safeColor = max(color, 0.0);

    // Fresnel toe offset
    float x = min(safeColor.r, min(safeColor.g, safeColor.b));
    float offset = (x < 0.08) ? (x - 6.25 * x * x) : 0.04;
    safeColor -= offset;

    float peak      = max(safeColor.r, max(safeColor.g, safeColor.b));
    float startComp = (targetPeak * compressionStart) - 0.04;

    [branch]
    if (peak >= startComp && startComp > 0.0)
    {
        // Rational compression curve (asymptote at targetPeak)
        float d       = targetPeak - startComp;
        float newPeak = targetPeak - (d * d) / (peak + d - startComp);
        safeColor *= newPeak / max(peak, FLT_MIN);

        // Physical desaturation toward neutral as brightness increases
        float desatRate = 0.15 / targetPeak;
        float g = 1.0 - 1.0 / (desatRate * (peak - newPeak) + 1.0);
        safeColor = lerp(safeColor, newPeak.xxx, g);

        return safeColor + offset;
    }

    // Below compression: return original to preserve scRGB negatives
    return color;
}

// ============================================================================
// 8. Main Pixel Shader (Pass 3)
// ============================================================================

void PS_PhotorealHDR(
    float4 vpos     : SV_Position,
    float2 texcoord : TEXCOORD,
    out float4 fragColor : SV_Target)
{
    int2   pos = int2(vpos.xy);
    float4 src = tex2Dfetch(SamplerBackBuffer, pos);

    int    space     = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;
    float  whitePt    = (space <= 1) ? SCRGB_WHITE_NITS : fWhitePoint;

    // ---- Fast Bypass ----
    // All controls at neutral and features disabled → bit-exact passthrough.
    // Preserves upstream alpha without decode/encode round-trip.
    [branch]
    if (fExposure == 0.0 && fBlackPoint == 0.0 && fContrast == 1.0 &&
        fTemperature == 0.0 && fTint == 0.0 && fSaturation == 1.0 &&
        !bEnableKhronosNeutral && !bEnableDynamicAWB)
    {
        fragColor = src;
        return;
    }

    // Decode and cache original for NaN/Inf fallback
    float3 original_lin = DecodeToLinear(src.rgb, space);
    float3 color = original_lin;

    // ---- Matrix Setup (hoisted once, shared by WB and Saturation) ----
    float3x3 to_LMS, to_RGB;

    [branch]
    if (space >= 3) {
        to_LMS = RGB2020_to_LMS;
        to_RGB = LMS_to_RGB2020;
    } else {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
    }

    // ---- 1. Exposure (linear EV shift) ----
    if (fExposure != 0.0)
        color *= exp2(fExposure);

    // ---- 2. White Balance (Manual + Dynamic AWB + Skin Protection) ----
    if (fTemperature != 0.0 || fTint != 0.0 || bEnableDynamicAWB)
        color = ApplyLMSWhiteBalance(
            color, fTemperature, fTint,
            lumaCoeffs, to_LMS, to_RGB, whitePt
        );

    // ---- 3. Dehaze / Black Point ----
    if (fBlackPoint > 0.0)
    {
        float luma = dot(color, lumaCoeffs);
        if (luma > FLT_MIN)
        {
            float bpNits  = fBlackPoint * whitePt;
            float bpRatio = ComputeBlackPointRatio(luma, bpNits);
            color *= bpRatio;
        }
    }

    // ---- 4. Filmic Contrast (signed-luminance safe) ----
    if (abs(fContrast - 1.0) > 1e-6)
    {
        float luma    = dot(color, lumaCoeffs);
        float absLuma = abs(luma);
        if (absLuma > FLT_MIN)
        {
            float pivot         = 0.18 * whitePt;
            float contrastLuma  = PowNonNegPreserveZero(absLuma / pivot, fContrast) * pivot;
            float contrastRatio = min(contrastLuma / absLuma, 100.0);
            color *= contrastRatio;
        }
    }

    // ---- 5. Intelligent Saturation ----
    color = ApplyIntelligentSaturation(
        color, fSaturation, space, lumaCoeffs, to_LMS, to_RGB
    );

    // ---- 6. Khronos PBR Neutral Tone Mapping ----
    [branch]
    if (bEnableKhronosNeutral)
    {
        // Normalize so 1.0 = diffuse white (Khronos convention)
        color /= max(whitePt, FLT_MIN);

        // Peak is 1.0 for SDR. For HDR, it's the ratio of display capability
        // to reference white. max(1.0, ...) prevents compressing below
        // diffuse white even if the user sets Display Peak < Ref White.
        float targetPeak = (space <= 1)
            ? 1.0
            : max(1.0, fDisplayPeakNits / whitePt);

        color = ApplyKhronosPBRNeutral(color, targetPeak, fCompressionStart);

        // Restore to working nits
        color *= whitePt;
    }

    // ---- Safety: NaN / Inf Recovery ----
    if (any(IsNan3(color)) || any(IsInf3(color)))
        color = original_lin;

    // ---- Encode to Display Space ----
    float3 encoded = EncodeFromLinear(color, space);

    // SDR hard clamp: required for display correctness.
    // When Khronos is enabled, the Fresnel toe offset re-addition can push
    // peak values up to ~1.04 normalized, producing a small expected clip.
    // Without Khronos, this is the only out-of-range protection.
    [flatten]
    if (space <= 1)
        encoded = saturate(encoded);

    fragColor = float4(encoded, src.a);
}

// ============================================================================
// 9. Technique
// ============================================================================

technique PhotorealHDR_Mastering <
    ui_label = "Photoreal HDR V5.9 (Mastering Edition)";
    ui_tooltip =
        "Photorealistic color grading for SDR and HDR.\n\n"
        "Pipeline:\n"
        "  1. Exposure (linear EV shift)\n"
        "  2. White Balance (Manual LMS + Dynamic AWB + Skin Protection)\n"
        "  3. Dehaze / Black Point (C1 parabolic toe)\n"
        "  4. Filmic Contrast (signed-luminance safe)\n"
        "  5. Intelligent Saturation (Oklab chroma)\n"
        "  6. Khronos PBR Neutral (hue-preserving highlight compression)\n\n"
        "AWB Architecture:\n"
        "  Pass 1: Weighted Minkowski p=5 estimation\n"
        "          (2x2 bilinear area-average, luminance/saturation gates)\n"
        "  Pass 2: Temporal EMA smoothing (persistent 1x1 state)\n"
        "  Pass 3: Von Kries adaptation + full grading pipeline\n\n"
        "Companion: Bilateral Contrast v8.4.5+";
>
{
    // Pass 1: Write weighted p-norm colors → 256×256 with area-average
    // decimation (2×2 bilinear box taps). Hardware generates mip chain.
    // Mip 8 (1×1) becomes the global weighted scene average.
    pass CalculateAWB
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_AWBDownsample;
        RenderTarget = TexSceneAvg;
        GenerateMips = true;
    }

    // Pass 2: Blend current frame's 1×1 average with previous temporal state.
    // Prevents AWB flicker during camera movement or lighting transitions.
    pass TemporalSmooth
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_AWBTemporal;
        RenderTarget = TexAWB_Temporal;
    }

    // Pass 3: Read temporally-smoothed AWB estimate, run full color grading
    // pipeline, and output to screen.
    pass MainRender
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_PhotorealHDR;
    }
}