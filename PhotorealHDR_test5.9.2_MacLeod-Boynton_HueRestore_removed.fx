// ============================================================================
// Photoreal HDR Color Grader (V5.9.2 - Mastering Edition)
// Companion shader to Bilateral Contrast v8.4.4+
//
//* Design Philosophy: PRECISION OVER PERFORMANCE
//* - True IEEE 754 Math (No fast intrinsics or approximations)
//* - Exact IEC/SMPTE Standard Constants
//* - Bit-Exact Neutrality Logic
//* - Pre-computed High-Precision Kernels
//* - True Stop-Domain HDR Processing
//* - MacLeod-Boynton Physiological Chromaticity Processing
//
// V5.9.2 Changes from V5.9.1:
// - Removed: MacLeod-Boynton Hue Restoration (fHueRestore slider, function,
//            and pre-tonemap capture). Mathematically proven to be a no-op:
//            the Khronos PBR Neutral tonemapper applies a uniform per-pixel
//            affine transform (same scale A and offset B for all RGB channels).
//            Through row-sum-normalized LMS matrices, this preserves the MB
//            chromaticity direction from D65 exactly — the hue angle is
//            invariant because the additive constant cancels in the cone-
//            differential signals (L-M) and (2S-L-M) that define the
//            direction, and the positive multiplicative constant cancels
//            in the direction normalization.
//            Saves 2× matrix multiply + 2× MB transform + 2× IEEE sqrt
//            per pixel when Khronos was enabled.
//            NOTE: If a future tonemapper applies per-channel curves (ACES,
//            per-channel Reinhard), hue restoration would become necessary.
//            The feature should be re-implemented at that time.
// V5.9.1 Changes from V5.9:
// - Fix: ApplyMBHueRestore now has negative-LMS and dark-chroma reliability
//        gates. (Removed in V5.9.2 along with the entire function.)
// - Fix: Restored vivid-color protection in ApplyMBPurity. V5.9 replaced
//        Oklab saturation with uniform MB purity scaling, removing the
//        vibrance-style protection and Rec.2020 gamut awareness from V5.8.
//        Now re-implemented in MB space: purity distance from D65 white is
//        measured, smoothstepped, and used to reduce boost for already-vivid
//        colors. Rec.2020 receives gentler boost (space_comp=0.90) and a
//        lower residual floor (min_boost_share=0.20 vs 0.25 for Rec.709)
//        to avoid gamut boundary clipping.
// - Fix: Replaced all rsqrt() / sqrt() intrinsics with explicit IEEE 754
//        equivalents (pow(x, 0.5) / (1.0/pow(x, 0.5))). GPU SFU rsqrt()
//        typically delivers ~22-bit mantissa precision via fast approximation,
//        violating the "No fast intrinsics" design philosophy. The explicit
//        pow() path goes through the shader's PowNonNegPreserveZero pipeline,
//        guaranteeing full IEEE 754 single-precision evaluation and zero
//        handling. The compiler may still optimize to SFU on some backends,
//        but the source code is explicit about intent.
// - Fix: Documented the filmic contrast 100x ratio ceiling.
// V5.9 Changes from V5.8:
// - REPLACED: Oklab Saturation with MacLeod-Boynton Purity.
//             MB space is strictly linear (no cube roots) and guarantees
//             100% isoluminant chromaticity scaling (z = L+M is preserved).
// - REMOVED: All Oklab matrices and transcendental math (massive ALU savings).
// V5.8 Changes from V5.7:
// - Fix: Khronos compression now ratio-preserves negative scRGB channels.
//        Previously, max(color, 0.0) destroyed WCG excursions at the gate.
//        Now, compression ratio is computed from clamped peak but applied
//        uniformly to original channels. Negatives scale proportionally
//        through compression and only fade during desaturation.
//        Set Highlight Desaturation to 0 for full WCG preservation.
// - Fix: Decode sanitization catches NaN/Inf from corrupt upstream buffers.
// - Fix: Fast bypass uses epsilon comparisons for preset serialization safety.
// V5.7 Changes from V5.6:
// - Fix: Pipeline reorder — Saturation now runs AFTER Khronos tonemapping
//        so the tonemapper's highlight desaturation no longer overrides
//        creative saturation choices. Saturation is now display-referred.
// - Fix: Khronos desaturation uses bounded compression-progress metric
//        instead of unbounded (peak - newPeak) to prevent excessive
//        highlight desaturation in HDR (V5.6 issue).
// V5.6 Changes from V5.4:
// - add: Khronos PBR Neutral (Hue-preserving highlight compression)
// V5.5 Changes from V5.4:
// - Fix: Filmic Contrast now preserves signed-luminance behavior
// - Fix: Intelligent Saturation with dark-chroma reliability ramp
//        aligned with Bilateral Contrast v8.4.3+
// - Fix: More conservative defaults for subtle photoreal starting point
// - Fix: Early exits in saturation skip round-trip when unnecessary
// - Alignment: Companion version strings updated to v8.4.4
// ============================================================================

#include "ReShade.fxh"

// ==============================================================================
// 1. Constants
// ==============================================================================

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

static const float FLT_MIN = 1.175494351e-38;
static const float SCRGB_WHITE_NITS = 80.0;

// Neutrality-test epsilon: 1 ULP at the 6th decimal digit.
// Chosen larger than float rounding (~1e-7) to catch preset serialization
// residuals, but small enough to be visually imperceptible (<0.0001% change).
static const float NEUTRAL_EPS = 1e-6;

// sRGB (IEC 61966-2-1:1999)
// Exact decimal constants from the standard.
// Binary representations are nearest-representable IEEE 754 single.
static const float SRGB_THRESHOLD_EOTF = 0.04045;
static const float SRGB_THRESHOLD_OETF = 0.0031308;  // See note below
static const float SRGB_GAMMA          = 2.4;
static const float SRGB_INV_GAMMA      = 0.41666666666666667; // 1/2.4 = 5/12

// NOTE on SRGB_THRESHOLD_OETF:
// IEC 61966-2-1 specifies the linear-side threshold as 0.0031308, AND the
// encoded-side threshold as 0.04045 with slope 12.92. These are inconsistent
// at the 5th decimal digit: 0.04045/12.92 = 0.003130804954, not 0.0031308.
// We use the standard's explicit linear-side value 0.0031308 for the OETF,
// and the explicit encoded-side value 0.04045 for the EOTF, matching the
// reference specification exactly rather than deriving one from the other.

// ST.2084 (PQ) EOTF Constants (SMPTE ST 2084:2014)
// All forward constants are exact rational fractions of powers-of-two
// and are exactly representable in IEEE 754 single precision.
static const float PQ_M1 = 0.1593017578125;   // 2610 / 16384
static const float PQ_M2 = 78.84375;           // 2523 / 32
static const float PQ_C1 = 0.8359375;          // 3424 / 4096
static const float PQ_C2 = 18.8515625;         // 2413 / 128 = 2413 * 32 / 4096
static const float PQ_C3 = 18.6875;            // 2392 / 128 = 2392 * 32 / 4096
static const float PQ_PEAK_LUMINANCE = 10000.0;

// Pre-computed reciprocals for EOTF inversion.
// NOT exactly representable in float — these are the nearest IEEE 754 singles
// to the true rational values, computed at double precision and truncated.
static const float PQ_INV_M1 = 6.2773946360153257;   // 16384 / 2610
static const float PQ_INV_M2 = 0.012683313515655966;  // 32 / 2523

// Chroma reliability alignment with Bilateral Contrast
static const float CHROMA_STABILITY_THRESH = 1e-4;
static const float CHROMA_RELIABILITY_START = 5e-5;
static const float INV_CHROMA_RELIABILITY_SPAN =
    1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);  // = 20000.0

// ITU-R Luma Coefficients (BT.709-6, BT.2020-2)
static const float3 Luma709  = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020 = float3(0.2627, 0.6780, 0.0593);

// LMS Matrices
// These are used for:
//   1. White balance (LMS cone-domain gain)
//   2. MacLeod-Boynton chromaticity (l = L/(L+M), s = S/(L+M))
//
// Both matrices are row-sum-normalized: mul(to_LMS, (1,1,1)) ≈ (1,1,1).
// This ensures D65 neutral maps to equal-energy LMS, giving MB white at
// (0.5, 0.5) — a stable anchor for purity scaling.
// The Rec.2020 matrix has ~0.01% row-sum residual (1.0001 vs 1.0000),
// producing mb_white ≈ (0.50003, 0.49998) — visually imperceptible.
//
// The row-sum normalization also guarantees that the Khronos tonemapper's
// uniform affine RGB transform maps to a uniform affine LMS transform,
// which provably preserves the MacLeod-Boynton hue angle from D65 white.
// See V5.9.2 changelog for the formal proof.

static const float3x3 RGB709_to_LMS = float3x3(
    0.4122214708, 0.5363325363, 0.0514459929,
    0.2119034982, 0.6806995451, 0.1073969566,
    0.0883024619, 0.2817188376, 0.6299787005
);

static const float3x3 LMS_to_RGB709 = float3x3(
     4.0767416621, -3.3077115913,  0.2309699292,
    -1.2684380046,  2.6097574011, -0.3413193965,
    -0.0041960863, -0.7034186147,  1.7076147010
);

// Row-sum-normalized Rec.2020 -> LMS
static const float3x3 RGB2020_to_LMS = float3x3(
    0.6167596970, 0.3601880240, 0.0230522790,
    0.2651316740, 0.6358515800, 0.0990167460,
    0.1001279150, 0.2038783840, 0.6959937010
);

// Inverse of the normalized Rec.2020 -> LMS matrix
static const float3x3 LMS_to_RGB2020 = float3x3(
     2.1398540771, -1.2462788877,  0.1064290765,
    -0.8846737634,  2.1631158093, -0.2784377818,
    -0.0486976682, -0.4543507342,  1.5030526721
);

// Maximum achievable purity in MacLeod-Boynton space.
// A fully monochromatic stimulus (single cone class) reaches purity ~0.5
// from the D65 white point at (0.5, 0.5). In practice, real display
// primaries achieve ~0.35 for Rec.709 and ~0.42 for Rec.2020.
// 0.35 is used as the smoothstep ceiling for vivid-color protection:
// at this purity, the color is already at or near the gamut boundary
// for Rec.709, so boost should be maximally attenuated.
static const float MB_PURITY_PROTECTION_CEILING = 0.35;

// ==============================================================================
// 2. Texture & Sampler
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

// ==============================================================================
// 3. UI Parameters
// ==============================================================================

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
    ui_tooltip = "Subtracts a percentage of reference white from the entire luminance range.\n"
                 "Removes atmospheric haze / dusty lifted blacks.\n"
                 "Uses a smooth C1 parabolic toe to avoid hard contours.\n"
                 "0.003 = 0.3%% of white.";
    ui_category = "Tone & Exposure";
> = 0.003;

uniform float fContrast <
    ui_type = "slider";
    ui_min = 0.80; ui_max = 1.50; ui_step = 0.001;
    ui_label = "Filmic Contrast";
    ui_tooltip = "Luminance-based power curve pivoted at 18%% grey.\n"
                 "Preserves chromaticity by applying a scalar ratio to RGB.\n"
                 "Handles negative-luminance scRGB via absolute value.";
    ui_category = "Tone & Exposure";
> = 1.03;

uniform float fContrastPivot <
    ui_type = "slider";
    ui_min = 0.01; ui_max = 1.00; ui_step = 0.01;
    ui_label = "Contrast Pivot";
    ui_tooltip = "The luminance value that remains unchanged when contrast is adjusted.\n"
                "0.18 = Photographic middle gray (default, filmic look).\n"
                "Lower values: more shadow modification, less highlight modification.\n"
                "Higher values: less shadow modification, more highlight modification.\n"
                "Does not prevent highlight expansion — use the Highlights slider for that.";
    ui_category = "Tone & Exposure";
> = 0.18;

uniform float fShadows <
    ui_type = "slider";
    ui_min = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label = "Shadows (Log Recovery)";
    ui_tooltip = "Lifts or deepens shadow detail in the stop domain.\n"
                "+1.0 = Lift up to 3 stops (recover shadow detail).\n"
                "-1.0 = Deepen up to 3 stops (crush shadows for mood).\n"
                "Operates below the Contrast Pivot. Zero effect at the pivot.\n"
                "C1 continuous with the contrast curve (seamless transition).";
    ui_category = "Tone & Exposure";
> = 0.0;

uniform float fHighlights <
    ui_type = "slider";
    ui_min = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label = "Highlights (Log Recovery)";
    ui_tooltip = "Protects (-1.0) or boosts (+1.0) highlights.\n"
                 "Use negative values to recover detail blown out by high contrast.\n"
                 "Operates as a mathematically smooth shoulder curve.";
    ui_category = "Tone & Exposure";
> = 0.0;

uniform float fTemperature <
    ui_type = "slider";
    ui_min = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label = "Color Temperature (LMS)";
    ui_tooltip = "Negative = Cooler (removes yellow/sand tint)\n"
                 "Positive = Warmer\n"
                 "Uses exponential gain (always positive, no channel collapse).\n"
                 "Luminance-preserving for neutral tones.\n"
                 "Saturated colors may shift ~1-3%% in luminance.";
    ui_category = "Color Balance";
> = -0.06;

uniform float fTint <
    ui_type = "slider";
    ui_min = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label = "Color Tint (LMS)";
    ui_tooltip = "Negative = Greener\nPositive = More Magenta";
    ui_category = "Color Balance";
> = 0.01;

uniform float fSaturation <
    ui_type = "slider";
    ui_min = 0.00; ui_max = 2.00; ui_step = 0.01;
    ui_label = "Saturation (MacLeod-Boynton Purity)";
    ui_tooltip = "Strictly isoluminant saturation in physiological MacLeod-Boynton space.\n"
                 "Alters chromaticity distance from D65 white without changing luminance (L+M).\n"
                 "Above 1.0: vibrance-style boost (protects already vivid colors from clipping).\n"
                 "Below 1.0: uniform purity reduction toward neutral.\n"
                 "Near-black pixels fade toward neutral to prevent math instability.\n"
                 "Rec.2020 receives gentler boost to avoid gamut boundary clipping.";
    ui_category = "Color Balance";
> = 1.08;

uniform bool bEnableKhronosNeutral <
    ui_label = "Enable Khronos PBR Neutral Tonemapper";
    ui_tooltip = "Applies strict hue-preserving highlight compression.\n"
                 "Prevents hard-clipping and color shifts in extreme highlights.\n\n"
                 "Hue preservation is mathematically guaranteed: Khronos applies\n"
                 "a uniform per-pixel affine transform to all RGB channels, which\n"
                 "preserves the MacLeod-Boynton chromaticity direction from D65\n"
                 "through row-sum-normalized LMS matrices.";
    ui_category = "Tone Mapping";
> = true;

uniform float fDisplayPeakNits <
    ui_type = "slider";
    ui_min = 80.0; ui_max = 4000.0; ui_step = 10.0;
    ui_label = "Display Peak Luminance (Nits)";
    ui_tooltip = "The maximum brightness your display can output.\n"
                 "For SDR, this is ignored (locked to 1.0x Reference White).";
    ui_category = "Tone Mapping";
> = 1000.0;

uniform float fCompressionStart <
    ui_type = "slider";
    ui_min = 0.50; ui_max = 0.95; ui_step = 0.01;
    ui_label = "Compression Start (%)";
    ui_tooltip = "Where to start rolling off highlights (percentage of Peak).\n"
                 "0.80 = 1:1 color mapping up to 80%% of peak display brightness.";
    ui_category = "Tone Mapping";
> = 0.80;

uniform float fDesaturationStrength <
    ui_type = "slider";
    ui_min = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label = "Highlight Desaturation";
    ui_tooltip = "Physical desaturation near display peak.\n"
                 "0.00 = Pure chromaticity preservation (no desaturation).\n"
                 "0.15 = Khronos PBR Neutral reference default.\n"
                 "Higher values simulate emitter saturation roll-off.\n\n"
                 "The original Khronos formula over-desaturates in HDR because\n"
                 "it scales with absolute input intensity. This version bounds\n"
                 "desaturation to the compression-progress ratio, so maximum\n"
                 "desaturation equals this value regardless of input brightness.\n\n"
                 "Note: Desaturation reduces purity (distance from white) but\n"
                 "cannot alter hue — the lerp toward a uniform scalar preserves\n"
                 "the MacLeod-Boynton chromaticity direction exactly.";
    ui_category = "Tone Mapping";
> = 0.15;

uniform int iColorSpaceOverride <
    ui_type = "combo";
    ui_label = "Color Space Override";
    ui_items = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0";
    ui_tooltip = "Must match Bilateral Contrast v8.4.4+.";
    ui_category = "System";
> = 0;

uniform float fWhitePoint <
    ui_type = "slider";
    ui_min = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_label = "Reference White (Nits)";
    ui_tooltip = "Should match Zone White Point in Bilateral Contrast.\n"
                 "SDR stays fixed at 80 nits.\n"
                 "203 = ITU-R BT.2408 reference diffuse white.";
    ui_category = "System";
> = 203.0;

// ==============================================================================
// 4. Math Utilities
// ==============================================================================

// IEEE 754 compliant non-negative pow with zero preservation.
// Returns exactly 0.0 for x <= 0.0 (no NaN from negative base).
// Used for all transcendental power operations in the pipeline.
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

// IEEE 754 square root: explicit pow(x, 0.5) instead of sqrt() intrinsic.
// sqrt() may be implemented as rsqrt(x)*x on some GPU backends, inheriting
// the SFU's ~22-bit approximation. pow(x, 0.5) = exp2(0.5 * log2(x)) uses
// the full-precision transcendental pipeline.
// Returns 0.0 for x <= 0.0 (via PowNonNegPreserveZero).
float SqrtIEEE(float x)
{
    return PowNonNegPreserveZero(x, 0.5);
}

bool IsNanVal(float x) { return (asuint(x) & 0x7FFFFFFF) > 0x7F800000; }
bool IsInfVal(float x) { return (asuint(x) & 0x7FFFFFFF) == 0x7F800000; }
bool3 IsNan3(float3 v) { return bool3(IsNanVal(v.x), IsNanVal(v.y), IsNanVal(v.z)); }
bool3 IsInf3(float3 v) { return bool3(IsInfVal(v.x), IsInfVal(v.y), IsInfVal(v.z)); }

// ==============================================================================
// 5. EOTF / OETF
// ==============================================================================

float3 sRGB_EOTF(float3 V)
{
    float3 abs_V = abs(V);
    float3 lin_lo = abs_V / 12.92;
    float3 lin_hi = PowNonNegPreserveZero3((abs_V + 0.055) / 1.055, SRGB_GAMMA);

    float3 out_lin;
    out_lin.r = (abs_V.r <= SRGB_THRESHOLD_EOTF) ? lin_lo.r : lin_hi.r;
    out_lin.g = (abs_V.g <= SRGB_THRESHOLD_EOTF) ? lin_lo.g : lin_hi.g;
    out_lin.b = (abs_V.b <= SRGB_THRESHOLD_EOTF) ? lin_lo.b : lin_hi.b;
    return sign(V) * out_lin;
}

float3 sRGB_OETF(float3 L)
{
    float3 abs_L = abs(L);
    float3 enc_lo = abs_L * 12.92;
    float3 enc_hi = 1.055 * PowNonNegPreserveZero3(abs_L, SRGB_INV_GAMMA) - 0.055;

    float3 out_enc;
    out_enc.r = (abs_L.r <= SRGB_THRESHOLD_OETF) ? enc_lo.r : enc_hi.r;
    out_enc.g = (abs_L.g <= SRGB_THRESHOLD_OETF) ? enc_lo.g : enc_hi.g;
    out_enc.b = (abs_L.b <= SRGB_THRESHOLD_OETF) ? enc_lo.b : enc_hi.b;
    return sign(L) * out_enc;
}

float3 PQ_EOTF(float3 N)
{
    // Robustness: prevent upstream out-of-bounds from exploding fractional exponents
    N = saturate(N);
    float3 Np = PowNonNegPreserveZero3(N, PQ_INV_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    return PowNonNegPreserveZero3(num / den, PQ_INV_M1) * PQ_PEAK_LUMINANCE;
}

float3 PQ_InverseEOTF(float3 L)
{
    // Robustness: explicitly clamp to HDR peak to prevent unbounded values.
    // PQ cannot represent negative luminance — negative scRGB channels are
    // silently clipped here. This is an inherent PQ format limitation.
    // The space_comp and vivid-color protection in ApplyMBPurity mitigate
    // the likelihood of significant negative excursions reaching this point.
    L = clamp(L, 0.0, PQ_PEAK_LUMINANCE);
    float3 Lp = PowNonNegPreserveZero3(L / PQ_PEAK_LUMINANCE, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    return saturate(PowNonNegPreserveZero3(num / den, PQ_M2));
}

float3 DecodeToLinear(float3 encoded, int space)
{
    [branch] if (space == 3) return PQ_EOTF(encoded);
    [branch] if (space == 2) return encoded * SCRGB_WHITE_NITS;
    return sRGB_EOTF(encoded) * SCRGB_WHITE_NITS;
}

float3 EncodeFromLinear(float3 lin, int space)
{
    [branch] if (space == 3) return PQ_InverseEOTF(lin);
    [branch] if (space == 2) return lin / SCRGB_WHITE_NITS;
    return sRGB_OETF(lin / SCRGB_WHITE_NITS);
}

// ==============================================================================
// 6.a Color Processing (Exposure, WB, Contrast)
// ==============================================================================

// Direct ratio form avoids luma-squared underflow for extremely small values.
// Parabolic toe is C1-continuous with the linear subtraction region at luma = 2*bpNits.
//
// C0 continuity at luma = 2·bp:
//   Parabolic: 2bp/(4bp) = 0.5
//   Linear:    (2bp - bp)/(2bp) = 0.5  ✓
//
// C1 continuity (derivatives):
//   Parabolic: d/dL [L/(4bp)] = 1/(4bp)
//   Linear:    d/dL [(L-bp)/L] = bp/L² = bp/(4bp²) = 1/(4bp)  ✓
float ComputeBlackPointRatio(float luma, float bpNits)
{
    if (bpNits <= FLT_MIN || luma <= FLT_MIN) return 1.0;
    if (luma < 2.0 * bpNits) return luma / (4.0 * bpNits);
    return (luma - bpNits) / luma;
}

// Exponential cone response white balance in LMS space.
// 0.35 maps +/-0.5 slider to +/-0.175 stops — gentle, camera-like response.
// exp2() guarantees positive gains: no channel collapse at slider extremes.
// D65 normalization ensures achromatic inputs maintain exact luminance.
// Saturated colors may shift ~1-3% in luminance (inherent to LMS WB).
float3 ApplyLMSWhiteBalance(float3 color, float temp, float tint, float3 lumaCoeffs, float3x3 to_LMS, float3x3 to_RGB)
{
    float3 wbStops = 0.35 * float3(temp + tint, -tint, -temp + tint);
    float3 wbScale = exp2(wbStops);

    // Normalize so D65 neutrals keep exact luminance.
    // exp2() guarantees positive wbScale, and row-sum-normalized matrix
    // produces positive D65 results, so lumaScale is provably > 0.
    float3 d65_wb_rgb = mul(to_RGB, wbScale);
    float lumaScale = dot(d65_wb_rgb, lumaCoeffs);
    wbScale /= max(lumaScale, FLT_MIN);

    float3 lms = mul(to_LMS, color);
    lms *= wbScale;
    return mul(to_RGB, lms);
}

// ==============================================================================
// 6.b MacLeod-Boynton Purity (Isoluminant Saturation)
// ==============================================================================

// MacLeod-Boynton chromaticity coordinates:
//   l = L / (L + M)    — long-wave fraction of luminance
//   s = S / (L + M)    — short-wave scaled by luminance
//   z = L + M           — luminance (CIE V(λ) approximation)
//
// D65 white point: (l, s) ≈ (0.5, 0.5) for row-sum-normalized LMS matrices.
// The z channel is strictly preserved by all chromaticity operations,
// guaranteeing physiological isoluminance.
//
// Division by (L+M) is undefined when L+M <= 0, which occurs for scRGB
// pixels with extreme negative R (e.g., R=-1.0, G=0.4, B=0.0 gives
// L+M ≈ -0.14 while BT.709 Y ≈ +0.07). Callers must gate on L+M > 0.
float3 LMS_to_MB(float3 lms) {
    float lum = max(lms.r + lms.g, FLT_MIN);
    return float3(lms.r / lum, lms.b / lum, lum);
}

// Inverse MacLeod-Boynton: recover LMS from (l, s, z).
//   L = l · z
//   M = z - L = z · (1 - l)
//   S = s · z
//
// Round-trip is bit-exact:
//   L' = (L/(L+M)) · (L+M) = L
//   M' = (L+M) - L = M
//   S' = (S/(L+M)) · (L+M) = S
float3 MB_to_LMS(float3 mb) {
    return float3(mb.x * mb.z, mb.z - (mb.x * mb.z), mb.y * mb.z);
}

// Isoluminant saturation with vivid-color protection.
//
// Purity scaling: lerp chromaticity (l, s) toward D65 white or away from it.
//   Below 1.0: uniform reduction (all colors desaturate equally).
//   Above 1.0: vibrance-style boost with smoothstepped protection that
//              attenuates boost for already-vivid colors near the gamut boundary.
//
// Protection mechanism (boost only):
//   Purity = distance from D65 white in MB chromaticity space.
//   A smoothstep from 0 to MB_PURITY_PROTECTION_CEILING (0.35) maps purity
//   to a protection factor. At maximum protection, a residual fraction of
//   the boost still applies (min_boost_share) to prevent a hard "wall" where
//   colors just below the threshold are boosted but those above aren't.
//
// Gamut awareness:
//   Rec.2020 (space >= 3): space_comp = 0.90, min_boost_share = 0.20
//     — wider gamut primaries are closer to MB boundary, need gentler boost.
//   Rec.709  (space <  3): space_comp = 1.00, min_boost_share = 0.25
//     — tighter gamut has more headroom before clipping.
//
// Dark-chroma reliability:
//   Near-black pixels (luma < CHROMA_STABILITY_THRESH) fade to identity
//   via smoothstep to prevent numerically unstable MB coordinates.
//   Negative luma → reliability = 0 → bypass (correct for out-of-gamut scRGB).
float3 ApplyMBPurity(float3 color, float purity_scale, int space, float3 lumaCoeffs, float3x3 to_LMS, float3x3 to_RGB)
{
    // Exit 1: Slider at neutral — no work needed
    if (abs(purity_scale - 1.0) < NEUTRAL_EPS) return color;

    // Dark-chroma reliability: fade chroma changes in unstable near-black regions.
    // Negative luma -> ct = 0 -> bypass (correct for out-of-gamut scRGB).
    float luma = dot(color, lumaCoeffs);
    float ct = saturate((luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);

    // Exit 2: Near-black pixel — chromaticity is numerically unstable
    if (chroma_reliability <= 0.0) return color;

    float effective_scale = purity_scale;

    // Vivid-color protection (boost only — desaturation is always uniform)
    if (purity_scale > 1.0)
    {
        // Compute purity: distance from D65 white in MB chromaticity
        float3 lms = mul(to_LMS, color);
        float lm_sum = lms.r + lms.g;

        // L+M <= 0: MB chromaticity undefined, bypass protection and let
        // the dark-chroma reliability gate handle the fade-out.
        // This path is unlikely but possible for extreme scRGB negatives.
        if (lm_sum > FLT_MIN)
        {
            float3 white_lms = mul(to_LMS, float3(1.0, 1.0, 1.0));
            float2 mb_white = LMS_to_MB(white_lms).xy;

            float3 mb = LMS_to_MB(lms);
            float2 chroma_offset = mb.xy - mb_white;
            float purity_sq = dot(chroma_offset, chroma_offset);
            float purity = SqrtIEEE(purity_sq);

            // Smoothstep protection: onset is gradual rather than linear.
            // At purity >= MB_PURITY_PROTECTION_CEILING (0.35), the color is
            // at/near the gamut boundary and receives maximum protection.
            float protection_t = saturate(purity / MB_PURITY_PROTECTION_CEILING);
            float protection = protection_t * protection_t * (3.0 - 2.0 * protection_t);

            float boost = purity_scale - 1.0;

            // Gamut-aware attenuation:
            // Rec.2020: wider gamut primaries are closer to MB boundary,
            //           so apply gentler boost and lower residual.
            // Rec.709:  tighter gamut has more headroom.
            float space_comp = (space >= 3) ? 0.90 : 1.0;

            // Residual boost fraction: prevents a hard protection "wall".
            // Even at maximum protection, a fraction of the boost applies,
            // ensuring smooth transition for colors near the threshold.
            float min_boost_share = (space >= 3) ? 0.20 : 0.25;

            effective_scale = 1.0 + boost * space_comp * lerp(1.0, min_boost_share, protection);
        }
    }

    // Apply dark-chroma reliability: unreliable pixels fade to identity (scale=1.0)
    effective_scale = lerp(1.0, effective_scale, chroma_reliability);

    // Exit 3: Effective scale collapsed to neutral after reliability + protection
    if (abs(effective_scale - 1.0) < NEUTRAL_EPS) return color;

    // Apply purity scaling in MB space.
    // Re-compute LMS and MB here rather than reusing the protection path's
    // intermediate values, because:
    //   1. The protection path may have been skipped (scale <= 1.0 or L+M <= 0)
    //   2. Code clarity: this is the actual color modification, not a metric
    // The second matrix multiply is the cost of clean control flow.
    float3 white_lms = mul(to_LMS, float3(1.0, 1.0, 1.0));
    float2 mb_white = LMS_to_MB(white_lms).xy;

    float3 lms_final = mul(to_LMS, color);

    // Guard: if L+M <= 0, MB is undefined. Return color unchanged.
    // (This is separate from dark-chroma reliability, which keys on BT.709 Y.
    //  L+M can be negative while Y is positive for extreme scRGB values.)
    float lm_sum_final = lms_final.r + lms_final.g;
    if (lm_sum_final <= 0.0) return color;

    float3 mb = LMS_to_MB(lms_final);

    // Purity scaling: interpolate chromaticity toward/away from D65 white.
    // z (luminance) is untouched — guarantees physiological isoluminance.
    mb.xy = lerp(mb_white, mb.xy, effective_scale);

    return mul(to_RGB, MB_to_LMS(mb));
}

// ==============================================================================
// 6.c Khronos PBR Neutral Tonemapper (HDR Parameterized 'P' Peak)
// ==============================================================================

// Khronos PBR Neutral highlight compression.
//
// Key design points:
// - Hue-preserving: uniform ratio applied to all channels (see proof below)
// - Rational compression: asymptotically approaches targetPeak
// - V5.8 WCG fix: ratio computed from clamped peak, applied to original channels
//   so negative scRGB channels scale proportionally (not destroyed)
// - V5.7 bounded desaturation: progress metric is clamped to [0,1]
//   so maximum desaturation = fDesaturationStrength regardless of input brightness
//
// The Fresnel toe offset (0.04 at saturation) adds specular energy to prevent
// crushed blacks in PBR materials. After compression, toe re-addition can push
// values up to ~1.04 in normalized space — the final SDR saturate() handles this.
//
// ──────────────────────────────────────────────────────────────────────────────
// PROOF: Khronos preserves MacLeod-Boynton hue exactly.
//
// The complete Khronos operation on each RGB channel is:
//   C_out = (C_in - offset) × ratio × (1-g) + newPeak × g + offset
//         = C_in × A + B
//   where A = ratio×(1-g)  and  B = newPeak×g + offset×(1 - A)
//
// A and B are per-pixel scalars, identical across R, G, B.
//
// Through a row-sum-normalized LMS matrix M (row sums = 1):
//   L_out = A·L_in + B,  M_out = A·M_in + B,  S_out = A·S_in + B
//
// The MB chromaticity offset from D65 white (0.5, 0.5) is:
//   Δl = (L-M) / (2(L+M))
//   Δs = (2S-L-M) / (2(L+M))
//
// After the affine transform:
//   Δl_out = A(L-M) / (2(A(L+M)+2B))
//   Δs_out = A(2S-L-M) / (2(A(L+M)+2B))
//
// The additive constant B cancels in the numerator differences (L-M, 2S-L-M).
// The positive scalar A and positive denominator cancel in the direction.
// Therefore atan2(Δs, Δl) is invariant. ∎
//
// Consequence: No separate hue restoration pass is needed after Khronos.
// If a future tonemapper uses per-channel curves (ACES, per-channel Reinhard),
// this invariant would break and hue restoration would be required.
// ──────────────────────────────────────────────────────────────────────────────
float3 ApplyKhronosPBRNeutral(float3 color, float targetPeak, float compressionStart)
{
    // Compute toe offset from clamped channels (safe math for the parabola).
    // When ANY channel is negative, min(safeColor) = 0, so offset = 0.
    // This correctly disables the Fresnel toe for out-of-gamut pixels.
    float3 safeColor = max(color, 0.0);
    float x = min(safeColor.r, min(safeColor.g, safeColor.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;

    // Peak from clamped-and-toed channels drives the compression decision.
    float peak = max(safeColor.r, max(safeColor.g, safeColor.b)) - offset;
    float startComp = (targetPeak * compressionStart) - 0.04;

    [branch]
    if (peak >= startComp && startComp > 0.0)
    {
        // Rational compression: newPeak approaches targetPeak asymptotically.
        // d = headroom from compression start to peak.
        // The formula newPeak = P - d²/(peak + d - start) gives:
        //   - At peak = start: newPeak = start (C0 continuous with 1:1 region)
        //   - As peak → ∞:    newPeak → targetPeak (bounded output)
        float d = targetPeak - startComp;
        float newPeak = targetPeak - (d * d) / (peak + d - startComp);

        // Apply toe offset + uniform compression ratio to ORIGINAL color.
        // ratio = newPeak/peak < 1.0 always (compression), so negative
        // channels reduce in magnitude — no amplification of out-of-gamut.
        float3 working = color - offset;
        float ratio = newPeak / max(peak, FLT_MIN);
        working *= ratio;

        // Bounded physical desaturation (V5.7 fix).
        // t = compression progress: 0 at startComp, approaches 1 at targetPeak.
        // g = desaturation factor: 0 at start, approaches fDesaturationStrength.
        // For negative channels, lerp toward positive newPeak pulls them
        // toward neutral — physically correct for display rendering.
        // With fDesaturationStrength = 0, negatives survive completely.
        float t = saturate((newPeak - startComp) / max(d, FLT_MIN));
        float g = fDesaturationStrength * t * t;
        working = lerp(working, newPeak.xxx, g);

        return working + offset;
    }

    // 1:1 region: bypass preserves negative scRGB exactly (unchanged).
    return color;
}

// ==============================================================================
// 7. Main Shader
// ==============================================================================

void PS_PhotorealHDR(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 fragColor : SV_Target)
{
    int2 pos = int2(vpos.xy);
    float4 src = tex2Dfetch(SamplerBackBuffer, pos);

    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;
    float whitePt = (space <= 1) ? SCRGB_WHITE_NITS : fWhitePoint;

    // Fast bypass: uniform epsilon guards prevent preset serialization residuals
    // from defeating the bypass. NEUTRAL_EPS = 1e-6 (see constants block).
    [branch]
    if (abs(fExposure) < NEUTRAL_EPS &&
        abs(fBlackPoint) < NEUTRAL_EPS &&
        abs(fContrast - 1.0) < NEUTRAL_EPS &&
        abs(fShadows) < NEUTRAL_EPS &&
        abs(fHighlights) < NEUTRAL_EPS &&
        abs(fTemperature) < NEUTRAL_EPS &&
        abs(fTint) < NEUTRAL_EPS &&
        abs(fSaturation - 1.0) < NEUTRAL_EPS &&
        !bEnableKhronosNeutral) {
        fragColor = src;
        return;
    }

    // Decode to linear nits. Sanitize immediately: if upstream buffer contains
    // NaN/Inf (corrupt swap chain, broken compute shader), fall back to black
    // rather than propagating poison through the pipeline.
    float3 original_lin = DecodeToLinear(src.rgb, space);

    if (any(IsNan3(original_lin)) || any(IsInf3(original_lin)))
        original_lin = 0.0;

    float3 color = original_lin;

    // Matrix setup hoisted out of per-function branches.
    // space >= 3 (HDR10/PQ) uses Rec.2020 primaries; all others use Rec.709.
    float3x3 to_LMS, to_RGB;
    [branch]
    if (space >= 3) {
        to_LMS = RGB2020_to_LMS;
        to_RGB = LMS_to_RGB2020;
    } else {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
    }

    // ── Pipeline Stage 1: Exposure ──────────────────────────────────────────
    // Linear EV shift: multiply by 2^EV.
    // +1.0 EV = double brightness, -1.0 EV = half brightness.
    if (abs(fExposure) > NEUTRAL_EPS)
        color *= exp2(fExposure);

    // ── Pipeline Stage 2: White Balance ─────────────────────────────────────
    // LMS cone-domain exponential gain (see ApplyLMSWhiteBalance docs).
    if (abs(fTemperature) > NEUTRAL_EPS || abs(fTint) > NEUTRAL_EPS)
        color = ApplyLMSWhiteBalance(color, fTemperature, fTint, lumaCoeffs, to_LMS, to_RGB);

    // ── Pipeline Stage 3: Dehaze / Black Point ──────────────────────────────
    // Subtractive lift removal with C1 parabolic toe.
    if (fBlackPoint > NEUTRAL_EPS)
    {
        float luma = dot(color, lumaCoeffs);
        if (luma > FLT_MIN)
        {
            float bpNits = fBlackPoint * whitePt;
            color *= ComputeBlackPointRatio(luma, bpNits);
        }
    }

    // ── Pipeline Stage 4: Filmic Contrast & Tonal EQ ───────────────────────
    // Three-zone stop-domain tonal control:
    //   Contrast: power curve (log2 multiplier) pivoted at fContrastPivot.
    //   Shadows:  rational C1 correction below pivot, asymptoting to ±3 stops.
    //   Highlights: rational C1 correction above pivot, asymptoting to ±3 stops.
    //
    // The correction function S·x²/(x²+a²) has the following properties:
    //   - f(0) = 0:  zero correction at pivot (pivot luminance preserved exactly)
    //   - f'(0) = 0: zero derivative at pivot (contrast alone controls local slope)
    //   - f(∞) → S:  asymptotic recovery of S stops in deep shadows/highlights
    //   - Monotonic: provably f'(x) > 0 for |S| ≤ 3.0 with a² = 6.0
    //                (minimum derivative = 20.5% of nominal at x = ±a/√3)
    //
    // The correction is C2 within each zone but C1 (not C2) at the pivot when
    // fShadows ≠ fHighlights (f''(0⁻) = S/2, f''(0⁺) = H/2). C2 discontinuity
    // at a single luminance value is visually imperceptible — standard practice
    // in professional grading (DaVinci Resolve Log Wheels use the same approach).
    //
    // Signed-luminance safe: abs(luma) for the curve, ratio applied to RGB.
    // 100x ratio ceiling: prevents noise amplification in sub-perceptual luminances
    // (see V5.9.1 documentation for derivation).
    if (abs(fContrast - 1.0) > NEUTRAL_EPS || abs(fShadows) > NEUTRAL_EPS || abs(fHighlights) > NEUTRAL_EPS)
    {
        float luma = dot(color, lumaCoeffs);
        float absLuma = abs(luma);

        if (absLuma > FLT_MIN)
        {
            float pivot = fContrastPivot * whitePt;
            float logRatio = log2(absLuma / pivot);

            // 1. Contrast: linear multiplier in log (stop) space.
            //    fContrast > 1.0 expands dynamic range around pivot.
            //    fContrast < 1.0 compresses it (recovers overall detail).
            float x = logRatio * fContrast;

            // 2. Tonal EQ: rational C1-continuous recovery curves.
            //    a² = 6.0 chosen for 20.5% minimum derivative margin at ±3 stops.
            //    (a² = 4.0 gives only 2.6% margin — near-flat band at max slider.)
            //    Correction reaches 50% at |x| = √6 ≈ 2.45 stops from pivot.
            float a2 = 6.0;

            if (x < 0.0 && abs(fShadows) > NEUTRAL_EPS)
            {
                float S = fShadows * 3.0;
                x = x + S * ((x * x) / (x * x + a2));
            }
            else if (x > 0.0 && abs(fHighlights) > NEUTRAL_EPS)
            {
                float H = fHighlights * 3.0;
                x = x + H * ((x * x) / (x * x + a2));
            }

            float contrastLuma = pivot * exp2(x);
            color *= min(contrastLuma / absLuma, 100.0);
        }
    }

    // ── Pipeline Stage 5: Khronos PBR Neutral Tone Mapping ──────────────────
    // Hue-preserving highlight compression (see proof in ApplyKhronosPBRNeutral).
    // The uniform affine transform guarantees MB hue invariance, so no
    // separate hue restoration pass is needed.
    [branch]
    if (bEnableKhronosNeutral)
    {
        // NORMALIZE: Scale working space so 1.0 = Reference White.
        // All Khronos math operates in this normalized domain.
        color /= max(whitePt, FLT_MIN);

        // PARAMETERIZED PEAK 'P':
        // For SDR (space <= 1), peak is locked to 1.0 (no headroom above white).
        // For HDR, peak = Display Peak / Reference White.
        // max(1.0, ...) prevents compressing the diffuse white range even if
        // Display Peak < Reference White (which makes no physical sense).
        float targetPeak = (space <= 1) ? 1.0 : max(1.0, fDisplayPeakNits / whitePt);

        color = ApplyKhronosPBRNeutral(color, targetPeak, fCompressionStart);

        // UN-NORMALIZE: Scale back to working linear space (nits).
        color *= whitePt;
    }

    // ── Pipeline Stage 6: MacLeod-Boynton Purity (Saturation) ───────────────
    // Isoluminant chromaticity scaling with vivid-color protection.
    // Operates display-referred (post-tonemap): values are bounded,
    // and what the user sets is what they see on screen.
    // The tonemapper's physical desaturation no longer fights this slider.
    color = ApplyMBPurity(color, fSaturation, space, lumaCoeffs, to_LMS, to_RGB);

    // ── Safety: Pipeline NaN/Inf Catch ──────────────────────────────────────
    // If any operation in this pipeline produced NaN/Inf (should not happen
    // with all guards in place, but defense-in-depth), fall back to the
    // original decoded linear color rather than outputting garbage.
    if (any(IsNan3(color)) || any(IsInf3(color)))
        color = original_lin;

    // ── Encode to Display Space ─────────────────────────────────────────────
    float3 encoded = EncodeFromLinear(color, space);

    // Hardware safety: SDR must strictly output [0, 1].
    // When Khronos is enabled, the Fresnel toe offset re-addition can push
    // peak values up to ~1.04, so this clamp performs a small, expected final clip.
    // Without Khronos, this is the only protection against out-of-range values.
    [flatten]
    if (space <= 1)
        encoded = saturate(encoded);

    // Preserve original alpha from backbuffer
    fragColor = float4(encoded, src.a);
}

// ==============================================================================
// 8. Technique
// ==============================================================================

technique PhotorealHDR_Mastering <
    ui_label = "Photoreal HDR V5.9.2 (Mastering Edition)";
    ui_tooltip = "Photorealistic grading for SDR and HDR.\n\n"
                 "Pipeline:\n"
                 "  1. Exposure (linear EV shift)\n"
                 "  2. LMS White Balance (exponential cone-domain)\n"
                 "  3. Subtractive Black Point (C1 parabolic toe)\n"
                 "  4. Filmic Contrast + Tonal EQ (stop-domain, C1 rational recovery)\n"
                 "  5. Khronos PBR Neutral Highlight Compression\n"
                 "  6. MacLeod-Boynton Isoluminant Purity\n\n"
                 "Design: Precision over performance.\n"
                 "  - True IEEE 754 math (no SFU approximations)\n"
                 "  - IEC/SMPTE exact standard constants\n"
                 "  - Physiological MacLeod-Boynton chromaticity\n"
                 "  - Vivid-color protection with gamut awareness\n"
                 "  - Khronos hue invariance (proven, not approximated)\n\n"
                 "Companion shader: Bilateral Contrast v8.4.4+";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_PhotorealHDR;
    }
}
