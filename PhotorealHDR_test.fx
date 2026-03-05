// =========================================================================
// Photoreal HDR Color Grader (V5 - Mastering Edition)
// Designed to remove "game-y" dusty filters and yellow tints.
// Companion shader to Bilateral Contrast v8.4.1.
//
// V5 Changes from V4:
// - Fix: Black point scaled relative to white point (visible in SDR and HDR)
// - Fix: Luminance-based contrast (preserves chromaticity — true photoreal)
// - Fix: Luminance-preserving LMS white balance (no brightness shift from WB)
// - Fix: sRGB OETF threshold matches v8.4.1 exactly (bit-exact round-trip)
// - Fix: Processing order matches photographic workflow (WB before tone)
// - Fix: Contrast pivot scales with white point (correct for HDR)
// - Fix: NaN/Inf output guard
// - Fix: tex2Dfetch with POINT sampling (no bilinear interpolation)
// - Fix: [flatten] hints on uniform branches
// =========================================================================

#include "ReShade.fxh"

// ==============================================================================
// 1. Constants (Matching Bilateral Contrast v8.4.1)
// ==============================================================================

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

static const float FLT_MIN = 1.175494351e-38;
static const float SCRGB_WHITE_NITS = 80.0;

// sRGB OETF threshold derived from EOTF for perfect round-trip (matches v8.4.1)
static const float SRGB_THRESHOLD_OETF = (0.04045 / 12.92);

// ST.2084 (PQ) EOTF Constants (SMPTE ST 2084-2014)
static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;
static const float PQ_PEAK_LUMINANCE = 10000.0;

// ITU-R Luma Coefficients
static const float3 Luma709  = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020 = float3(0.2627, 0.6780, 0.0593);

// LMS Matrices (Oklab M1 / Inverse — Björn Ottosson 2020)
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

// Row-Sum-Normalized Rec.2020 → LMS (matching v8.4.1)
static const float3x3 RGB2020_to_LMS = float3x3(
    0.616759697, 0.360188024, 0.023052279,
    0.265131674, 0.635851580, 0.099016746,
    0.100127915, 0.203878384, 0.695993701
);

// Inverse of RGB2020_to_LMS
// NOTE: Limited to 7 significant digits. Round-trip error ~0.05%.
// The luminance-preserving WB normalization compensates for residual error.
static const float3x3 LMS_to_RGB2020 = float3x3(
     2.1398453, -1.2462738,  0.1064285,
    -0.8846699,  2.1631066, -0.2784367,
    -0.0486970, -0.4543485,  1.5030455
);

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
    ui_tooltip = "Subtracts a percentage of the white point from luminance.\n"
                 "Cuts through the 'dusty' lifted black levels.\n"
                 "0.005 = 0.5%% of white (0.4 nits SDR, 1.0 nit HDR).";
    ui_category = "Tone & Exposure";
> = 0.005;

uniform float fContrast <
    ui_type = "slider";
    ui_min = 0.80; ui_max = 1.50; ui_step = 0.001;
    ui_label = "Filmic Contrast";
    ui_tooltip = "Luminance-based power curve pivoted at 18%% grey.\n"
                 "Preserves chromaticity (hue and saturation unchanged).";
    ui_category = "Tone & Exposure";
> = 1.05;

uniform float fTemperature <
    ui_type = "slider";
    ui_min = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label = "Color Temperature (LMS)";
    ui_tooltip = "Negative = Cooler (removes yellow/sand tint)\n"
                 "Positive = Warmer\n"
                 "Luminance-preserving: brightness stays constant.";
    ui_category = "Color Balance";
> = -0.12;

uniform float fTint <
    ui_type = "slider";
    ui_min = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label = "Color Tint (LMS)";
    ui_tooltip = "Negative = Greener\nPositive = More Magenta";
    ui_category = "Color Balance";
> = 0.02;

uniform float fSaturation <
    ui_type = "slider";
    ui_min = 0.00; ui_max = 2.00; ui_step = 0.01;
    ui_label = "Saturation";
    ui_tooltip = "Luminance-based linear interpolation.\n"
                 "Values above 1.0 may push saturated colors out of gamut.";
    ui_category = "Color Balance";
> = 1.15;

uniform int iColorSpaceOverride <
    ui_type = "combo";
    ui_label = "Color Space Override";
    ui_items = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0";
    ui_tooltip = "MUST MATCH the setting used in Bilateral Contrast.";
    ui_category = "System";
> = 0;

uniform float fWhitePoint <
    ui_type = "slider";
    ui_min = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_label = "Reference White (Nits)";
    ui_tooltip = "Should match 'Zone White Point' in Bilateral Contrast.\n"
                 "Only affects HDR modes. SDR is fixed at 80 nits.\n"
                 "203 = ITU-R BT.2408 reference diffuse white.";
    ui_category = "System";
> = 203.0;

// ==============================================================================
// 4. Math Utilities
// ==============================================================================

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

bool IsNanVal(float x) { return (asuint(x) & 0x7FFFFFFF) > 0x7F800000; }
bool IsInfVal(float x) { return (asuint(x) & 0x7FFFFFFF) == 0x7F800000; }
bool3 IsNan3(float3 v) { return bool3(IsNanVal(v.x), IsNanVal(v.y), IsNanVal(v.z)); }
bool3 IsInf3(float3 v) { return bool3(IsInfVal(v.x), IsInfVal(v.y), IsInfVal(v.z)); }

// ==============================================================================
// 5. EOTF / OETF (Exact match with Bilateral Contrast v8.4.1)
// ==============================================================================

float3 sRGB_EOTF(float3 V)
{
    float3 abs_V = abs(V);
    float3 lin_lo = abs_V / 12.92;
    float3 lin_hi = PowNonNegPreserveZero3((abs_V + 0.055) / 1.055, 2.4);
    float3 out_lin;
    out_lin.r = (abs_V.r <= 0.04045) ? lin_lo.r : lin_hi.r;
    out_lin.g = (abs_V.g <= 0.04045) ? lin_lo.g : lin_hi.g;
    out_lin.b = (abs_V.b <= 0.04045) ? lin_lo.b : lin_hi.b;
    return sign(V) * out_lin;
}

float3 sRGB_OETF(float3 L)
{
    float3 abs_L = abs(L);
    float3 enc_lo = abs_L * 12.92;
    float3 enc_hi = 1.055 * PowNonNegPreserveZero3(abs_L, 1.0 / 2.4) - 0.055;
    float3 out_enc;
    // [v5 Fix] Use derived threshold matching v8.4.1 for bit-exact round-trip
    out_enc.r = (abs_L.r <= SRGB_THRESHOLD_OETF) ? enc_lo.r : enc_hi.r;
    out_enc.g = (abs_L.g <= SRGB_THRESHOLD_OETF) ? enc_lo.g : enc_hi.g;
    out_enc.b = (abs_L.b <= SRGB_THRESHOLD_OETF) ? enc_lo.b : enc_hi.b;
    return sign(L) * out_enc;
}

float3 PQ_EOTF(float3 N)
{
    float3 Np = PowNonNegPreserveZero3(max(N, 0.0), 1.0 / PQ_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    return PowNonNegPreserveZero3(num / den, 1.0 / PQ_M1) * PQ_PEAK_LUMINANCE;
}

float3 PQ_InverseEOTF(float3 L)
{
    float3 Lp = PowNonNegPreserveZero3(max(L, 0.0) / PQ_PEAK_LUMINANCE, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    return PowNonNegPreserveZero3(num / den, PQ_M2);
}

// [v5 Fix] Added [flatten] to prevent warp divergence on uniform branch
float3 DecodeToLinear(float3 encoded, int space)
{
    [flatten] if (space == 3) return PQ_EOTF(encoded);
    [flatten] if (space == 2) return encoded * SCRGB_WHITE_NITS;
    return sRGB_EOTF(encoded) * SCRGB_WHITE_NITS;
}

float3 EncodeFromLinear(float3 lin, int space)
{
    [flatten] if (space == 3) return PQ_InverseEOTF(lin);
    [flatten] if (space == 2) return lin / SCRGB_WHITE_NITS;
    return sRGB_OETF(lin / SCRGB_WHITE_NITS);
}

// ==============================================================================
// 6. Photoreal Processing Functions
// ==============================================================================

// [v5 Fix] Luminance-preserving LMS White Balance
// Normalizes WB scaling at D65 to prevent brightness shifts from color corrections.
// Since LMS matrices are row-sum-1, D65 white (1,1,1) RGB maps to (1,1,1) LMS.
// After WB scaling, we measure the luminance change at D65 and invert it.
float3 ApplyLMSWhiteBalance(float3 color, float temp, float tint, int space, float3 lumaCoeffs)
{
    float3x3 to_LMS, to_RGB;
    // [Fix X3020] Use if/else for matrix selection
    [flatten]
    if (space >= 3) {
        to_LMS = RGB2020_to_LMS;
        to_RGB = LMS_to_RGB2020;
    } else {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
    }

    // LMS cone response scaling:
    // Temperature: shifts L vs S (red-yellow vs blue axis, Planckian locus)
    // Tint: shifts (L+S) vs M (magenta vs green axis, perpendicular to Planckian)
    float3 wbScale = float3(
        1.0 + temp + tint, // L (long): warm↑ magenta↑
        1.0 - tint,        // M (medium): magenta↓ green↑
        1.0 - temp + tint  // S (short): warm↓ cool↑ magenta↑
    );

    // Compute D65 luminance normalization factor (uniform per frame, not per pixel).
    // D65 white in achromatic-normalized LMS = (1,1,1).
    // After WB: LMS = wbScale. Convert back to RGB and measure luminance change.
    float3 d65_wb_rgb = mul(to_RGB, wbScale);
    float lumaScale = dot(d65_wb_rgb, lumaCoeffs);

    // Pre-normalize wbScale so neutral grey maintains exact luminance.
    // Guard: lumaScale should be positive for any reasonable temp/tint within ±0.5.
    wbScale /= max(lumaScale, FLT_MIN);

    // Apply: RGB → LMS → scale → RGB
    float3 lms = mul(to_LMS, color);
    lms *= wbScale;
    return mul(to_RGB, lms);
}

// ==============================================================================
// 7. Main Shader
// ==============================================================================

void PS_PhotorealHDR(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 fragColor : SV_Target)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;
    float whitePt = (space <= 1) ? SCRGB_WHITE_NITS : fWhitePoint;

    // === 1. Decode to Linear Light (nits) ===
    // [v5 Fix] tex2Dfetch with POINT sampling for bit-exact pixel reads
    float3 color = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, int2(vpos.xy)).rgb, space);

    // === 2. Exposure (linear energy multiplier, applied first) ===
    color *= exp2(fExposure);

    // === 3. White Balance (LMS, luminance-preserving) ===
    // [v5 Fix] Processing order: WB before tonal adjustments matches camera raw workflow.
    // Color correction establishes the "true" scene colors before contrast/dehaze operate.
    color = ApplyLMSWhiteBalance(color, fTemperature, fTint, space, lumaCoeffs);

    // === 4. Black Point / Dehaze ===
    // [v5 Fix] Scale black point relative to white point for meaningful range in SDR and HDR.
    // fBlackPoint = 0.005 → 0.5% of white → 0.4 nits (SDR) or 1.0 nit (HDR @ 203).
    float bpNits = fBlackPoint * whitePt;
    float luma = dot(color, lumaCoeffs);
    float absLuma = max(abs(luma), FLT_MIN);
    float newLuma = max(0.0, absLuma - bpNits);
    // Ratio-preserving scale: all channels dimmed equally, preserving chromaticity.
    // Below bpNits, newLuma = 0 → pixel goes to black (hard clip, intentional for dehaze).
    color *= newLuma / absLuma;

    // === 5. Contrast (Luminance-based, chromaticity-preserving) ===
    // [v5 Fix] Changed from per-channel pow to luminance-based ratio scaling.
    // Per-channel contrast shifts hue and boosts saturation — not photoreal.
    // Luminance-based contrast only modifies tonal relationships while preserving color.
    // Pivot = 18% grey (Zone V, standard photographic reflectance).
    float pivot = 0.18 * whitePt;
    luma = dot(color, lumaCoeffs);
    absLuma = max(abs(luma), FLT_MIN);
    // Only apply contrast to pixels with meaningful luminance.
    // Below 1e-4 nits (~-13 stops), contrast amplifies quantization noise.
    if (absLuma > 1e-4) {
        float contrastLuma = pow(absLuma / pivot, fContrast) * pivot;
        color *= contrastLuma / absLuma;
    }

    // === 6. Saturation ===
    // Recalculate luma after all tonal modifications.
    luma = dot(color, lumaCoeffs);
    color = lerp((float3)luma, color, fSaturation);

    // === 7. Safety: NaN/Inf Guard ===
    // Extreme parameter combinations (high contrast + saturation) can overflow pow().
    if (any(IsNan3(color)) || any(IsInf3(color))) {
        // Fallback: re-decode original pixel (safe passthrough)
        color = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, int2(vpos.xy)).rgb, space);
    }

    // === 8. Encode to Display Space ===
    float3 encoded = EncodeFromLinear(color, space);

    [flatten]
    if (space <= 1) {
        encoded = saturate(encoded);
    }

    fragColor = float4(encoded, 1.0);
}

// ==============================================================================
// 8. Technique
// ==============================================================================

technique PhotorealHDR_Mastering <
    ui_label = "Photoreal HDR V5 (Mastering Edition)";
    ui_tooltip = "Photorealistic grading designed for HDR and SDR displays.\n\n"
                 "Processing Pipeline:\n"
                 "  1. Exposure (linear EV shift)\n"
                 "  2. White Balance (luminance-preserving LMS)\n"
                 "  3. Dehaze / Black Point (relative to white point)\n"
                 "  4. Filmic Contrast (luminance-based, chromaticity-preserving)\n"
                 "  5. Saturation (luminance-weighted)\n\n"
                 "V5 Fixes:\n"
                 "- Luminance-preserving white balance\n"
                 "- Luminance-based contrast (no hue/saturation shift)\n"
                 "- Black point scaled to white point (works in SDR and HDR)\n"
                 "- Matched v8.4.1 EOTF/OETF thresholds\n"
                 "- NaN/Inf output safety\n\n"
                 "Companion shader: Bilateral Contrast v8.4.1";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_PhotorealHDR;
    }
}
