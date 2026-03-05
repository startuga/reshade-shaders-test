// =========================================================================
// Photoreal HDR Color Grader (V5.2 - Mastering Edition)
// Designed to remove "game-y" dusty filters and yellow tints.
// Companion shader to Bilateral Contrast v8.4.1.
//
// V5.2 Changes from V5.1:
// - Feature: Smart Saturation / Perceptual Vibrance (Protects highly 
//   saturated colors like skies from neon-clipping while boosting dull tones)
// - Fix: UI Labels updated to reflect Vibrance logic
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

// Row-Sum-Normalized Rec.2020 -> LMS (matching v8.4.1)
static const float3x3 RGB2020_to_LMS = float3x3(
    0.616759697, 0.360188024, 0.023052279,
    0.265131674, 0.635851580, 0.099016746,
    0.100127915, 0.203878384, 0.695993701
);

// Inverse of RGB2020_to_LMS
// NOTE: Limited to 7 significant digits. Round-trip error ~0.05%.
// The luminance-preserving WB normalization compensates for residual brightness error.
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
                 "0.005 = 0.5%% of white (0.4 nits SDR, 1.0 nit HDR).\n"
                 "Uses smooth Hermite rolloff (no hard clip edge).";
    ui_category = "Tone & Exposure";
> = 0.005;

uniform float fContrast <
    ui_type = "slider";
    ui_min = 0.80; ui_max = 1.50; ui_step = 0.001;
    ui_label = "Filmic Contrast";
    ui_tooltip = "Luminance-based power curve pivoted at 18%% grey.\n"
                 "Preserves chromaticity (hue and saturation unchanged).\n"
                 "Values above 1.0 push shadows darker and highlights brighter.";
    ui_category = "Tone & Exposure";
> = 1.05;

uniform float fTemperature <
    ui_type = "slider";
    ui_min = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label = "Color Temperature (LMS)";
    ui_tooltip = "Negative = Cooler (removes yellow/sand tint)\n"
                 "Positive = Warmer\n"
                 "Luminance-preserving for neutral tones.\n"
                 "Saturated colors may shift ~1-3%% in luminance.";
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
    ui_label = "Saturation / Vibrance";
    ui_tooltip = "Smart Perceptual Saturation.\n"
                 "Boosts dull colors (sand, stone) while protecting already vivid colors (sky, fire) from neon-clipping.";
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
// 5. EOTF / OETF
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

// Luminance-preserving LMS White Balance
// Normalization ensures D65 neutral tones maintain exact luminance.
// Saturated colors may have slight residual luminance shift (~1-3%) due to
// non-diagonal LMS scaling — this is inherent to all multiplicative WB.
float3 ApplyLMSWhiteBalance(float3 color, float temp, float tint, int space, float3 lumaCoeffs)
{
    float3x3 to_LMS, to_RGB;

    [flatten]
    if (space >= 3) {
        to_LMS = RGB2020_to_LMS;
        to_RGB = LMS_to_RGB2020;
    } else {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
    }

    // LMS cone response scaling
    float3 wbScale = float3(
        1.0 + temp + tint, // L (long): warm↑ magenta↑
        1.0 - tint,        // M (medium): magenta↓ green↑
        1.0 - temp + tint  // S (short): warm↓ cool↑ magenta↑
    );

    // Compute D65 luminance normalization factor.
    // D65 white in LMS = (1,1,1) due to row-sum-normalized matrices.
    // After WB: LMS = wbScale. Convert back to RGB and measure luminance.
    float3 d65_wb_rgb = mul(to_RGB, wbScale);
    float lumaScale = dot(d65_wb_rgb, lumaCoeffs);

    // Pre-normalize so neutral grey maintains exact luminance.
    wbScale /= max(abs(lumaScale), FLT_MIN);

    // Apply: RGB -> LMS -> scale -> RGB
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

    // Fast bypass when all controls are at neutral positions
    [branch]
    if (fExposure == 0.0 && fBlackPoint == 0.0 && fContrast == 1.0 &&
        fTemperature == 0.0 && fTint == 0.0 && fSaturation == 1.0) {
        fragColor = tex2Dfetch(SamplerBackBuffer, int2(vpos.xy));
        return;
    }

    // === 1. Decode to Linear Light (nits) ===
    // Cache original for NaN/Inf fallback
    float3 original_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, int2(vpos.xy)).rgb, space);
    float3 color = original_lin;

    // === 2. Exposure (linear energy multiplier, applied first) ===
    color *= exp2(fExposure);

    // === 3. White Balance (LMS, luminance-preserving) ===
    color = ApplyLMSWhiteBalance(color, fTemperature, fTint, space, lumaCoeffs);

    // === 4. Black Point / Dehaze (Smooth Hermite Knee) ===
    // [v5.1 Fix] Replaced hard max(0, luma - bp) with C1-continuous Hermite rolloff.
    // Eliminates visible banding/contour at the clip threshold in smooth gradients.
    // Transition zone: [0, 2*bpNits]. At default 0.005*80 = 0.4 nits, zone is 0-0.8 nits.
    float bpNits = fBlackPoint * whitePt;
    float luma = dot(color, lumaCoeffs);
    float absLuma = max(abs(luma), FLT_MIN);

    float bpRatio;
    if (bpNits < FLT_MIN) {
        bpRatio = 1.0;  // No black point adjustment
    } else {
        float t = saturate(absLuma / (2.0 * bpNits));  // 0 at black, 1 at 2*bpNits
        bpRatio = t * t * (3.0 - 2.0 * t);             // Hermite smoothstep (C1 continuous)
    }
    color *= bpRatio;

    // === 5. Contrast (Luminance-based, chromaticity-preserving) ===
    float pivot = 0.18 * whitePt;
    luma = dot(color, lumaCoeffs);
    absLuma = max(abs(luma), FLT_MIN);

    float contrastLuma = PowNonNegPreserveZero(absLuma / pivot, fContrast) * pivot;
    // [v5.1 Fix] Clamp ratio to prevent FP overflow with extreme HDR values + high contrast.
    // 100x allows ~6.6 stops of boost while preventing Inf propagation.
    float contrastRatio = min(contrastLuma / absLuma, 100.0);
    color *= contrastRatio;

    // === 6. Perceptual Vibrance / Smart Saturation ===
    // Calculates pixel purity (current saturation) in linear space
    float max_c = max(color.r, max(color.g, color.b));
    float min_c = min(color.r, min(color.g, color.b));
    
    // Avoids div-by-zero and deep black artifacts
    float sat_current = (max_c > 1e-6) ? (max_c - min_c) / max_c : 0.0;
    
    // Color protection curve: already vivid colors reduce the saturation impact.
    // Factor 0.75 dictates the protection strength (higher = more protection for vibrant colors).
    float vibrance_protection = 1.0 - (sat_current * 0.75);
    
    // Recalculate luma after tonal modifications
    luma = dot(color, lumaCoeffs);
    
    // The effective saturation adapts to each pixel
    float effective_sat = 1.0 + (fSaturation - 1.0) * vibrance_protection;
    
    color = lerp((float3)luma, color, effective_sat);

    // === 7. Safety: NaN/Inf Guard ===
    if (any(IsNan3(color)) || any(IsInf3(color))) {
        color = original_lin;
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
    ui_label = "Photoreal HDR V5.2 (Mastering Edition)";
    ui_tooltip = "Photorealistic grading designed for HDR and SDR displays.\n\n"
                 "Processing Pipeline:\n"
                 "  1. Exposure (linear EV shift)\n"
                 "  2. White Balance (luminance-preserving LMS)\n"
                 "  3. Dehaze / Black Point (smooth Hermite knee)\n"
                 "  4. Filmic Contrast (luminance-based, chromaticity-preserving)\n"
                 "  5. Smart Vibrance (perceptual saturation)\n\n"
                 "V5.2 Fixes:\n"
                 "- Added Perceptual Vibrance logic (protects naturally vivid colors)\n"
                 "- Smooth Hermite black point (no gradient banding)\n"
                 "- Contrast ratio overflow protection for HDR\n\n"
                 "Companion shader: Bilateral Contrast v8.4.1";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_PhotorealHDR;
    }
}
