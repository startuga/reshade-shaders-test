// ============================================================================
// Photoreal HDR Color Grader (V5.5 - Mastering Edition)
// Companion shader to Bilateral Contrast v8.4.4
//
// V5.5 Changes from V5.4:
// - Fix: Filmic Contrast now preserves signed-luminance behavior
// - Fix: Intelligent Saturation with dark-chroma reliability ramp
//        aligned with Bilateral Contrast v8.4.3+
// - Fix: More conservative defaults for subtle photoreal starting point
// - Fix: Early exits in saturation skip Oklab round-trip when unnecessary
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

// sRGB thresholds
static const float SRGB_THRESHOLD_EOTF = 0.04045;
static const float SRGB_THRESHOLD_OETF = (0.04045 / 12.92);

// ST.2084 (PQ) EOTF Constants
static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;
static const float PQ_PEAK_LUMINANCE = 10000.0;

// Chroma reliability alignment with Bilateral Contrast
static const float CHROMA_STABILITY_THRESH = 1e-4;
static const float CHROMA_RELIABILITY_START = 5e-5;
static const float INV_CHROMA_RELIABILITY_SPAN =
    1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);

// ITU-R Luma Coefficients
static const float3 Luma709  = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020 = float3(0.2627, 0.6780, 0.0593);

// Oklab / LMS Matrices
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

static const float3x3 LMS_to_Oklab = float3x3(
    0.2104542553, 0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050, 0.4505937099,
    0.0259040371, 0.7827717662, -0.8086757660
);

static const float3x3 Oklab_to_LMSPrime = float3x3(
    1.0,  0.3963377774,  0.2158037573,
    1.0, -0.1055613458, -0.0638541728,
    1.0, -0.0894841775, -1.2914855480
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
    ui_label = "Saturation / Vibrance";
    ui_tooltip = "Intelligent Oklab chroma adjustment.\n"
                 "Above 1.0: vibrance-style boost (protects already vivid colors).\n"
                 "Below 1.0: uniform chroma reduction.\n"
                 "Near-black pixels fade toward neutral (bilateral-aligned reliability).\n"
                 "Rec.2020 receives gentler boost to avoid gamut boundary clipping.";
    ui_category = "Color Balance";
> = 1.08;

uniform int iColorSpaceOverride <
    ui_type = "combo";
    ui_label = "Color Space Override";
    ui_items = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0";
    ui_tooltip = "Must match Bilateral Contrast v8.4.4.";
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

float TrueSmoothstep(float edge0, float edge1, float x)
{
    float diff = edge1 - edge0;
    if (abs(diff) < FLT_MIN) return step(edge0, x);
    float t = saturate((x - edge0) / diff);
    return t * t * (3.0 - 2.0 * t);
}

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
    out_lin.r = (abs_V.r <= SRGB_THRESHOLD_EOTF) ? lin_lo.r : lin_hi.r;
    out_lin.g = (abs_V.g <= SRGB_THRESHOLD_EOTF) ? lin_lo.g : lin_hi.g;
    out_lin.b = (abs_V.b <= SRGB_THRESHOLD_EOTF) ? lin_lo.b : lin_hi.b;
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
// 6. Color Processing
// ==============================================================================

// Direct ratio form avoids luma-squared underflow for extremely small values.
// Parabolic toe is C1-continuous with the linear subtraction region at luma = 2*bpNits.
float ComputeBlackPointRatio(float luma, float bpNits)
{
    if (bpNits <= FLT_MIN || luma <= FLT_MIN)
        return 1.0;

    if (luma < 2.0 * bpNits) {
        return luma / (4.0 * bpNits);
    }

    return (luma - bpNits) / luma;
}

float3 ApplyLMSWhiteBalance(float3 color, float temp, float tint, int space, float3 lumaCoeffs)
{
    float3x3 to_LMS, to_RGB;

    [branch]
    if (space >= 3) {
        to_LMS = RGB2020_to_LMS;
        to_RGB = LMS_to_RGB2020;
    } else {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
    }

    // Exponential cone response: always positive, no channel collapse at extremes.
    // 0.35 maps +/-0.5 slider to +/-0.175 stops — gentle, camera-like response.
    float3 wbStops = 0.35 * float3(
         temp + tint,
        -tint,
        -temp + tint
    );

    float3 wbScale = exp2(wbStops);

    // Normalize so D65 neutrals keep exact luminance.
    float3 d65_wb_rgb = mul(to_RGB, wbScale);
    float lumaScale = dot(d65_wb_rgb, lumaCoeffs);
    wbScale /= max(abs(lumaScale), FLT_MIN);

    float3 lms = mul(to_LMS, color);
    lms *= wbScale;
    return mul(to_RGB, lms);
}

// Intelligent saturation: RGB-based vivid protection + Oklab chroma modification.
// Protection metric uses RGB channel spread (fast, numerically stable).
// Actual chroma scaling uses Oklab (perceptually uniform, hue-preserving).
// Dark-chroma reliability aligned with Bilateral Contrast v8.4.3+.
//
// Early exits skip the Oklab round-trip when:
//   1. Slider is at neutral (1.0)
//   2. Chroma reliability is zero (near-black pixel)
//   3. Effective chroma gain collapses to neutral after reliability fade
float3 ApplyIntelligentSaturation(float3 color, float saturation, int space, float3 lumaCoeffs)
{
    // Exit 1: Slider at neutral
    if (abs(saturation - 1.0) < 1e-6)
        return color;

    float luma = dot(color, lumaCoeffs);

    // Dark-chroma reliability: fade chroma changes in unstable near-black regions.
    // Negative luma -> reliability = 0 -> bypass (correct for out-of-gamut scRGB).
    float ct = saturate((luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);

    // Exit 2: Near-black pixel — chroma is numerically unstable
    if (chroma_reliability <= 0.0)
        return color;

    // Vivid-color protection metric from RGB channel spread.
    // peak uses abs() to handle scRGB negatives as denominator;
    // max_c/min_c use signed values so (max_c - min_c) measures true channel spread.
    float peak = max(abs(color.r), max(abs(color.g), abs(color.b)));
    float max_c = max(color.r, max(color.g, color.b));
    float min_c = min(color.r, min(color.g, color.b));

    float sat_current = 0.0;
    if (peak > 1e-6)
        sat_current = saturate((max_c - min_c) / peak);

    // Smoothstep makes protection onset gradual rather than linear
    float protection = sat_current * sat_current * (3.0 - 2.0 * sat_current);

    float chroma_gain = saturation;

    if (saturation > 1.0)
    {
        float boost = saturation - 1.0;
        // Wider gamut (Rec.2020) gets gentler boost to avoid gamut boundary clipping.
        float space_comp = (space >= 3) ? 0.90 : 1.0;
        // At maximum protection (fully vivid color), a residual fraction of
        // the boost still applies. This prevents a hard "wall" where colors
        // just below the protection threshold get boosted but those above don't.
        // Rec.2020: 20% residual (wider gamut, closer to clipping).
        // Rec.709:  25% residual (tighter gamut, more headroom).
        float min_boost_share = (space >= 3) ? 0.20 : 0.25;

        chroma_gain = 1.0 + boost * space_comp * lerp(1.0, min_boost_share, protection);
    }

    // Apply dark-chroma reliability: unreliable pixels fade to identity (gain=1.0)
    chroma_gain = lerp(1.0, chroma_gain, chroma_reliability);

    // Exit 3: Effective gain collapsed to neutral (e.g., vivid color at low reliability)
    if (abs(chroma_gain - 1.0) < 1e-6)
        return color;

    // Oklab round-trip for perceptually uniform chroma modification
    float3x3 to_LMS, to_RGB;

    [branch]
    if (space >= 3) {
        to_LMS = RGB2020_to_LMS;
        to_RGB = LMS_to_RGB2020;
    } else {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
    }

    float3 lms = mul(to_LMS, color);
    float3 lms_p = sign(lms) * pow(max(abs(lms), FLT_MIN), 1.0 / 3.0);
    float3 lab = mul(LMS_to_Oklab, lms_p);

    lab.yz *= chroma_gain;

    float3 lms_p_out = mul(Oklab_to_LMSPrime, lab);
    float3 lms_out = lms_p_out * lms_p_out * lms_p_out;

    return mul(to_RGB, lms_out);
}

// ==============================================================================
// 7. Main Shader
// ==============================================================================

void PS_PhotorealHDR(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 fragColor : SV_Target)
{
    int2 pos = int2(vpos.xy);
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;
    float whitePt = (space <= 1) ? SCRGB_WHITE_NITS : fWhitePoint;

    // Fast bypass when all controls are at neutral positions
    [branch]
    if (fExposure == 0.0 && fBlackPoint == 0.0 && fContrast == 1.0 &&
        fTemperature == 0.0 && fTint == 0.0 && fSaturation == 1.0) {
        fragColor = tex2Dfetch(SamplerBackBuffer, pos);
        return;
    }

    // Cache original for NaN/Inf fallback
    float3 original_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, pos).rgb, space);
    float3 color = original_lin;

    // 1. Exposure
    if (fExposure != 0.0)
        color *= exp2(fExposure);

    // 2. White Balance
    if (fTemperature != 0.0 || fTint != 0.0)
        color = ApplyLMSWhiteBalance(color, fTemperature, fTint, space, lumaCoeffs);

    // 3. Dehaze / Black Point
    if (fBlackPoint > 0.0)
    {
        float luma = dot(color, lumaCoeffs);
        if (luma > FLT_MIN)
        {
            float bpNits = fBlackPoint * whitePt;
            float bpRatio = ComputeBlackPointRatio(luma, bpNits);
            color *= bpRatio;
        }
    }

    // 4. Filmic Contrast (signed-luminance safe)
    // Uses abs(luma) so negative-luminance scRGB pixels get the same contrast
    // ratio as their absolute value, preserving WCG channel signs.
    // Zero-luminance guard prevents ratio explosion at sub-FLT_MIN values
    // when fContrast < 1.0 (pow amplifies tiny bases with fractional exponents).
    if (abs(fContrast - 1.0) > 1e-6)
    {
        float luma = dot(color, lumaCoeffs);
        float absLuma = abs(luma);

        if (absLuma > FLT_MIN)
        {
            float pivot = 0.18 * whitePt;
            float contrastLuma = PowNonNegPreserveZero(absLuma / pivot, fContrast) * pivot;
            float contrastRatio = min(contrastLuma / absLuma, 100.0);
            color *= contrastRatio;
        }
    }

    // 5. Intelligent Saturation
    color = ApplyIntelligentSaturation(color, fSaturation, space, lumaCoeffs);

    // Safety
    if (any(IsNan3(color)) || any(IsInf3(color)))
        color = original_lin;

    // Encode to display space
    float3 encoded = EncodeFromLinear(color, space);

    [flatten]
    if (space <= 1)
        encoded = saturate(encoded);

    fragColor = float4(encoded, 1.0);
}

// ==============================================================================
// 8. Technique
// ==============================================================================

technique PhotorealHDR_Mastering <
    ui_label = "Photoreal HDR V5.5 (Mastering Edition)";
    ui_tooltip = "Photorealistic grading for SDR and HDR.\n\n"
                 "Pipeline:\n"
                 "  1. Exposure (linear EV shift)\n"
                 "  2. LMS White Balance (exponential, luminance-preserving)\n"
                 "  3. Subtractive Black Point (C1 parabolic toe)\n"
                 "  4. Filmic Contrast (signed-luminance safe)\n"
                 "  5. Intelligent Saturation (Oklab chroma, dark reliability fade)\n\n"
                 "V5.5 Fixes:\n"
                 "- Fix: Signed-luminance contrast with zero-guard\n"
                 "- Fix: Saturation dark-chroma reliability aligned with bilateral\n"
                 "- Fix: Gamut-width-aware boost compensation (Rec.2020 gentler)\n"
                 "- Fix: Early exits skip Oklab round-trip when unnecessary\n\n"
                 "Companion shader: Bilateral Contrast v8.4.4";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_PhotorealHDR;
    }
}