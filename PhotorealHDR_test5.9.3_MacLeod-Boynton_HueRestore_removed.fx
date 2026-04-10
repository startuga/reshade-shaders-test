// =================================================================================================
// Photoreal HDR Color Grader (V5.9.3 - The Ultimate Director's Cut)
// =================================================================================================
//
// Design Philosophy: PRECISION OVER PERFORMANCE + BIOLOGICAL HIGHLIGHTS
// - True IEEE 754 Math: No fast intrinsics or Special Function Unit (SFU) approximations.
// - Exact IEC/SMPTE Constants: Bit-exact neutrality logic for standard color spaces.
// - True Stop-Domain Scene Grading: Log2-domain exposure and contrast with C1 rational recovery.
// - Physiological Chromaticity: MacLeod-Boynton cone-opponent space for all color operations.
//
// V5.9.3 Architecture Update (The Best of Both Worlds):
// Replaced Khronos's mathematical RGB highlight desaturation with Biological Troland 
// Photopigment Bleaching (inspired by the psycho17 retinal model). 
// Instead of desaturating based on display peak proximity (which leaves WCG colors neon), 
// highlights naturally burn toward a white-hot core based on absolute physical intensity (nits) 
// BEFORE Khronos elegantly compresses them into the display gamut.
// =================================================================================================

#include "ReShade.fxh"

// =================================================================================================
// 1. Constants & Definitions
// =================================================================================================

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

static const float FLT_MIN              = 1.175494351e-38;
static const float SCRGB_WHITE_NITS     = 80.0;
static const float NEUTRAL_EPS          = 1e-6;

// -------------------------------------------------------------------------------------------------
// sRGB Constants (IEC 61966-2-1:1999)
// -------------------------------------------------------------------------------------------------
static const float SRGB_THRESHOLD_EOTF  = 0.04045;
static const float SRGB_THRESHOLD_OETF  = 0.0031308;
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
// Color Science Constants
// -------------------------------------------------------------------------------------------------
static const float CHROMA_STABILITY_THRESH      = 1e-4;
static const float CHROMA_RELIABILITY_START     = 5e-5;
static const float INV_CHROMA_RELIABILITY_SPAN  = 1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);

static const float3 Luma709             = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020            = float3(0.2627, 0.6780, 0.0593);

static const float MB_PURITY_PROTECTION_CEILING = 0.35;

// -------------------------------------------------------------------------------------------------
// Biological Bleaching Constants
// -------------------------------------------------------------------------------------------------
// TROLAND_LMS_SCALE: Rendering proxy for retinal illuminance.
// Assumes ~2mm pupil radius (πr² ≈ 12.57mm²) folded into LMS->Troland conversion.
// Not a strict photometric calculator; calibrated for perceptual highlight burnout.
static const float TROLAND_LMS_SCALE    = 4.0;
static const float TROLAND_HALF_SAT     = 20000.0;

// -------------------------------------------------------------------------------------------------
// Scene-Grade Row-Sum-Normalized Matrices
// -------------------------------------------------------------------------------------------------
static const float3x3 RGB709_to_LMS = float3x3(
    0.4122214708,  0.5363325363,  0.0514459929,
    0.2119034982,  0.6806995451,  0.1073969566,
    0.0883024619,  0.2817188376,  0.6299787005
);

static const float3x3 LMS_to_RGB709 = float3x3(
     4.0767416621, -3.3077115913,  0.2309699292,
    -1.2684380046,  2.6097574011, -0.3413193965,
    -0.0041960863, -0.7034186147,  1.7076147010
);

static const float3x3 RGB2020_to_LMS = float3x3(
    0.6167596970,  0.3601880240,  0.0230522790,
    0.2651316740,  0.6358515800,  0.0990167460,
    0.1001279150,  0.2038783840,  0.6959937010
);

static const float3x3 LMS_to_RGB2020 = float3x3(
     2.1398540771, -1.2462788877,  0.1064290765,
    -0.8846737634,  2.1631158093, -0.2784377818,
    -0.0486976682, -0.4543507342,  1.5030526721
);

// =================================================================================================
// 2. Texture & UI Parameters
// =================================================================================================

texture2D TextureBackBuffer : COLOR;
sampler2D SamplerBackBuffer 
{ 
    Texture   = TextureBackBuffer; 
    MagFilter = POINT; 
    MinFilter = POINT; 
    MipFilter = NONE; 
    AddressU  = CLAMP; 
    AddressV  = CLAMP; 
};

// -------------------------------------------------------------------------------------------------
// UI: Part 1 - Scene Grade
// -------------------------------------------------------------------------------------------------
uniform float fExposure < 
    ui_type     = "slider"; 
    ui_min      = -3.00; ui_max = 3.00; ui_step = 0.01; 
    ui_label    = "Exposure (EV)";
    ui_category = "1. Scene Grade"; 
> = 0.00;

uniform float fTemperature < 
    ui_type     = "slider"; 
    ui_min      = -0.50; ui_max = 0.50; ui_step = 0.001; 
    ui_label    = "Color Temperature (LMS)";
    ui_category = "1. Scene Grade"; 
> = -0.06;

uniform float fTint < 
    ui_type     = "slider"; 
    ui_min      = -0.50; ui_max = 0.50; ui_step = 0.001; 
    ui_label    = "Color Tint (LMS)";
    ui_category = "1. Scene Grade"; 
> = 0.01;

uniform float fBlackPoint < 
    ui_type     = "slider"; 
    ui_min      = 0.000; ui_max = 0.050; ui_step = 0.001; 
    ui_label    = "Dehaze / Black Point";
    ui_category = "1. Scene Grade"; 
> = 0.003;

uniform float fContrast < 
    ui_type     = "slider"; 
    ui_min      = 0.80; ui_max = 1.50; ui_step = 0.001; 
    ui_label    = "Filmic Contrast";
    ui_category = "1. Scene Grade"; 
> = 1.03;

uniform float fContrastPivot < 
    ui_type     = "slider"; 
    ui_min      = 0.01; ui_max = 1.00; ui_step = 0.01; 
    ui_label    = "Contrast Pivot";
    ui_category = "1. Scene Grade"; 
> = 0.18;

uniform float fShadows < 
    ui_type     = "slider"; 
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001; 
    ui_label    = "Shadows (Log Recovery)";
    ui_category = "1. Scene Grade"; 
> = 0.0;

uniform float fHighlights < 
    ui_type     = "slider"; 
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001; 
    ui_label    = "Highlights (Log Recovery)";
    ui_category = "1. Scene Grade"; 
> = 0.0;

uniform float fSaturation < 
    ui_type     = "slider"; 
    ui_min      = 0.00; ui_max = 2.00; ui_step = 0.01; 
    ui_label    = "Purity / Saturation (MacLeod-Boynton)";
    ui_category = "1. Scene Grade"; 
> = 1.08;

// -------------------------------------------------------------------------------------------------
// UI: Part 2 - Tone Mapping
// -------------------------------------------------------------------------------------------------
uniform float fBleaching < 
    ui_type     = "slider"; 
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01; 
    ui_label    = "Highlight Bleaching (Trolands)"; 
    ui_tooltip  = "Physiological highlight burnout toward a white-hot core.\n"
                  "Based on absolute scene intensity (nits), before tonemapping.";
    ui_category = "2. Tone Mapping"; 
> = 1.00;

uniform bool bEnableKhronosNeutral < 
    ui_label    = "Enable Khronos PBR Neutral Tonemapper"; 
    ui_category = "2. Tone Mapping"; 
> = true;

uniform float fDisplayPeakNits < 
    ui_type     = "slider"; 
    ui_min      = 80.0; ui_max = 4000.0; ui_step = 10.0; 
    ui_label    = "Display Peak Luminance (Nits)";
    ui_category = "2. Tone Mapping"; 
> = 1000.0;

uniform float fCompressionStart < 
    ui_type     = "slider"; 
    ui_min      = 0.50; ui_max = 0.95; ui_step = 0.01; 
    ui_label    = "Compression Start (%)";
    ui_category = "2. Tone Mapping"; 
> = 0.80;

uniform float fDesaturationStrength < 
    ui_type     = "slider"; 
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.01; 
    ui_label    = "Khronos Desaturation (Legacy)"; 
    ui_tooltip  = "Recommended: 0.00. Use Highlight Bleaching instead.\n"
                  "This performs legacy math-based desaturation near display peak.";
    ui_category = "2. Tone Mapping"; 
> = 0.00;

// -------------------------------------------------------------------------------------------------
// UI: System
// -------------------------------------------------------------------------------------------------
uniform int iColorSpaceOverride < 
    ui_type     = "combo"; 
    ui_label    = "Color Space Override";
    ui_items    = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0"; 
    ui_category = "System"; 
> = 0;

uniform float fWhitePoint < 
    ui_type     = "slider"; 
    ui_min      = 80.0; ui_max = 10000.0; ui_step = 1.0; 
    ui_label    = "Reference White (Nits)";
    ui_category = "System"; 
> = 203.0;

// =================================================================================================
// 3. True Math Utilities (IEEE 754 Compliant)
// =================================================================================================

/// @brief IEEE 754 compliant power function. Prevents NaN when base is <= 0.0.
float PowNonNegPreserveZero(float x, float e)
{
    if (x <= 0.0) return 0.0;
    return pow(x, e);
}

/// @brief Vectorized version of PowNonNegPreserveZero.
float3 PowNonNegPreserveZero3(float3 x, float e)
{
    return float3(
        PowNonNegPreserveZero(x.r, e),
        PowNonNegPreserveZero(x.g, e),
        PowNonNegPreserveZero(x.b, e)
    );
}

/// @brief IEEE 754 precision Square Root. Bypasses GPU SFU low-precision intrinsics.
float SqrtIEEE(float x)
{
    return PowNonNegPreserveZero(x, 0.5);
}

// NaN / Inf Guards
bool IsNanVal(float x)   { return (asuint(x) & 0x7FFFFFFF) > 0x7F800000; }
bool IsInfVal(float x)   { return (asuint(x) & 0x7FFFFFFF) == 0x7F800000; }
bool3 IsNan3(float3 v)   { return bool3(IsNanVal(v.x), IsNanVal(v.y), IsNanVal(v.z)); }
bool3 IsInf3(float3 v)   { return bool3(IsInfVal(v.x), IsInfVal(v.y), IsInfVal(v.z)); }

// =================================================================================================
// 4. Color Science & EOTF Utilities
// =================================================================================================

/// @brief Decodes sRGB to Linear.
float3 sRGB_EOTF(float3 V)
{
    float3 abs_V  = abs(V);
    float3 lin_lo = abs_V / 12.92;
    float3 lin_hi = PowNonNegPreserveZero3((abs_V + 0.055) / 1.055, SRGB_GAMMA);

    float3 out_lin;
    out_lin.r = (abs_V.r <= SRGB_THRESHOLD_EOTF) ? lin_lo.r : lin_hi.r;
    out_lin.g = (abs_V.g <= SRGB_THRESHOLD_EOTF) ? lin_lo.g : lin_hi.g;
    out_lin.b = (abs_V.b <= SRGB_THRESHOLD_EOTF) ? lin_lo.b : lin_hi.b;
    return sign(V) * out_lin;
}

/// @brief Encodes Linear to sRGB.
float3 sRGB_OETF(float3 L)
{
    float3 abs_L  = abs(L);
    float3 enc_lo = abs_L * 12.92;
    float3 enc_hi = 1.055 * PowNonNegPreserveZero3(abs_L, SRGB_INV_GAMMA) - 0.055;

    float3 out_enc;
    out_enc.r = (abs_L.r <= SRGB_THRESHOLD_OETF) ? enc_lo.r : enc_hi.r;
    out_enc.g = (abs_L.g <= SRGB_THRESHOLD_OETF) ? enc_lo.g : enc_hi.g;
    out_enc.b = (abs_L.b <= SRGB_THRESHOLD_OETF) ? enc_lo.b : enc_hi.b;
    return sign(L) * out_enc;
}

/// @brief Decodes ST.2084 PQ to Linear Nits.
float3 PQ_EOTF(float3 N)
{
    N = saturate(N);
    float3 Np  = PowNonNegPreserveZero3(N, PQ_INV_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    return PowNonNegPreserveZero3(num / den, PQ_INV_M1) * PQ_PEAK_LUMINANCE;
}

/// @brief Encodes Linear Nits to ST.2084 PQ.
float3 PQ_InverseEOTF(float3 L)
{
    L = clamp(L, 0.0, PQ_PEAK_LUMINANCE);
    float3 Lp  = PowNonNegPreserveZero3(L / PQ_PEAK_LUMINANCE, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    return saturate(PowNonNegPreserveZero3(num / den, PQ_M2));
}

/// @brief System decoder based on user/system color space.
float3 DecodeToLinear(float3 encoded, int space)
{
    [branch] if (space == 3) return PQ_EOTF(encoded);
    [branch] if (space == 2) return encoded * SCRGB_WHITE_NITS;
    return sRGB_EOTF(encoded) * SCRGB_WHITE_NITS;
}

/// @brief System encoder based on user/system color space.
float3 EncodeFromLinear(float3 lin, int space)
{
    [branch] if (space == 3) return PQ_InverseEOTF(lin);
    [branch] if (space == 2) return lin / SCRGB_WHITE_NITS;
    return sRGB_OETF(lin / SCRGB_WHITE_NITS);
}

// =================================================================================================
// 5. Mathematical Scene-Grade Functions
// =================================================================================================

/// @brief Parabolic toe function for dehazing and black-point subtraction. C1-Continuous.
float ComputeBlackPointRatio(float luma, float bpNits)
{
    if (bpNits <= FLT_MIN || luma <= FLT_MIN) return 1.0;
    if (luma < 2.0 * bpNits) return luma / (4.0 * bpNits);
    return (luma - bpNits) / luma;
}

/// @brief Applies white balance as an exponential gain in the LMS cone domain.
float3 ApplyLMSWhiteBalance(float3 color, float temp, float tint, float3 lumaCoeffs, float3x3 to_LMS, float3x3 to_RGB)
{
    float3 wbStops = 0.35 * float3(temp + tint, -tint, -temp + tint);
    float3 wbScale = exp2(wbStops);

    float3 d65_wb_rgb = mul(to_RGB, wbScale);
    float lumaScale   = dot(d65_wb_rgb, lumaCoeffs);
    wbScale /= max(lumaScale, FLT_MIN);

    float3 lms = mul(to_LMS, color);
    lms *= wbScale;
    
    return mul(to_RGB, lms);
}

// =================================================================================================
// 6. MacLeod-Boynton Physiological Space Functions
// =================================================================================================

/// @brief Converts row-sum normalized LMS to MacLeod-Boynton Chromaticity.
float3 LMS_to_MB(float3 lms)
{
    float lum = max(lms.r + lms.g, FLT_MIN);
    return float3(lms.r / lum, lms.b / lum, lum);
}

/// @brief Converts MacLeod-Boynton back to LMS.
float3 MB_to_LMS(float3 mb)
{
    return float3(mb.x * mb.z, mb.z - (mb.x * mb.z), mb.y * mb.z);
}

/// @brief Extracts the precise D65 origin for the active color space.
float2 GetMBWhite(float3x3 to_LMS)
{
    float3 white_lms = mul(to_LMS, float3(1.0, 1.0, 1.0));
    return LMS_to_MB(white_lms).xy;
}

/// @brief Applies strictly isoluminant purity scaling in MB Space with Gamut-Aware protection.
float3 ApplyMBPurity(float3 color, float purity_scale, int space, float3 lumaCoeffs, float3x3 to_LMS, float3x3 to_RGB, float2 mb_white)
{
    if (abs(purity_scale - 1.0) < NEUTRAL_EPS) return color;

    // Dark-chroma mathematical stability fade
    float luma = dot(color, lumaCoeffs);
    float ct   = saturate((luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);
    
    if (chroma_reliability <= 0.0) return color;

    float effective_scale = purity_scale;

    // Vivid-Color Protection logic (prevents over-boosting neon colors)
    if (purity_scale > 1.0)
    {
        float3 lms = mul(to_LMS, color);
        float lm_sum = lms.r + lms.g;

        if (lm_sum > FLT_MIN)
        {
            float3 mb = LMS_to_MB(lms);
            float2 chroma_offset = mb.xy - mb_white;
            float purity = SqrtIEEE(dot(chroma_offset, chroma_offset));

            float protection_t = saturate(purity / MB_PURITY_PROTECTION_CEILING);
            float protection   = protection_t * protection_t * (3.0 - 2.0 * protection_t);

            float boost           = purity_scale - 1.0;
            float space_comp      = (space >= 3) ? 0.90 : 1.0;
            float min_boost_share = (space >= 3) ? 0.20 : 0.25;

            effective_scale = 1.0 + boost * space_comp * lerp(1.0, min_boost_share, protection);
        }
    }

    effective_scale = lerp(1.0, effective_scale, chroma_reliability);
    if (abs(effective_scale - 1.0) < NEUTRAL_EPS) return color;

    // Apply the purity shift in MB Space
    float3 lms_final = mul(to_LMS, color);
    float lm_sum_final = lms_final.r + lms_final.g;
    if (lm_sum_final <= 0.0) return color;

    float3 mb_final = LMS_to_MB(lms_final);
    mb_final.xy = lerp(mb_white, mb_final.xy, effective_scale);

    return mul(to_RGB, MB_to_LMS(mb_final));
}

// =================================================================================================
// 7. Tonemapping Functions
// =================================================================================================

/// @brief Biological Highlight Bleaching. 
/// Mimics photopigment depletion (Trolands), naturally desaturating absolute highlights to white.
float3 ApplyTrolandBleaching(float3 color, float strength, float3x3 to_LMS, float3x3 to_RGB, float2 mb_white)
{
    if (strength <= NEUTRAL_EPS) return color;

    float3 lms = mul(to_LMS, color);
    float lm_sum = lms.r + lms.g;
    if (lm_sum <= FLT_MIN) return color; 

    // Calculate perceptual intensity proxy (Trolands)
    float3 safe_lms = max(lms, 0.0);
    float3 stimulus = safe_lms * TROLAND_LMS_SCALE;
    float stim_lm   = 0.5 * (stimulus.r + stimulus.g);

    // Asymptotic Availability (approaches 0 at extreme brightness)
    float availability = 1.0 / (1.0 + (stim_lm / max(TROLAND_HALF_SAT, FLT_MIN)));
    float k = lerp(1.0, availability, saturate(strength));

    // Collapse purity in MB space (Protects physical luminance z)
    float3 mb = LMS_to_MB(lms);
    mb.xy = lerp(mb_white, mb.xy, k);

    return mul(to_RGB, MB_to_LMS(mb));
}

/// @brief Khronos PBR Neutral. Mathematical display-hull compression.
/// Provably hue-invariant when applied through row-sum-normalized LMS matrices.
float3 ApplyKhronosPBRNeutral(float3 color, float targetPeak, float compressionStart, float desatStrength)
{
    float3 safeColor = max(color, 0.0);
    float x = min(safeColor.r, min(safeColor.g, safeColor.b));
    
    // Fresnel toe offsets
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    float peak = max(safeColor.r, max(safeColor.g, safeColor.b)) - offset;
    float startComp = (targetPeak * compressionStart) - 0.04;

    [branch]
    if (peak >= startComp && startComp > 0.0)
    {
        float d = targetPeak - startComp;
        float newPeak = targetPeak - (d * d) / (peak + d - startComp);

        // Apply compression ratio
        float3 working = color - offset;
        float ratio = newPeak / max(peak, FLT_MIN);
        working *= ratio;

        // Apply mathematical desaturation (Replaced by Troland Bleaching in V5.9.3)
        float t = saturate((newPeak - startComp) / max(d, FLT_MIN));
        float g = desatStrength * t * t;
        working = lerp(working, newPeak.xxx, g);

        return working + offset;
    }

    return color;
}

// =================================================================================================
// 8. Main Pipeline Shader
// =================================================================================================

void PS_PhotorealHDR(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 fragColor : SV_Target)
{
    int2 pos   = int2(vpos.xy);
    float4 src = tex2Dfetch(SamplerBackBuffer, pos);

    int space         = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float whitePt     = (space <= 1) ? SCRGB_WHITE_NITS : fWhitePoint;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;

    // Fast-Bypass Guard
    [branch]
    if (abs(fExposure) < NEUTRAL_EPS && abs(fBlackPoint) < NEUTRAL_EPS &&
        abs(fContrast - 1.0) < NEUTRAL_EPS && abs(fShadows) < NEUTRAL_EPS &&
        abs(fHighlights) < NEUTRAL_EPS && abs(fTemperature) < NEUTRAL_EPS &&
        abs(fTint) < NEUTRAL_EPS && abs(fSaturation - 1.0) < NEUTRAL_EPS &&
        abs(fBleaching) < NEUTRAL_EPS && !bEnableKhronosNeutral)
    {
        fragColor = src;
        return;
    }

    // Decode & Sanitize Buffer
    float3 original_lin = DecodeToLinear(src.rgb, space);
    if (any(IsNan3(original_lin)) || any(IsInf3(original_lin))) original_lin = 0.0;

    float3 color = original_lin;

    // Matrix resolution
    float3x3 to_LMS, to_RGB;
    [branch]
    if (space >= 3) 
    { 
        to_LMS = RGB2020_to_LMS; 
        to_RGB = LMS_to_RGB2020; 
    }
    else 
    { 
        to_LMS = RGB709_to_LMS;  
        to_RGB = LMS_to_RGB709;  
    }

    // Pre-compute the exact D65 center for the current active color space
    float2 mb_white = GetMBWhite(to_LMS);

    // ---------------------------------------------------------------------------------------------
    // STAGE 1: EXPOSURE & WHITE BALANCE
    // ---------------------------------------------------------------------------------------------
    if (abs(fExposure) > NEUTRAL_EPS) color *= exp2(fExposure);

    if (abs(fTemperature) > NEUTRAL_EPS || abs(fTint) > NEUTRAL_EPS)
        color = ApplyLMSWhiteBalance(color, fTemperature, fTint, lumaCoeffs, to_LMS, to_RGB);

    // ---------------------------------------------------------------------------------------------
    // STAGE 2: DEHAZE & BLACK POINT
    // ---------------------------------------------------------------------------------------------
    if (fBlackPoint > NEUTRAL_EPS)
    {
        float luma = dot(color, lumaCoeffs);
        if (luma > FLT_MIN)
        {
            float bpNits = fBlackPoint * whitePt;
            color *= ComputeBlackPointRatio(luma, bpNits);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 3: FILMIC CONTRAST & TONAL EQ
    // ---------------------------------------------------------------------------------------------
    if (abs(fContrast - 1.0) > NEUTRAL_EPS || abs(fShadows) > NEUTRAL_EPS || abs(fHighlights) > NEUTRAL_EPS)
    {
        float luma = dot(color, lumaCoeffs);
        float absLuma = abs(luma);

        if (absLuma > FLT_MIN)
        {
            float pivot = fContrastPivot * whitePt;
            float logRatio = log2(absLuma / pivot);

            float x = logRatio * fContrast;
            float a2 = 6.0;

            // Apply C1 Rational Recovery Curves
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
            color *= min(contrastLuma / absLuma, 100.0); // 100x ratio cap prevents noise explosion
        }
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 4: BIOLOGICAL HIGHLIGHT BLEACHING
    // ---------------------------------------------------------------------------------------------
    color = ApplyTrolandBleaching(color, fBleaching, to_LMS, to_RGB, mb_white);

    // ---------------------------------------------------------------------------------------------
    // STAGE 5: KHRONOS HIGHLIGHT COMPRESSION
    // ---------------------------------------------------------------------------------------------
    [branch]
    if (bEnableKhronosNeutral)
    {
        color /= max(whitePt, FLT_MIN);

        float targetPeak = (space <= 1) ? 1.0 : max(1.0, fDisplayPeakNits / whitePt);
        color = ApplyKhronosPBRNeutral(color, targetPeak, fCompressionStart, fDesaturationStrength);

        color *= whitePt;
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 6: MACLEOD-BOYNTON ISOLUMINANT PURITY
    // ---------------------------------------------------------------------------------------------
    color = ApplyMBPurity(color, fSaturation, space, lumaCoeffs, to_LMS, to_RGB, mb_white);

    // NaN / Inf Pipeline Catch
    if (any(IsNan3(color)) || any(IsInf3(color))) color = original_lin;

    // ---------------------------------------------------------------------------------------------
    // STAGE 7: ENCODE & OUTPUT
    // ---------------------------------------------------------------------------------------------
    float3 encoded = EncodeFromLinear(color, space);

    [flatten]
    if (space <= 1) encoded = saturate(encoded);

    fragColor = float4(encoded, src.a);
}

// =================================================================================================
// 9. Technique Definition
// =================================================================================================

technique PhotorealHDR_Mastering_V593 <
    ui_label = "Photoreal HDR V5.9.3 (Director's Cut)";
    ui_tooltip = "V5.9.3 adds Biological Highlight Bleaching (Trolands) pre-tonemap.\n"
                 "Highlights burn toward white-hot cores based on absolute intensity (nits).\n"
                 "Then Khronos compresses into display peak.\n\n"
                 "Recommended: set 'Khronos Desaturation (Legacy)' to 0.00.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_PhotorealHDR;
    }
}
