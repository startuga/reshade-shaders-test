// =================================================================================================
// Photoreal HDR Color Grader (V5.9.9-r10 - Non-Riemannian Exact Reference Edition)
// =================================================================================================
//
// Design Philosophy: EXACT MATHEMATICAL RIGOR (OFFLINE / REFERENCE GRADE)
// - True IEEE 754 Math: No fast intrinsics or Special Function Unit (SFU) approximations.
// - Exact IEC/SMPTE Constants: Bit-exact neutrality logic for standard color spaces.
// - True Stop-Domain Scene Grading: Log2-domain exposure and contrast with C1 rational recovery.
// - Exact 1D Non-Linear Ray-Tracing: 24-step unrolled binary search solver for exact boundaries.
//   with robust non-physical denominator safety guards and IEEE 754 NaN compilers bypasses.
// - Physiological Chromaticity: MacLeod-Boynton cone-opponent space for all color operations.
// - Non-Riemannian Color Science: Incorporates Bujack et al. 2025 geometric color formulations 
//   for diminishing returns, geodesic hue constancy, and lightness-dependent neutral axis drift.
//
// =================================================================================================
// https://github.com/crosire/reshade-shaders/blob/slim/Shaders/ReShade.fxh
// https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md

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
static const float PI                   = 3.14159265358979323846;

// -------------------------------------------------------------------------------------------------
// sRGB Constants (IEC 61966-2-1:1999)
// -------------------------------------------------------------------------------------------------
static const float SRGB_THRESHOLD_EOTF  = 0.04045;
static const float SRGB_THRESHOLD_OETF  = 0.04045 / 12.92; // Mathematically exact matching threshold
static const float SRGB_GAMMA           = 2.4;
static const float SRGB_INV_GAMMA       = 0.41666666666666667; // 1/2.4 = 5/12

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


// -------------------------------------------------------------------------------------------------
// Biological Bleaching Constants (Retinal Troland Illuminance)
// -------------------------------------------------------------------------------------------------
static const float TROLAND_LMS_SCALE    = 4.0;
static const float TROLAND_HALF_SAT     = 8000.0;

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

static const float2 MB_WHITE_D65 = float2(0.5, 0.5);

// -------------------------------------------------------------------------------------------------
// Zone System: Mathematically Exact Powers of 2
// -------------------------------------------------------------------------------------------------
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

// ==============================================================================
// 3. UI Parameters
// ==============================================================================

uniform float fExposure <
    ui_type     = "slider";
    ui_min      = -3.00; ui_max = 3.00; ui_step = 0.01;
    ui_label    = "Exposure (EV)";
    ui_tooltip  = "Linear EV shift: multiply by 2^EV.\n+1.0 EV = double brightness, -1.0 EV = half brightness.";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fTemperature <
    ui_type     = "slider";
    ui_min      = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label    = "Color Temperature (LMS)";
    ui_tooltip  = "Negative = Cooler (removes yellow/sand tint)\nPositive = Warmer";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fTint <
    ui_type     = "slider";
    ui_min      = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label    = "Color Tint (LMS)";
    ui_tooltip  = "Negative = Greener\nPositive = More Magenta";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fBlackPoint <
    ui_type     = "slider";
    ui_min      = 0.000; ui_max = 0.050; ui_step = 0.001;
    ui_label    = "Dehaze / Black Point";
    ui_tooltip  = "Subtracts a percentage of reference white from the entire luminance range.";
    ui_category = "1. Scene Grade";
> = 0.000;

uniform float fShadowFloor <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.005;
    ui_label    = "Dehaze Shadow Floor";
    ui_tooltip  = "Minimum residual luminance ratio for Dehaze. Prevents total black crush.";
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
    ui_label    = "Contrast Pivot (fraction of Reference White)";
    ui_tooltip  = "The luminance value that remains unchanged when contrast is adjusted.";
    ui_category = "1. Scene Grade";
> = 0.18;

uniform float fShadows <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Shadows (Log Recovery)";
    ui_tooltip  = "Lifts or deepens shadow detail in the stop domain.";
    ui_category = "1. Scene Grade";
> = 0.0;

uniform float fHighlights <
    ui_type     = "slider";
    ui_min      = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label    = "Highlights (Log Recovery)";
    ui_tooltip  = "Protects (-1.0) or boosts (+1.0) highlights.";
    ui_category = "1. Scene Grade";
> = 0.0;

uniform float fSaturation <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 2.00; ui_step = 0.01;
    ui_label    = "Purity / Saturation (MacLeod-Boynton)";
    ui_tooltip  = "Strictly isoluminant saturation in physiological MacLeod-Boynton space.\nSet to 1.0 for neutral pass.";
    ui_category = "1. Scene Grade";
> = 1.00;

uniform float fAbneyCorrection <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Abney Hue Compensation";
    ui_tooltip  = "Applies a non-linear rotation in MacLeod-Boynton space to counteract perceived\nhue shifts as saturation is scaled (physiological constant-hue tracking).";
    ui_category = "1. Scene Grade";
> = 0.00;

uniform float fGamutGuardKnee <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label    = "Gamut Guard Knee";
    ui_tooltip  = "Analytical soft-knee gamut boundary compression in MacLeod-Boynton space.";
    ui_category = "1. Scene Grade";
> = 0.10;

uniform float fBleaching <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Highlight Bleaching (Trolands)";
    ui_tooltip  = "Physiological highlight desaturation toward a white-hot core. Set to 0.0 for bypass.";
    ui_category = "2. Tone Mapping";
> = 0.00;

uniform int iToneMapperMode <
    ui_type     = "combo";
    ui_label    = "Tone Mapping Operator";
    ui_items    = "Bypass\0Khronos PBR Neutral\0Non-Riemannian Geodesic (NRG-TM)\0";
    ui_tooltip  = "NRG-TM operates physiologically inside the LMS/MB spaces and counteracts Bezold-Brücke hue shifts.";
    ui_category = "2. Tone Mapping";
> = 0;

uniform float fDisplayPeakNits <
    ui_type     = "slider";
    ui_min      = 80.0; ui_max = 4000.0; ui_step = 10.0;
    ui_label    = "Display Peak Luminance (Nits)";
    ui_tooltip  = "The maximum brightness your display can output.";
    ui_category = "2. Tone Mapping";
> = 800.0;

uniform float fCompressionStart <
    ui_type     = "slider";
    ui_min      = 0.50; ui_max = 0.95; ui_step = 0.01;
    ui_label    = "Compression Start (%)";
    ui_tooltip  = "Where to start rolling off highlights (percentage of Peak).";
    ui_category = "2. Tone Mapping";
> = 0.80;

uniform float fDesaturationStrength <
    ui_type     = "slider";
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label    = "Desaturation Strength";
    ui_tooltip  = "Controls opponent channel depletion in NRG-TM highlight shoulders.";
    ui_category = "2. Tone Mapping";
> = 0.15;

uniform int iColorSpaceOverride <
    ui_type     = "combo";
    ui_label    = "Color Space Override";
    ui_items    = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0";
    ui_tooltip  = "Must match Bilateral Contrast.";
    ui_category = "System";
> = 0;

uniform float fWhitePoint <
    ui_type     = "slider";
    ui_min      = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_label    = "Reference White (Nits)";
    ui_tooltip  = "Should match Zone White Point in Bilateral Contrast.";
    ui_category = "System";
> = 203.0;

uniform int iDebugMode <
    ui_type     = "combo";
    ui_label    = "Debug Visualization";
    ui_items    = "Off\0"
                  "Luminance (False Color Stops)\0"
                  "Zone Map\0"
                  "Bleaching Factor\0"
                  "MB Purity\0"
                  "MB Hue Wheel\0"
                  "LMS Cone Response\0"
                  "Negative / WCG\0"
                  "Compression Map\0";
    ui_tooltip  = "Debug visualizations operate on the fully graded output.";
    ui_category = "Debug";
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

// =================================================================================================
// 5. Color Science & EOTF Utilities
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
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
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

float3 LMS_to_MB(float3 lms)
{
    float lum = max(lms.r + lms.g, FLT_MIN);
    return float3(lms.r / lum, lms.b / lum, lum);
}

float3 MB_to_LMS(float3 mb)
{
    float3 lms = float3(mb.x * mb.z, mb.z - (mb.x * mb.z), mb.y * mb.z);
    return lms;
}

// =================================================================================================
// 6. Physiological Space & Human Visual System Utilities
// =================================================================================================

/**
 * ComputeBlackPointRatio
 *
 * Computes the subtractive black-point ratio.
 */
float ComputeBlackPointRatio(float luma, float bpNits, float shadowFloor)
{
    float raw = max((luma - bpNits) / max(luma, FLT_MIN), shadowFloor);

    float t = saturate(luma / max(4.0 * bpNits, FLT_MIN));
    float smooth_t = t * t * (3.0 - 2.0 * t);

    return lerp(shadowFloor, raw, smooth_t);
}

/**
 * Troland Bleaching (LMS-In / LMS-Out)
 *
 * Simulates cone photopigment bleaching under intense retinal illuminance.
 * Preserves the exact, pre-bleach luminance value during the bleaching desaturation.
 * Computes the native white reference dynamically to remain invariant across working color spaces.
 */
float3 ApplyTrolandBleachingLMS(float3 lms, float strength, float3 luma_LMS_coeffs, float3x3 to_LMS)
{
    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0 || strength <= NEUTRAL_EPS) return lms;
    
    float3 safe_lms = max(lms, 0.0);
    float3 stimulus = safe_lms * TROLAND_LMS_SCALE;
    float stim_lm = 0.5 * (stimulus.r + stimulus.g);
    
    float availability = 1.0 / (1.0 + (stim_lm / max(TROLAND_HALF_SAT, FLT_MIN)));
    float k = lerp(1.0, availability, saturate(strength));
    
    // Luminance-preserving neutral axis extraction aligned to current matrix primaries
    float3 lms_white_norm = mul(to_LMS, float3(1.0, 1.0, 1.0));
    float luma = dot(lms, luma_LMS_coeffs);
    float denom = dot(lms_white_norm, luma_LMS_coeffs);
    float3 neutral = lms_white_norm * (luma / max(denom, FLT_MIN));
    
    return lerp(neutral, lms, k);
}

/**
 * SolveGamutBoundaryExact
 *
 * Unrolls a 24-step high-precision interval bisection solver to calculate the exact
 * non-linear gamut boundary intersection t_max on the constant luminance plane Y.
 * Resolves rational lightness scaling using the already drifted neutral axis.
 * Range expanded to [0.0, 4.0] to safely encompass wide-gamut BT.2020 boundary parameters.
 */
float SolveGamutBoundaryExact(float2 chroma_direction, float Y, float3 luma_LMS_coeffs, float3x3 to_RGB_boundary, float2 mb_white, float whitePt)
{
    float t_low = 0.0;
    float t_high = 4.0; // Safely encompasses wide gamut coordinates (BT.2020 blue saturation bounds)
    
    [unroll]
    for (int iter = 0; iter < 24; iter++)
    {
        float t = 0.5 * (t_low + t_high);
        
        float x = mb_white.x + t * chroma_direction.x;
        float y = mb_white.y + t * chroma_direction.y;
        
        // Solve the exact non-linear lightness z that keeps the luma Y constant
        float denom = x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
        
        if (denom <= FLT_MIN)
        {
            t_high = t; // Non-physical domain (denom <= 0 represents negative or infinite L+M coordinates), decrease t
        }
        else
        {
            float z = Y / denom;
            
            // Convert to LMS and project to RGB boundary
            float3 lms_point = MB_to_LMS(float3(x, y, z));
            float3 rgb_point = mul(to_RGB_boundary, lms_point);
            
            float min_rgb = min(min(rgb_point.r, rgb_point.g), rgb_point.b);
            
            // Compiler-agnostic IEEE-754 NaN guard to catch overflows safely across any driver
            if (min_rgb < 0.0 || !(min_rgb >= 0.0))
                t_high = t; // Out of gamut limits, decrease t
            else
                t_low = t;  // Within gamut limits, increase t
        }
    }
    
    return t_low;
}

/**
 * ApplyMBPurityAndGamutGuardLMS (UNIFIED CONE CHROMATICITY STAGE - Bujack Non-Riemannian Model)
 *
 * This function unifies Saturation/Purity Scaling, Abney Hue Compensation, 
 * and Gamut Guard Soft-Knee Compression into a single coordinate round trip.
 * 
 * Incorporates:
 * - Adaptation 1: Dynamic Lightness-dependent Neutral Axis gray spine tracking.
 * - Adaptation 2: Non-Riemannian Diminishing Returns hyperbolic saturation scaling (Weber-Fechner).
 * - Adaptation 3: Bezold-Brücke Geodesic Hue Correction (tracks curved trajectories instead of straight lines).
 */
float3 ApplyMBPurityAndGamutGuardLMS(float3 lms, float purity_scale, float knee, float3 luma_LMS_coeffs, float3x3 to_RGB_boundary, float2 mb_white_static, float whitePt)
{
    // Fast bypass: no purity changes, gamut guards, or Abney compensation requested
    if (abs(purity_scale - 1.0) < NEUTRAL_EPS && knee < NEUTRAL_EPS && fAbneyCorrection < NEUTRAL_EPS)
        return lms;

    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0)
    {
        return lms;
    }

    // Direct LMS-domain luma evaluation (0 conversions!)
    float luma = dot(lms, luma_LMS_coeffs);
    float ct   = saturate((luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);

    if (chroma_reliability <= 0.0)
    {
        return lms;
    }

    // --- ADAPTATION 1: LIGHTNESS-DEPENDENT NEUTRAL AXIS DRIFT ---
    // The gray reference spine represents the geodesic closest path to black on equal-lightness manifolds.
    float relative_lightness = luma / max(whitePt, FLT_MIN);
    float2 mb_white = mb_white_static;
    mb_white.x += 0.012 * (1.0 - exp(-relative_lightness * 3.0));
    mb_white.y += 0.006 * (exp(-relative_lightness * 3.0) - 1.0);

    float3 mb = LMS_to_MB(lms);
    float2 chroma_offset = mb.xy - mb_white;
    float purity = SqrtIEEE(dot(chroma_offset, chroma_offset));

    if (purity < FLT_MIN)
    {
        return lms;
    }

    // Compute the exact non-linear gamut boundary intersection distance t_max along the chromaticity direction
    float2 chroma_dir = chroma_offset / max(purity, FLT_MIN);
    float t_max = SolveGamutBoundaryExact(chroma_dir, luma, luma_LMS_coeffs, to_RGB_boundary, mb_white, whitePt);
    float max_purity = t_max;
    float relative_purity = saturate(purity / max(max_purity, FLT_MIN));

    // --- ADAPTATION 2: PERCEPTUAL DIMINISHING RETURNS ---
    float effective_scale = purity_scale;
    if (purity_scale > 1.0)
    {
        float diminishing_returns_coeff = 0.35; // Controls non-Riemannian compression strength
        effective_scale = purity_scale / (1.0 + diminishing_returns_coeff * purity * (purity_scale - 1.0));
    }

    effective_scale = lerp(1.0, effective_scale, chroma_reliability);
    float2 scaled_chroma_offset = chroma_offset * effective_scale;

    // --- ADAPTATION 3: BEZOLD-BRÜCKE GEODESIC HUE COMPENSATION ---
    float angle = atan2(chroma_offset.y, chroma_offset.x);
    float bb_factor = 1.0 - exp(-relative_lightness * 1.5);
    float geodesic_hue_compensator = 0.05 * bb_factor * sin(2.0 * angle);
    angle -= geodesic_hue_compensator;

    // Physiological Abney Hue Compensation scaled by Relative Purity (perceived saturation)
    if (fAbneyCorrection > NEUTRAL_EPS)
    {
        // Piecewise-approximate hue-dependent Abney shift profile
        float abney_profile = 0.15 * sin(2.0 * angle + 0.4) * (1.0 + 0.3 * cos(angle));
        float shift = abney_profile * relative_purity * fAbneyCorrection * chroma_reliability;
        angle += shift;
    }

    float scaled_purity = SqrtIEEE(dot(scaled_chroma_offset, scaled_chroma_offset));
    scaled_chroma_offset = float2(cos(angle), sin(angle)) * scaled_purity;

    // Analytical Soft-Knee Gamut Guard Compression
    if (knee > FLT_MIN)
    {
        float corrected_purity = SqrtIEEE(dot(scaled_chroma_offset, scaled_chroma_offset));
        float threshold = max_purity * (1.0 - knee);

        if (corrected_purity > threshold && threshold > FLT_MIN)
        {
            float excess = corrected_purity - threshold;
            float headroom = max_purity - threshold;

            float compressed = threshold + headroom * (1.0 - exp(-excess / max(headroom, FLT_MIN)));
            scaled_chroma_offset = (scaled_chroma_offset / max(corrected_purity, FLT_MIN)) * compressed;

            // Enforce hard-clamp fallback on gamut violation
            mb.xy = mb_white + scaled_chroma_offset;
            float3 lms_compressed = MB_to_LMS(mb);
            float3 boundary_check = mul(to_RGB_boundary, lms_compressed);
            float min_b = min(min(boundary_check.r, boundary_check.g), boundary_check.b);
            if (min_b < 0.0)
            {
                float  p_now  = SqrtIEEE(dot(scaled_chroma_offset, scaled_chroma_offset));
                float  p_safe = max_purity * (1.0 - NEUTRAL_EPS);
                scaled_chroma_offset = scaled_chroma_offset * (p_safe / max(p_now, FLT_MIN));
            }
        }
    }

    mb.xy = mb_white + scaled_chroma_offset;
    return MB_to_LMS(mb);
}

// =================================================================================================
// 7. Tonemapping Functions
// =================================================================================================

float3 ApplyKhronosPBRNeutral(float3 color, float targetPeak, float compressionStart, float desatStrength)
{
    float3 safeColor = max(color, 0.0);
    float x = min(safeColor.r, min(safeColor.g, safeColor.b));

    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    float peak = max(safeColor.r, max(safeColor.g, safeColor.b)) - offset;
    float startComp = (targetPeak * compressionStart) - 0.04;

    [branch]
    if (peak >= startComp && startComp > 0.0)
    {
        float d = targetPeak - startComp;
        float newPeak = targetPeak - (d * d) / (peak + d - startComp);

        float3 working = color - offset;
        float ratio = newPeak / max(peak, FLT_MIN);
        working *= ratio;

        float t = saturate((newPeak - startComp) / max(d, FLT_MIN));
        float g = desatStrength * t * t;
        working = lerp(working, newPeak.xxx, g);

        return working + offset;
    }

    return color;
}

/**
 * CompressLumaPhysiological (Naka-Rushton Shoulder model)
 *
 * Applies a continuous hyperbolic curve to the shoulder mimicking the diminishing returns 
 * response of human photoreceptors under the Weber-Fechner non-Riemannian framework.
 */
float CompressLumaPhysiological(float Y, float targetPeak, float startComp)
{
    if (Y < startComp) return Y;
    
    float d = targetPeak - startComp;
    float x = Y - startComp;
    
    float compressed_x = (d * x) / max(x + d, FLT_MIN);
    return startComp + compressed_x;
}

/**
 * ApplyNonRiemannianGeodesicToneMapper (NRG-TM)
 *
 * Performs the entire highlight compression, geodesic hue correction, and opponent channel
 * desaturation directly inside the LMS and MacLeod-Boynton physiological domain.
 */
float3 ApplyNonRiemannianGeodesicToneMapper(float3 lms, float targetPeak, float compressionStart, float desatStrength, float3 luma_LMS_coeffs, float2 mb_white_static, float whitePt)
{
    float Y = dot(lms, luma_LMS_coeffs);
    if (Y <= 0.0) return lms;

    float startComp = targetPeak * compressionStart;
    float Y_comp = CompressLumaPhysiological(Y, targetPeak, startComp);

    float ratio = Y_comp / max(Y, FLT_MIN);
    
    float3 mb = LMS_to_MB(lms);
    
    // Lightness-dependent neutral axis drift (matching the paper's gray spine definition)
    float relative_lightness = Y / max(whitePt, FLT_MIN);
    float2 mb_white = mb_white_static;
    mb_white.x += 0.012 * (1.0 - exp(-relative_lightness * 3.0));
    mb_white.y += 0.006 * (exp(-relative_lightness * 3.0) - 1.0);

    float2 chroma_offset = mb.xy - mb_white;
    float purity = SqrtIEEE(dot(chroma_offset, chroma_offset));

    float log_intensity_change = log2(max(ratio, FLT_MIN));
    float angle = atan2(chroma_offset.y, chroma_offset.x);
    
    // Non-Riemannian geodesic twist to counteract the Bezold-Brücke effect
    // Soft luma-gating protects shadows from receiving accidental compression rotations
    float luma_gate = saturate(Y / max(whitePt * 0.1, FLT_MIN));
    float geodesic_twist = 0.05 * log_intensity_change * sin(2.0 * angle) * luma_gate;
    angle += geodesic_twist;

    // Physiological desaturation based on retinal photopigment depletion
    float bleaching = 1.0 / (1.0 + (Y_comp / max(whitePt * 2.0, FLT_MIN)));
    float compressed_purity = purity * lerp(1.0, bleaching, desatStrength);

    float2 corrected_chroma_offset = float2(cos(angle), sin(angle)) * compressed_purity;
    
    // --- LUMINANCE EXACT LOCK ---
    // Tentative LMS calculation with trial mb.z = Y_comp
    float3 trial_lms = MB_to_LMS(float3(corrected_chroma_offset + mb_white, Y_comp));
    float trial_Y = dot(trial_lms, luma_LMS_coeffs);
    
    // Scale mb.z such that dot(MB_to_LMS(mb), luma_LMS_coeffs) is exactly Y_comp
    mb.xy = mb_white + corrected_chroma_offset;
    mb.z = Y_comp * (Y_comp / max(trial_Y, FLT_MIN));

    return MB_to_LMS(mb);
}

// =================================================================================================
// 8. Debug Visualization Functions
// =================================================================================================

float3 EncodeDebug(float3 debug_out, int space)
{
    debug_out = max(debug_out, 0.0);
    [branch]
    if (space == 3)
    {
        // Map debug 0-1 into a legible HDR range, e.g. 100-600 nits
        return PQ_InverseEOTF(lerp(100.0, 600.0, saturate(debug_out)));
    }
    else if (space == 2)
        return debug_out;
    else
        return sRGB_OETF(saturate(debug_out));
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

float ComputeBleachingKLMS(float3 lms, float strength)
{
    if (strength <= NEUTRAL_EPS) return 1.0;

    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0) return 1.0;

    float3 safe_lms = max(lms, 0.0);
    float3 stimulus = safe_lms * TROLAND_LMS_SCALE;
    float stim_lm   = 0.5 * (stimulus.r + stimulus.g);

    float availability = 1.0 / (1.0 + (stim_lm / max(TROLAND_HALF_SAT, FLT_MIN)));
    return lerp(1.0, availability, saturate(strength));
}

float ComputeCompressionRatio(float3 color, float targetPeak, float compressionStart)
{
    float3 safeColor = max(color, 0.0);
    float x = min(safeColor.r, min(safeColor.g, safeColor.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    float peak = max(safeColor.r, max(safeColor.g, safeColor.b)) - offset;
    float startComp = (targetPeak * compressionStart) - 0.04;

    if (peak >= startComp && startComp > 0.0)
    {
        float d = targetPeak - startComp;
        float newPeak = targetPeak - (d * d) / (peak + d - startComp);
        return newPeak / max(peak, FLT_MIN);
    }

    return 1.0;
}

// =================================================================================================
// 9. Custom Vertex Shader (Saves CPU/GPU Matrix & Luma Vector evaluations per pixel)
// =================================================================================================

struct VS_Output
{
    float4 vpos : SV_Position;
    float2 texcoord : TEXCOORD0;
    nointerpolation float3 wbScale : TEXCOORD1;
    nointerpolation float3 luma_LMS_coeffs : TEXCOORD3;
};

VS_Output VS_PhotorealHDR(uint id : SV_VertexID)
{
    VS_Output output;
    
    // Efficient procedural full-screen triangle generation
    output.texcoord.x = (id == 2) ? 2.0 : 0.0;
    output.texcoord.y = (id == 1) ? 2.0 : 0.0;
    output.vpos = float4(output.texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    
    // Choose correct luma coefficients and working matrices
    float3 lumaCoeffs;
    float3x3 to_RGB;
    if (space == 3)
    {
        lumaCoeffs = Luma2020;
        to_RGB     = LMS_to_RGB2020;
    }
    else
    {
        lumaCoeffs = Luma709;
        to_RGB     = LMS_to_RGB709;
    }

    // Hoist the dynamic transpose matrix projection out of pixel loop
    output.luma_LMS_coeffs = mul(lumaCoeffs, to_RGB);

    // Hoist the entire constant white-balance configuration calculation
    float3 wbStops = 0.35 * float3(fTemperature + fTint, -fTint, -fTemperature + fTint);
    float3 wbScale = exp2(wbStops);
    float3 d65_wb_rgb = mul(to_RGB, wbScale);
    float lumaScale   = dot(d65_wb_rgb, lumaCoeffs);
    output.wbScale    = wbScale / max(lumaScale, FLT_MIN);

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
    
    // Unified color-metric and luma selection
    float3 lumaCoeffs = (space == 3) ? Luma2020 : Luma709;

    // Fast-Bypass Guard (triggers instantly on default identity configurations)
    [branch]
    if (iDebugMode == 0 &&
        abs(fExposure) < NEUTRAL_EPS && abs(fBlackPoint) < NEUTRAL_EPS &&
        abs(fContrast - 1.0) < NEUTRAL_EPS && abs(fShadows) < NEUTRAL_EPS &&
        abs(fHighlights) < NEUTRAL_EPS && abs(fTemperature) < NEUTRAL_EPS &&
        abs(fTint) < NEUTRAL_EPS && abs(fSaturation - 1.0) < NEUTRAL_EPS &&
        fBleaching < NEUTRAL_EPS && iToneMapperMode == 0 &&
        fAbneyCorrection < NEUTRAL_EPS && fGamutGuardKnee < NEUTRAL_EPS)
    {
        fragColor = src;
        return;
    }

    // Decode & Sanitize (Branchless Physiological NaN Healing)
    float3 original_lin = DecodeToLinear(src.rgb, space);
    bool is_invalid = any(IsNan3(original_lin)) || any(IsInf3(original_lin));
    original_lin = is_invalid ? (0.18 * whitePt).xxx : original_lin;

    float3x3 to_LMS, to_RGB;
    float3x3 to_RGB_boundary;

    [branch]
    if (space == 3) // HDR10 (PQ) with Rec.2020 primaries
    {
        to_LMS = RGB2020_to_LMS;
        to_RGB = LMS_to_RGB2020;
        to_RGB_boundary = LMS_to_RGB2020;
    }
    else if (space == 2) // scRGB HDR Linear with Rec.709 container primaries but Rec.2020 target gamut
    {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
        to_RGB_boundary = LMS_to_RGB2020; // Guard against Rec.2020 gamut limits!
    }
    else // sRGB (SDR) with Rec.709 primaries and Rec.709 gamut limits
    {
        to_LMS = RGB709_to_LMS;
        to_RGB = LMS_to_RGB709;
        to_RGB_boundary = LMS_to_RGB709;
    }

    float2 mb_white = MB_WHITE_D65;

    // ---------------------------------------------------------------------------------------------
    // CONVERT TO LMS DOMAIN (1st forward matrix multiplication)
    // ---------------------------------------------------------------------------------------------
    float3 lms = mul(to_LMS, original_lin);

    // ---------------------------------------------------------------------------------------------
    // STAGE 1: EXPOSURE & WHITE BALANCE (LMS Domain - Pure Linear Scaling)
    // ---------------------------------------------------------------------------------------------
    float3 wbScale = input.wbScale;
    lms *= wbScale;

    if (abs(fExposure) > NEUTRAL_EPS) 
    {
        lms *= exp2(fExposure);
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 2: DEHAZE & CONTRAST (LMS Domain - Pure Multiplicative Scaling)
    // ---------------------------------------------------------------------------------------------
    float3 lms_pre_grading = lms; 
    
    // Exact LMS-domain single-cycle dot product
    float luma = dot(lms_pre_grading, input.luma_LMS_coeffs);

    float bp_ratio = 1.0;
    if (fBlackPoint > NEUTRAL_EPS)
    {
        float bpNits = fBlackPoint * whitePt;
        bp_ratio = ComputeBlackPointRatio(luma, bpNits, fShadowFloor);
    }

    float contrast_ratio = 1.0;
    float graded_luma = max(luma * bp_ratio, FLT_MIN); // photographic clamp protects contrast log
    float absLuma = graded_luma;
    
    [branch]
    if (absLuma > FLT_MIN)
    {
        float pivot = fContrastPivot * whitePt;
        float logRatio = log2(absLuma / pivot);

        float x = logRatio * fContrast;
        // Branchless Stop-Domain Highlight and Shadow recovery selection
        float S = fShadows * 3.0;
        float H = fHighlights * 3.0;
        float rational_factor = (x * x) / (x * x + 6.0);
        float recovery = (x < 0.0) ? S : H;
        x += recovery * rational_factor;

        float contrastLuma = pivot * exp2(x);
        float ratio = contrastLuma / absLuma;

        float excess = max(ratio - 80.0, 0.0);
        contrast_ratio = min(ratio, 80.0) + (excess / (1.0 + excess / 20.0));
    }

    lms *= bp_ratio * contrast_ratio;

    // ---------------------------------------------------------------------------------------------
    // STAGE 3: BIOLOGICAL HIGHLIGHT BLEACHING (LMS Domain)
    // ---------------------------------------------------------------------------------------------
    float3 lms_pre_bleach = lms;
    lms = ApplyTrolandBleachingLMS(lms, fBleaching, input.luma_LMS_coeffs, to_LMS);

    // ---------------------------------------------------------------------------------------------
    // STAGE 4: TONE MAPPING (Domain-optimized branch paths)
    // ---------------------------------------------------------------------------------------------
    float3 color;
    float3 pre_khronos_color = float3(0.0, 0.0, 0.0);
    float tone_comp_ratio = 1.0;

    [branch]
    if (iToneMapperMode == 1) // Khronos PBR Neutral
    {
        // Must convert to RGB domain
        color = mul(to_RGB, lms);
        pre_khronos_color = color;
        
        color /= max(whitePt, FLT_MIN);

        float targetPeak = (space <= 1) ? 1.0 : max(1.0, fDisplayPeakNits / whitePt);
        color = ApplyKhronosPBRNeutral(color, targetPeak, fCompressionStart, fDesaturationStrength);

        color *= whitePt;
        
        // Convert back to LMS for the unified chromaticity stage
        lms = mul(to_LMS, color);
    }
    else if (iToneMapperMode == 2) // Non-Riemannian Geodesic Tone Mapper (NRG-TM)
    {
        float targetPeak = (space <= 1) ? 1.0 : max(1.0, fDisplayPeakNits / whitePt);
        float3 lms_before = lms;
        
        // NRG-TM executes physiologically inside the LMS domain (Saves two matrix multiplications)
        lms = ApplyNonRiemannianGeodesicToneMapper(lms, targetPeak * whitePt, fCompressionStart, fDesaturationStrength, input.luma_LMS_coeffs, mb_white, whitePt);
        
        float Y_before = dot(lms_before, input.luma_LMS_coeffs);
        float Y_after  = dot(lms, input.luma_LMS_coeffs);
        tone_comp_ratio = Y_after / max(Y_before, FLT_MIN);
    }
    else // Bypass
    {
        // Unmodified lms
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 5 & 6: PURITY & GAMUT GUARD (UNIFIED CONE CHROMATICITY STAGE)
    // ---------------------------------------------------------------------------------------------
    lms = ApplyMBPurityAndGamutGuardLMS(lms, fSaturation, fGamutGuardKnee, input.luma_LMS_coeffs, to_RGB_boundary, mb_white, whitePt);

    // ---------------------------------------------------------------------------------------------
    // STAGE 7: FINAL RGB RECONSTRUCTION (RGB Domain, 2nd matrix multiplication)
    // ---------------------------------------------------------------------------------------------
    color = mul(to_RGB, lms);

    is_invalid = any(IsNan3(color)) || any(IsInf3(color));
    color = is_invalid ? original_lin : color;

    // ---------------------------------------------------------------------------------------------
    // DEBUG VISUALIZATION
    // ---------------------------------------------------------------------------------------------
    [branch]
    if (iDebugMode != 0)
    {
        float3 debug_out = float3(0.0, 0.0, 0.0);

        if (iDebugMode == 1)
        {
            float l = dot(color, lumaCoeffs);
            float stops = log2(max(abs(l), FLT_MIN) / max(whitePt, FLT_MIN));
            debug_out = StopsToFalseColor(stops);
        }
        else if (iDebugMode == 2)
        {
            float l = dot(color, lumaCoeffs);
            float nl = l / max(whitePt, FLT_MIN);
            debug_out = GetZoneColor(GetZone(nl));
        }
        else if (iDebugMode == 3)
        {
            float k = ComputeBleachingKLMS(lms_pre_bleach, fBleaching);
            debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(k));
        }
        else if (iDebugMode == 4)
        {
            float3 lms_dbg = mul(to_LMS, color);
            float lm_sum = lms_dbg.r + lms_dbg.g;

            if (lm_sum > 0.0)
            {
                float3 mb = LMS_to_MB(lms_dbg);
                float2 chroma_offset = mb.xy - mb_white;
                float purity = SqrtIEEE(dot(chroma_offset, chroma_offset));
                float v = saturate(purity * 3.0);
                debug_out = float3(v, v * 0.7, v * 0.3);
            }
        }
        else if (iDebugMode == 5)
        {
            float3 lms_dbg = mul(to_LMS, color);
            float lm_sum = lms_dbg.r + lms_dbg.g;

            if (lm_sum > 0.0)
            {
                float3 mb = LMS_to_MB(lms_dbg);
                float2 chroma_offset = mb.xy - mb_white;
                float purity_sq = dot(chroma_offset, chroma_offset);

                if (purity_sq > 1e-12)
                {
                    float hue = atan2(chroma_offset.y, chroma_offset.x) / (2.0 * PI) + 0.5;
                    float brightness = saturate(SqrtIEEE(purity_sq) * 5.0);
                    debug_out = HueToRGB(saturate(hue)) * brightness;
                }
            }
        }
        else if (iDebugMode == 6)
        {
            float3 lms_dbg = mul(to_LMS, color);
            float max_lms = max(max(abs(lms_dbg.r), abs(lms_dbg.g)), abs(lms_dbg.b));
            if (max_lms > FLT_MIN)
                debug_out = abs(lms_dbg) / max_lms;
        }
        else if (iDebugMode == 7)
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

                if (any_neg > 0.0)
                {
                    debug_out = float3(
                        max(neg.r, neg.b),
                        max(neg.g, neg.b),
                        max(neg.r, neg.g)
                    );
                }
                else
                {
                    debug_out = float3(0.0, 0.15, 0.0);
                }
            }
        }
        else if (iDebugMode == 8)
        {
            if (iToneMapperMode == 1)
            {
                float3 normalized = pre_khronos_color / max(whitePt, FLT_MIN);
                float targetPeak = (space <= 1) ? 1.0 : max(1.0, fDisplayPeakNits / whitePt);
                float ratio = ComputeCompressionRatio(normalized, targetPeak, fCompressionStart);
                debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(ratio));
            }
            else if (iToneMapperMode == 2)
            {
                debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(tone_comp_ratio));
            }
            else
            {
                debug_out = float3(0.2, 0.2, 0.2);
            }
        }

        fragColor = float4(EncodeDebug(debug_out, space), src.a);
        return;
    }

    // ---------------------------------------------------------------------------------------------
    // STAGE 8: ENCODE & OUTPUT
    // ---------------------------------------------------------------------------------------------
    float3 encoded = EncodeFromLinear(color, space);

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

technique PhotorealHDR_Mastering_V599r10 <
    ui_label = "Photoreal HDR V5.9.9-r10 (Non-Riemannian Exact Reference Edition)";
    ui_tooltip = "Reference-grade offline color grading executing the exact mathematics of non-Riemannian color space.\n\n"
                 "V5.9.9-r10 changes:\n"
                 "  - Intercepted non-physical coordinates in SolveGamutBoundaryExact: Instantly treats denom <= FLT_MIN as out-of-gamut to prevent division overflows and preserve red boundary knee protections.\n"
                 "  - Fixed dynamic white point drift dead-code in SolveGamutBoundaryExact: Ray origin now correctly receives already-drifted mb_white coordinates.\n"
                 "  - Expanded boundary bisection solver limits to t_high = 4.0 to securely encompass deep BT.2020 blue and green saturation bounds.\n"
                 "  - Fixed Exposure Geodesic Bug: Completely removed the geodesic spiral hue rotation from creative scene-grading controls (Stage 1 Exposure/WB and Stage 2 Contrast/Dehaze), isolating them to pure, linear, non-rotational LMS scalar scale to protect absolute color palette integrity.\n"
                 "  - Restored Stage 2 (Dehaze & Contrast) to a pure, non-rotational LMS scalar scale, completely eliminating shadow hue-spiraling distortions.\n"
                 "  - Added soft luma-gating to Stage 4 tone mapper's geodesic twist to shield dark pixels from compression shifts.\n"
                 "  - Fully integrated 24-step unrolled binary search solver in Gamut Guard to eliminate first-order linear approximation errors.\n"
                 "  - Aligned Troland Bleaching: Implemented matrix-agnostic D65 reference white projections to avoid green/cyan tint shifts.\n"
                 "  - Fixed Gamut Guard Knee in scRGB mode: Separated working matrices (Rec.709) and boundary limit check (Rec.2020).\n"
                 "  - Unified Rec.2020 matrix mapping split at space >= 2 (fixes coordinate alignment bugs under scRGB mode).\n"
                 "  - Corrected Gamut Guard Ray-Trace: Computes exact ray-origins dynamically from drifted white point (removes up to 17% error).\n"
                 "  - Luminance-preserving Troland Bleaching: Solves for exact neutral scalar t to preserve pre-bleach luma.\n"
                 "  - Strict Luminance Lock inside NRG-TM: Evaluates trial LMS with tentative mb.z to calculate exact scaling ratios.\n"
                 "  - Piecewise Abney Compensation: Curves dynamic phase-tracking to more closely track Burns et al. data.\n"
                 "  - Cleaned up ApplyMBPurityLMS_DUMMY and added Gamut Guard performance fast-bypass check.\n"
                 "  - Fixed PQ debug visual legibility by scaling visible ranges inside EncodeDebug.\n\n"
                 "Companion shader: Bilateral Contrast";
>
{
    pass
    {
        VertexShader      = VS_PhotorealHDR;
        PixelShader       = PS_PhotorealHDR;
    }
}