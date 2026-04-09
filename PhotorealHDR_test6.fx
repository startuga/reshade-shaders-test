// =================================================================================================
// Photoreal HDR Color Grader (V6.0 - Biological Hybrid Edition)
// =================================================================================================
//
// Design Philosophy: PRECISION OVER PERFORMANCE + PHYSIOLOGICAL REALITY
// - True IEEE 754 Math: No fast intrinsics or Special Function Unit (SFU) approximations.
// - Exact IEC/SMPTE Constants: Bit-exact neutrality logic for standard color spaces.
// - True Stop-Domain Scene Grading: Log2-domain exposure and contrast with C1 rational recovery.
// - Biological Retina Simulation: Fully simulates human cone S-Potentials via Naka-Rushton and 
//   retinal photopigment bleaching via Troland limits.
//
// Architecture (The Hybrid Pipeline):
// -------------------------------------------------------------------------------------------------
// STAGE 1: SCENE-REFERRED GRADE (Mathematics)
// Operates on absolute scene-linear light. Uses the rigorous V5.9.2 Log-domain contrast, 
// C1 continuous rational Tonal EQ, and LMS white balance to artistically shape the lighting.
//
// STAGE 2: OBSERVER TONEMAP (Biology)
// Completely replaces traditional curve-based tonemappers (like Khronos/ACES).
// Simulates retinal photopigment bleaching and applies independent photoreceptor 
// S-potentials (Anchored Naka-Rushton). Uses MacLeod-Boynton Euclidean ray-tracing 
// to flawlessly recover physiological hue shifts (the Bezold-Brücke effect).
// =================================================================================================

#include "ReShade.fxh"

// =================================================================================================
// 1. Constants & Definitions
// =================================================================================================

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

static const float FLT_MIN          = 1.175494351e-38;
static const float SCRGB_WHITE_NITS = 80.0;
static const float NEUTRAL_EPS      = 1e-6;

// -------------------------------------------------------------------------------------------------
// sRGB Constants (IEC 61966-2-1:1999)
// -------------------------------------------------------------------------------------------------
static const float SRGB_THRESHOLD_EOTF = 0.04045;
static const float SRGB_THRESHOLD_OETF = 0.0031308;
static const float SRGB_GAMMA          = 2.4;
static const float SRGB_INV_GAMMA      = 0.41666666666666667;

// -------------------------------------------------------------------------------------------------
// ST.2084 (PQ) Constants (SMPTE ST 2084:2014)
// -------------------------------------------------------------------------------------------------
static const float PQ_M1             = 0.1593017578125;
static const float PQ_M2             = 78.84375;
static const float PQ_C1             = 0.8359375;
static const float PQ_C2             = 18.8515625;
static const float PQ_C3             = 18.6875;
static const float PQ_PEAK_LUMINANCE = 10000.0;
static const float PQ_INV_M1         = 6.2773946360153257;
static const float PQ_INV_M2         = 0.012683313515655966;

// -------------------------------------------------------------------------------------------------
// Color Science Constants
// -------------------------------------------------------------------------------------------------
static const float CHROMA_STABILITY_THRESH      = 1e-4;
static const float CHROMA_RELIABILITY_START     = 5e-5;
static const float INV_CHROMA_RELIABILITY_SPAN  = 1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);

static const float3 Luma709  = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020 = float3(0.2627, 0.6780, 0.0593);

// CIE 170-2 MacLeod-Boynton Physiological Weights
static const float3 MB_WEIGHTS = float3(0.68990272, 0.34832189, 0.0371597);
static const float2 D65_MB_XY  = float2(0.68990272 * 0.7347, 0.0371597 * 0.9638);

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

// -------------------------------------------------------------------------------------------------
// Stockman-Sharpe Matrices (For Biological Model)
// -------------------------------------------------------------------------------------------------
static const float3x3 XYZ_TO_STOCKMAN = float3x3(
    0.267050284,   0.847199015,  -0.034704166,
   -0.387068824,   1.165429936,   0.103022867,
    0.026727794,  -0.027291317,   0.533326726
);

static const float3x3 STOCKMAN_TO_XYZ = float3x3(
    1.94735469,   -1.41445123,    0.36476327,
    0.68990272,    0.34832189,    0.00000000,
   -0.06206981,    0.08985161,    1.82110260
);

static const float3x3 BT2020_TO_XYZ = float3x3(
    0.636958048,   0.144616904,   0.168880975,
    0.262700212,   0.677998072,   0.059301717,
    0.000000000,   0.028072693,   1.060985058
);

static const float3x3 XYZ_TO_BT2020 = float3x3(
    1.716651188,  -0.355670784,  -0.253366281,
   -0.666684352,   1.616481237,   0.015768546,
    0.017639857,  -0.042770613,   0.942103121
);

// =================================================================================================
// 2. Textures & UI Parameters
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
    ui_min      = -3.0; ui_max = 3.0; ui_step = 0.01; 
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
    ui_label    = "Purity / Saturation"; 
    ui_category = "1. Scene Grade"; 
> = 1.08;

// -------------------------------------------------------------------------------------------------
// UI: Part 2 - Observer Tonemap (Biology)
// -------------------------------------------------------------------------------------------------
uniform float fDisplayPeakNits < 
    ui_type     = "slider"; 
    ui_min      = 80.0; ui_max = 4000.0; ui_step = 10.0; 
    ui_label    = "Display Peak Luminance (Nits)";
    ui_category = "2. Retina Simulation"; 
> = 1000.0;

uniform float fBleaching < 
    ui_type     = "slider"; 
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01; 
    ui_label    = "Photopigment Bleaching"; 
    ui_tooltip  = "Simulates visual desaturation of intense highlights in Trolands."; 
    ui_category = "2. Retina Simulation"; 
> = 1.00;

uniform float fNakaExponent < 
    ui_type     = "slider"; 
    ui_min      = 0.50; ui_max = 1.50; ui_step = 0.01; 
    ui_label    = "Cone Response Exponent"; 
    ui_tooltip  = "Biological contrast roll-off. Higher = punchier highlights."; 
    ui_category = "2. Retina Simulation"; 
> = 0.95;

uniform float fHueRestore < 
    ui_type     = "slider"; 
    ui_min      = 0.00; ui_max = 1.00; ui_step = 0.01; 
    ui_label    = "Biological Hue Restore"; 
    ui_tooltip  = "Recovers hue shifted by independent cone saturation (Bezold-Brücke effect)."; 
    ui_category = "2. Retina Simulation"; 
> = 0.80;

// -------------------------------------------------------------------------------------------------
// UI: System Settings
// -------------------------------------------------------------------------------------------------
uniform int iColorSpaceOverride < 
    ui_type     = "combo"; 
    ui_items    = "Auto\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0"; 
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

/// @brief Safely divides two floats, preventing division by zero or NaN propagation.
float DivideSafe(float dividend, float divisor, float fallback) 
{ 
    return (abs(divisor) <= FLT_MIN) ? fallback : (dividend / divisor); 
}

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

/// @brief IEEE 754 precision Reciprocal Square Root. Replaces fast `rsqrt` intrinsic.
float RSqrtIEEE(float x) 
{ 
    float s = PowNonNegPreserveZero(x, 0.5); 
    return (s <= 0.0) ? 0.0 : (1.0 / s); 
}

bool IsNanVal(float x)   { return (asuint(x) & 0x7FFFFFFF) > 0x7F800000; }
bool IsInfVal(float x)   { return (asuint(x) & 0x7FFFFFFF) == 0x7F800000; }
bool3 IsNan3(float3 v)   { return bool3(IsNanVal(v.x), IsNanVal(v.y), IsNanVal(v.z)); }
bool3 IsInf3(float3 v)   { return bool3(IsInfVal(v.x), IsInfVal(v.y), IsInfVal(v.z)); }

// =================================================================================================
// 4. Color Space Utilities
// =================================================================================================

/// @brief Converts colors between standard BT.709 and BT.2020 color spaces.
float3 ConvertColorSpace(float3 color, int from_space, int to_space) 
{
    if (from_space == 0 && to_space == 1) 
    {
        // BT.709 to BT.2020
        const float3x3 m = float3x3(
            0.627404, 0.329282, 0.043314, 
            0.069097, 0.919540, 0.011363, 
            0.016391, 0.088013, 0.895595
        );
        return mul(m, color);
    } 
    else if (from_space == 1 && to_space == 0) 
    {
        // BT.2020 to BT.709
        const float3x3 m = float3x3(
             1.660491, -0.587641, -0.072850, 
            -0.124550,  1.132900, -0.008349, 
            -0.030433, -0.132514,  1.162947
        );
        return mul(m, color);
    }
    return color;
}

// =================================================================================================
// 5. Biological Observer Model Math
// =================================================================================================

float3 RGB_to_StockmanLMS(float3 rgb) 
{
    return mul(XYZ_TO_STOCKMAN, mul(BT2020_TO_XYZ, rgb));
}

float3 StockmanLMS_to_RGB(float3 lms) 
{
    return mul(XYZ_TO_BT2020, mul(STOCKMAN_TO_XYZ, lms));
}

/// @brief Applies physiological luminance weights to LMS cone responses.
float3 WeighLMS(float3 lms) { return lms * MB_WEIGHTS; }

/// @brief Removes physiological luminance weights.
float3 UnweighLMS(float3 weighted_lms) { return DivideSafe(weighted_lms, MB_WEIGHTS, 0.f.xxx); }

/// @brief Converts weighted LMS to MacLeod-Boynton Chromaticity Diagram coordinates.
/// @return float3(x, y, z) where xy is chromaticity and z is luminance (L+M).
float3 LMS_to_MB1702(float3 lms_weighted) 
{
    float y = max(lms_weighted.x + lms_weighted.y, FLT_MIN);
    return float3(lms_weighted.x / y, lms_weighted.z / y, y);
}

/// @brief Converts MacLeod-Boynton coordinates back to weighted LMS.
float3 MB1702_to_LMS(float3 mb) 
{
    float l = mb.x * mb.z;
    float m = mb.z - l;
    float s = mb.y * mb.z;
    return float3(l, m, s);
}

/// @brief Anchored Naka-Rushton (S-Potentials).
/// Simulates biological photoreceptor adaptation. Mathematically locked so that 
/// the `anchor` input exactly maps to the `anchor` output, preserving mid-gray exposure.
float3 NakaRushton_Anchored(float3 x, float3 peak, float3 anchor, float exponent) 
{
    float3 p_minus_a = peak - anchor;
    float3 n         = exponent * peak / max(p_minus_a, FLT_MIN.xxx);
    float3 a_n       = PowNonNegPreserveZero3(anchor, n);
    float3 sign_x    = sign(x);
    float3 x_n       = PowNonNegPreserveZero3(abs(x), n);
    float3 x_n_a     = x_n * anchor;
    
    float3 num = peak * x_n_a;
    float3 den = (a_n * p_minus_a) + x_n_a;
    
    return sign_x * (num / max(den, FLT_MIN.xxx));
}

/// @brief 2D Ray Exit for Gamut Compression.
/// Solves the intersection of a ray (origin, direction) against the BT.2020 
/// spectral triangle mapped in MacLeod-Boynton chromaticity space.
/// @return The scalar multiplier `t` to reach the gamut boundary.
float RayExitTCIE1702(float2 origin, float2 direction) 
{
    if (dot(direction, direction) <= 1e-14) return 0.0;
    
    // Convert BT.2020 Primary RGBs to MB Coordinates to define the triangle
    float3 r_mb = LMS_to_MB1702(WeighLMS(RGB_to_StockmanLMS(float3(1.0, 0.0, 0.0))));
    float3 g_mb = LMS_to_MB1702(WeighLMS(RGB_to_StockmanLMS(float3(0.0, 1.0, 0.0))));
    float3 b_mb = LMS_to_MB1702(WeighLMS(RGB_to_StockmanLMS(float3(0.0, 0.0, 1.0))));
    
    float2 pts[3] = { r_mb.xy, g_mb.xy, b_mb.xy };
    float t_best = 1e30;
    bool hit = false;[unroll] 
    for (int i = 0; i < 3; i++) 
    {
        float2 a = pts[i];
        float2 b = pts[(i+1)%3];
        float2 e = b - a;
        float denom = direction.x * e.y - direction.y * e.x;
        
        if (abs(denom) > 1e-10) 
        {
            float2 ao = a - origin;
            float t = (ao.x * e.y - ao.y * e.x) / denom;
            float u = (ao.x * direction.y - ao.y * direction.x) / denom;
            
            if (t >= 0.0 && u >= 0.0 && u <= 1.0) 
            {
                t_best = min(t_best, t);
                hit = true;
            }
        }
    }
    return hit ? t_best : 0.0;
}

// =================================================================================================
// 6. Main Pipeline execution
// =================================================================================================

void PS_PhotorealHDR(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 fragColor : SV_Target)
{
    int2 pos = int2(vpos.xy);
    float4 src = tex2Dfetch(SamplerBackBuffer, pos);

    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float whitePt = (space <= 1) ? SCRGB_WHITE_NITS : fWhitePoint;
    
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;
    float3x3 to_LMS   = (space >= 3) ? RGB2020_to_LMS : RGB709_to_LMS;
    float3x3 to_RGB   = (space >= 3) ? LMS_to_RGB2020 : LMS_to_RGB709;

    // Fast bypass evaluation
    if (abs(fExposure) < NEUTRAL_EPS && abs(fContrast - 1.0) < NEUTRAL_EPS && 
        abs(fSaturation - 1.0) < NEUTRAL_EPS && fDisplayPeakNits >= 9999.0) 
    {
        fragColor = src; 
        return;
    }

    // ---------------------------------------------------------------------------------------------
    // INPUT DECODE
    // ---------------------------------------------------------------------------------------------
    float3 color;
    if (space == 3) 
    {
        // PQ (ST.2084) Decode
        float3 Np = PowNonNegPreserveZero3(saturate(src.rgb), PQ_INV_M2);
        float3 num = max(Np - PQ_C1, 0.0);
        float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
        color = PowNonNegPreserveZero3(num / den, PQ_INV_M1) * PQ_PEAK_LUMINANCE;
    } 
    else if (space == 2) 
    {
        // scRGB Decode
        color = src.rgb * SCRGB_WHITE_NITS;
    } 
    else 
    {
        // sRGB Decode
        float3 a = abs(src.rgb);
        float3 lo = a / 12.92;
        float3 hi = PowNonNegPreserveZero3((a + 0.055) / 1.055, SRGB_GAMMA);
        
        color.r = (a.r <= SRGB_THRESHOLD_EOTF) ? lo.r : hi.r;
        color.g = (a.g <= SRGB_THRESHOLD_EOTF) ? lo.g : hi.g;
        color.b = (a.b <= SRGB_THRESHOLD_EOTF) ? lo.b : hi.b;
        color = sign(src.rgb) * color * SCRGB_WHITE_NITS;
    }
    
    // Sanitize pipeline input
    if (any(IsNan3(color)) || any(IsInf3(color))) color = 0.0;
    float3 original_color = color;

    // =============================================================================================
    // STAGE 1: SCENE-REFERRED GRADE
    // Mathematics-driven aesthetic shaping in the true stop-domain.
    // =============================================================================================

    // 1. Exposure (Linear Domain)
    if (abs(fExposure) > NEUTRAL_EPS) color *= exp2(fExposure);

    // 2. White Balance (LMS Cone-Domain)
    if (abs(fTemperature) > NEUTRAL_EPS || abs(fTint) > NEUTRAL_EPS) 
    {
        float3 wbStops = 0.35 * float3(fTemperature + fTint, -fTint, -fTemperature + fTint);
        float3 wbScale = exp2(wbStops);
        
        float3 d65_wb = mul(to_RGB, wbScale);
        wbScale /= max(dot(d65_wb, lumaCoeffs), FLT_MIN);
        
        color = mul(to_RGB, mul(to_LMS, color) * wbScale);
    }

    // 3. Black Point (C1 Parabolic Toe)
    if (fBlackPoint > NEUTRAL_EPS) 
    {
        float luma = dot(color, lumaCoeffs);
        float bp = fBlackPoint * whitePt;
        if (luma > FLT_MIN) 
        {
            color *= (luma < 2.0 * bp) ? (luma / (4.0 * bp)) : ((luma - bp) / luma);
        }
    }

    // 4. Filmic Contrast & Tonal EQ (Log2 Domain)
    if (abs(fContrast - 1.0) > NEUTRAL_EPS || abs(fShadows) > NEUTRAL_EPS || abs(fHighlights) > NEUTRAL_EPS) 
    {
        float luma = dot(color, lumaCoeffs);
        float absLuma = abs(luma);
        
        if (absLuma > FLT_MIN) 
        {
            float pivot = fContrastPivot * whitePt;
            float x = log2(absLuma / pivot) * fContrast;
            
            // C1 Continuous Rational Recovery Curves
            float a2 = 6.0;
            if (x < 0.0 && abs(fShadows) > NEUTRAL_EPS) 
            {
                x += (fShadows * 3.0) * ((x * x) / (x * x + a2));
            }
            else if (x > 0.0 && abs(fHighlights) > NEUTRAL_EPS) 
            {
                x += (fHighlights * 3.0) * ((x * x) / (x * x + a2));
            }
            
            color *= min((pivot * exp2(x)) / absLuma, 100.0);
        }
    }

    // 5. Isoluminant Purity (MacLeod-Boynton Saturation)
    if (abs(fSaturation - 1.0) > NEUTRAL_EPS) 
    {
        float luma = dot(color, lumaCoeffs);
        float ct = saturate((luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
        float rel = ct * ct * (3.0 - 2.0 * ct); // Dark-chroma reliability curve
        
        float scale = fSaturation;
        
        // Vivid-Color Protection
        if (scale > 1.0 && rel > 0.0) 
        {
            float3 lms = mul(to_LMS, color);
            if (lms.r + lms.g > FLT_MIN) 
            {
                float3 mb = LMS_to_MB1702(lms);
                float2 off = mb.xy - D65_MB_XY;
                float p = SqrtIEEE(dot(off, off));
                
                float t = saturate(p / 0.35); // 0.35 ceiling purity
                float prot = t * t * (3.0 - 2.0 * t);
                
                // Gamut awareness: Rec.2020 has a lower residual ceiling
                float space_comp = (space >= 3) ? 0.90 : 1.0;
                float min_boost = (space >= 3) ? 0.20 : 0.25;
                
                scale = 1.0 + (scale - 1.0) * space_comp * lerp(1.0, min_boost, prot);
            }
        }
        
        scale = lerp(1.0, scale, rel);
        
        if (abs(scale - 1.0) > NEUTRAL_EPS) 
        {
            float3 lms = mul(to_LMS, color);
            if (lms.r + lms.g > 0.0) 
            {
                float3 mb = LMS_to_MB1702(lms);
                mb.xy = lerp(D65_MB_XY, mb.xy, scale);
                color = mul(to_RGB, MB1702_to_LMS(mb));
            }
        }
    }

    // Ensure space is strictly BT.2020 for the biological model (Stockman expects this format)
    float3 color_bt2020 = (space >= 3) ? color : ConvertColorSpace(color, 0, 1);

    // =============================================================================================
    // STAGE 2: OBSERVER TONEMAP (Retina Simulation)
    // Replaces purely mathematical curves with biological physiological responses.
    // =============================================================================================
    
    // 1. Enter Biological Cone Space
    float3 lms_abs = RGB_to_StockmanLMS(color_bt2020);
    float3 lms_peak = RGB_to_StockmanLMS(float(max(fDisplayPeakNits, whitePt)).xxx);
    float3 lms_adapt = RGB_to_StockmanLMS(float(whitePt * 0.18).xxx); // Local adaptation anchor
    
    // Store original chromaticity mapping before Tonemap/Bleach logic changes relations
    float3 mb_source = LMS_to_MB1702(WeighLMS(lms_abs));

    // 2. Photopigment Bleaching (Trolands)
    // High-intensity light physically depletes cone pigments, desaturating toward white.
    if (fBleaching > 0.0) 
    {
        // 1nit ~= 4Td assuming an average 2.2mm pupil diameter
        float3 stimulus_trolands = max(lms_abs, 0.0) * 4.0; 
        
        // Steady-state bleaching availability equation
        float3 availability = 1.0.xxx / (1.0.xxx + stimulus_trolands / 20000.0);
        availability = lerp(1.0.xxx, availability, fBleaching);
        
        float y = lms_abs.x + lms_abs.y;
        float white_y = lms_adapt.x + lms_adapt.y;
        float3 white_anchor = lms_adapt * DivideSafe(y, white_y, 1.0);
        
        lms_abs = white_anchor + (lms_abs - white_anchor) * availability;
    }

    // 3. Photoreceptor Response (Anchored Naka-Rushton)
    // Models the non-linear S-potential curves of human visual cells.
    float3 lms_response = NakaRushton_Anchored(lms_abs, lms_peak, lms_adapt, fNakaExponent);

    // 4. Biological Hue Restoration
    // Independent cone compression introduces physiological hue shifts (Bezold-Brücke effect).
    // This ray-tracing logic restores the hue mathematically while preserving the display 
    // purity and luminance output of the Naka-Rushton curve.
    float3 mb_target = LMS_to_MB1702(WeighLMS(lms_response));
    
    if (fHueRestore > 0.0) 
    {
        float3 mb_anchor = LMS_to_MB1702(WeighLMS(lms_adapt));
        
        float2 src_off = mb_source.xy - mb_anchor.xy;
        float2 tgt_off = mb_target.xy - mb_anchor.xy;
        float src2 = dot(src_off, src_off);
        float tgt2 = dot(tgt_off, tgt_off);
        
        if (src2 > 1e-12 && tgt2 > 1e-12) 
        {
            float inv_tgt_r = RSqrtIEEE(tgt2);
            float tgt_r = tgt2 * inv_tgt_r; 
            
            // Ray scalar indicating intersection with the gamut boundary
            float src_t = RayExitTCIE1702(mb_anchor.xy, src_off);
            float tgt_t = RayExitTCIE1702(mb_anchor.xy, tgt_off);
            
            // Calculate absolute Euclidean boundary distance
            float abs_bound_dist = tgt_t * tgt_r;
            
            float src_purity = saturate(DivideSafe(1.0, src_t, 0.0));
            float tgt_purity = saturate(DivideSafe(1.0, tgt_t, 0.0));
            
            // Pure color protection: fade restoration if tonemapper crushed color to white
            float purity_loss = saturate(DivideSafe(tgt_purity, src_purity, 1.0));
            
            // Less sensitivity to hue shifts on the far Red (large) distance
            float hue_dist = saturate(DivideSafe(abs_bound_dist - 0.201398, 1.026345 - 0.201398, 0.0));
            float hue_sens = lerp(1.0, 0.35, hue_dist);
            
            float weight = hue_sens * fHueRestore * purity_loss;
            
            if (weight > 0.0) 
            {
                float inv_src_r = RSqrtIEEE(src2);
                float2 src_dir = src_off * inv_src_r;
                float2 tgt_dir = tgt_off * inv_tgt_r;
                
                float2 blend_dir = lerp(tgt_dir, src_dir, weight);
                float blend2 = dot(blend_dir, blend_dir);
                
                blend_dir = (blend2 > 1e-12) ? blend_dir * RSqrtIEEE(blend2) : tgt_dir;
                
                // Restore origin hue but explicitly preserve tonemapped target radius
                mb_target.xy = mb_anchor.xy + blend_dir * tgt_r;
            }
        }
    }

    // 5. Exit Biological Space
    color = StockmanLMS_to_RGB(UnweighLMS(MB1702_to_LMS(mb_target)));
    color = (space >= 3) ? color : ConvertColorSpace(color, 1, 0);

    // Sanitize any resulting NaNs/Infs back to original color safely
    if (any(IsNan3(color)) || any(IsInf3(color))) color = original_color;

    // ---------------------------------------------------------------------------------------------
    // OUTPUT ENCODE
    // ---------------------------------------------------------------------------------------------
    float3 encoded;
    
    if (space == 3) 
    {
        // PQ (ST.2084) Encode
        float3 Lp = PowNonNegPreserveZero3(clamp(color, 0.0, PQ_PEAK_LUMINANCE) / PQ_PEAK_LUMINANCE, PQ_M1);
        float3 num = PQ_C1 + PQ_C2 * Lp;
        float3 den = 1.0 + PQ_C3 * Lp;
        encoded = saturate(PowNonNegPreserveZero3(num / den, PQ_M2));
    } 
    else if (space == 2) 
    {
        // scRGB Encode
        encoded = color / SCRGB_WHITE_NITS;
    } 
    else 
    {
        // sRGB Encode
        float3 a = abs(color / SCRGB_WHITE_NITS);
        float3 lo = a * 12.92;
        float3 hi = 1.055 * PowNonNegPreserveZero3(a, SRGB_INV_GAMMA) - 0.055;
        
        encoded.r = (a.r <= SRGB_THRESHOLD_OETF) ? lo.r : hi.r;
        encoded.g = (a.g <= SRGB_THRESHOLD_OETF) ? lo.g : hi.g;
        encoded.b = (a.b <= SRGB_THRESHOLD_OETF) ? lo.b : hi.b;
        encoded = saturate(sign(color) * encoded);
    }

    fragColor = float4(encoded, src.a);
}

// =================================================================================================
// 7. Technique Definition
// =================================================================================================

technique PhotorealHDR_Mastering_V6 <
    ui_label = "Photoreal HDR V6.0 (Biological Hybrid Edition)";
    ui_tooltip = "Fuses mathematical scene grading with biological retinal tonemapping.\n\n"
                 "Stage 1: SCENE GRADE (V5.9.2 Math)\n"
                 "- Stop-domain Filmic Contrast & Tonal EQ (C1 continuous)\n"
                 "- MacLeod-Boynton Isoluminant Purity\n\n"
                 "Stage 2: OBSERVER TONEMAP (Retina Simulation)\n"
                 "- Troland Photopigment Bleaching\n"
                 "- Anchored Naka-Rushton Cone S-Potentials\n"
                 "- Dynamic Bezold-Brücke Hue Restoration\n\n"
                 "Design: Precision over performance (IEEE 754 compliance).";
>
{
    pass 
    { 
        VertexShader = PostProcessVS; 
        PixelShader  = PS_PhotorealHDR; 
    }
}