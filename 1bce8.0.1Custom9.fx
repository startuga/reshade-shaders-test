/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement - MASTERING EDITION
 *
 * Design Philosophy: PRECISION OVER PERFORMANCE
 * - True IEEE 754 Math (No fast intrinsics or approximations)
 * - Exact IEC/SMPTE Standard Constants
 * - Bit-Exact Neutrality Logic
 * - Pre-computed High-Precision Kernels
 * - True Stop-Domain HDR Processing
 * - Stable Near-Black Behavior
 *
 * Version: 8.2.9 (Fix: SDR Zone Logic, Scharr Normalization, Debug Consistency)
 * Author: startuga
 */

#include "ReShade.fxh"  // https://github.com/crosire/reshade-shaders/blob/slim/Shaders/ReShade.fxh
                        // https://github.com/crosire/reshade-shaders/blob/slim/REFERENCE.md

// ==============================================================================
// 1. High-Precision Constants & Color Science Definitions
// ==============================================================================

// Mathematical Constants
static const float FLT_MIN = 1.175494351e-38;   // Strict float32 min normalized
static const float LN_FLT_MIN = -87.33654475;   // ln(FLT_MIN) for exp optimization
static const float NEG_LN_SPATIAL_CUTOFF = 9.210340372; // -ln(1e-4)

// Algorithm Thresholds
static const float SPATIAL_CUTOFF = 1e-4;       // Tight cutoff for high-precision kernel radius
static const int MAX_LOOP_RADIUS = 32;          // Capped to prevent Driver Timeout (TDR)
static const float RATIO_MIN = 0.0001;          // Ratio safety clamp min
static const float RATIO_MAX = 10000.0;         // Ratio safety clamp max
static const float CHROMA_STABILITY_THRESH = 1e-4; // Linear value below which chroma weight fades out

// IEC 61966-2-1:1999 defines the EOTF threshold.
static const float SRGB_THRESHOLD_EOTF = 0.04045;
// Derived OETF threshold for perfect round-trip (compiler will fold this with high precision)
static const float SRGB_THRESHOLD_OETF = (0.04045 / 12.92); 

// ITU-R Rec.709 Luma Coefficients (Standard)
static const float3 Luma709 = float3(0.2126, 0.7152, 0.0722);

// ITU-R Rec.2020 Luma Coefficients (Standard)
static const float3 Luma2020 = float3(0.2627, 0.6780, 0.0593);

// ST.2084 (PQ) EOTF Constants (SMPTE ST 2084-2014)
static const float PQ_M1 = 0.1593017578125;    // 2610/16384 = 0.1593017578125 (exact)
static const float PQ_M2 = 78.84375;           // 2523/4096 x 128 = 78.84375 (exact)
static const float PQ_C1 = 0.8359375;          // 3424/4096 =0.8359375 = c3 − c2 + 1 (exact)
static const float PQ_C2 = 18.8515625;         // 2413/4096 x 32 = 18.8515625 (exact)
static const float PQ_C3 = 18.6875;            // 2392/4096 x 32 = 18.6875 (exact)
static const float PQ_PEAK_LUMINANCE = 10000.0;

// scRGB Standard Definition
// In standard Windows scRGB (16-bit float), 1.0 linear = 80 nits.
// 12.5 linear = 1000 nits.
static const float SCRGB_WHITE_NITS = 80.0;

// Zone System: Mathematically Exact Powers of 2
// Zone V (Middle Grey) is anchored at exactly 2^-2.5
static const float ZONE_I    = 0.04419417382; // pow(2, -4.5)
static const float ZONE_II   = 0.06250000000; // pow(2, -4.0)
static const float ZONE_III  = 0.08838834764; // pow(2, -3.5)
static const float ZONE_IV   = 0.12500000000; // pow(2, -3.0)
static const float ZONE_V    = 0.17677669529; // pow(2, -2.5)
static const float ZONE_VI   = 0.25000000000; // pow(2, -2.0)
static const float ZONE_VII  = 0.35355339059; // pow(2, -1.5)
static const float ZONE_VIII = 0.50000000000; // pow(2, -1.0)
static const float ZONE_IX   = 0.70710678118; // pow(2, -0.5)
static const float ZONE_X    = 1.00000000000; // pow(2, 0.0)
static const float ZONE_XI   = 2.00000000000; // pow(2, 1.0)

// Pre-computed Kernel for Structure Tensor (Binomial 3x3 - separable [1,2,1]⊗[1,2,1]/16)
static const float3x3 Structure_Gauss = float3x3(
    0.0625, 0.1250, 0.0625,  // 1/16, 2/16, 1/16
    0.1250, 0.2500, 0.1250,  // 2/16, 4/16, 2/16
    0.0625, 0.1250, 0.0625   // 1/16, 2/16, 1/16
);

// Edge Detection Kernels
static const float Sobel5x5_Gx[25] = {
    -1,-2,0,2,1, -4,-8,0,8,4, -6,-12,0,12,6, -4,-8,0,8,4, -1,-2,0,2,1
};
static const float Sobel5x5_Gy[25] = {
    -1,-4,-6,-4,-1, -2,-8,-12,-8,-2, 0,0,0,0,0, 2,8,12,8,2, 1,4,6,4,1
};
static const float LoG_Kernel[25] = {
    0,0,-1,0,0, 0,-1,-2,-1,0, -1,-2,16,-2,-1, 0,-1,-2,-1,0, 0,0,-1,0,0
};

// Debug Visualization Colors
float3 GetZoneColor(int index)
{
    [flatten]
    switch(clamp(index, 0, 12)) {
        case 0: return float3(0.5, 0.0, 0.5);      // Negative
        case 1: return float3(0.02, 0.02, 0.05);   // Zone 0
        case 2: return float3(0.1, 0.0, 0.1);      // Zone I
        case 3: return float3(0.2, 0.0, 0.3);      // Zone II
        case 4: return float3(0.3, 0.0, 0.5);      // Zone III
        case 5: return float3(0.2, 0.2, 0.8);      // Zone IV
        case 6: return float3(0.5, 0.5, 0.5);      // Zone V
        case 7: return float3(0.8, 0.8, 0.2);      // Zone VI
        case 8: return float3(1.0, 0.8, 0.3);      // Zone VII
        case 9: return float3(1.0, 0.6, 0.4);      // Zone VIII
        case 10: return float3(1.0, 0.9, 0.8);     // Zone IX
        case 11: return float3(1.0, 1.0, 1.0);     // Zone X
        case 12: return float3(1.0, 1.0, 0.5);     // Zone XI
        default: return float3(0, 0, 0);
    }
}

// ==============================================================================
// 2. Texture & System Config
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

// Safety Check: Ensure ReShade API has injected buffer constants.
// This prevents silent data corruption on 4K/Ultrawide displays.
#if !defined(BUFFER_WIDTH) || !defined(BUFFER_HEIGHT)
    #error "Bilateral Contrast: Missing BUFFER_WIDTH/HEIGHT. ReShade.fxh injection failed."
#endif

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

// ==============================================================================
// 3. UI Parameters
// ==============================================================================

uniform int iQualityPreset <
    ui_type = "combo";
    ui_label = "Quality Preset";
    ui_items = "Custom\0Reference (Mastering)\0";
    ui_category = "Presets";
> = 0;

uniform float fStrength <
    ui_type = "slider";
    ui_label = "Contrast Strength";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.001;
    ui_category = "Core Settings";
> = 2.5;

uniform float fShadowProtection <
    ui_type = "slider";
    ui_label = "Shadow Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0;

uniform float fMidtoneProtection <
    ui_type = "slider";
    ui_label = "Midtone Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0;

uniform float fHighlightProtection <
    ui_type = "slider";
    ui_label = "Highlight Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0;

uniform float fZoneWhitePoint <
    ui_type = "slider";
    ui_label = "Zone White Point (Nits)";
    ui_min = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_tooltip = "Only applies when using HDR/scRGB Color Space overrides.";
    ui_category = "Protection Zones";
> = 203.0; // ITU-R BT.2408 reference diffuse white

uniform float fNegativeProtection <
    ui_type = "slider";
    ui_label = "Negative Value Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Protection Zones";
> = 0;

uniform bool bAdaptiveStrength <
    ui_label = "Enable Adaptive Strength";
    ui_category = "Adaptive Processing";
> = true;

uniform int iAdaptiveMode <
    ui_type = "combo";
    ui_label = "Adaptive Mode";
    ui_items = "Dynamic Range\0Variance\0Hybrid\0Range-Variance Hybrid\0";
    ui_category = "Adaptive Processing";
> = 3;

uniform float fAdaptiveAmount <
    ui_type = "slider";
    ui_label = "Adaptive Amount";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Adaptive Processing";
> = 0.25;

uniform float fAdaptiveCurve <
    ui_type = "slider";
    ui_label = "Adaptive Curve";
    ui_min = 0.1; ui_max = 4.0; ui_step = 0.01;
    ui_category = "Adaptive Processing";
> = 1.25;

uniform int iRadius <
    ui_type = "slider";
    ui_label = "Filter Radius";
    ui_min = 1; ui_max = 32;
    ui_category = "Filter Parameters";
> = 8;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_min = 0.1; ui_max = 32.0; ui_step = 0.01;
    ui_category = "Filter Parameters";
> = 4;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Range Sigma (Stops)";
    ui_min = 0.01; ui_max = 4.0; ui_step = 0.001;
    ui_category = "Filter Parameters";
> = 0.35;

uniform float fSigmaChroma <
    ui_type = "slider";
    ui_label = "Chroma Sigma";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Filter Parameters";
> = 0.15;

uniform bool bChromaAwareBilateral <
    ui_label = "Chroma-Aware Filtering";
    ui_category = "Filter Parameters";
> = true;

uniform bool bAdaptiveRadius <
    ui_label = "Adaptive Radius";
    ui_category = "Adaptive Radius";
> = true;

uniform float fAdaptiveRadiusStrength <
    ui_type = "slider";
    ui_label = "Adaptive Radius Strength";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Adaptive Radius";
> = 0.7;

uniform float fChromaEdgeStrength <
    ui_type = "slider";
    ui_label = "Chroma Edge Influence";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Controls how strongly chroma edges reduce the filter radius. \n0.0 = Luma only. 1.0 = Max(Luma, Chroma).";
    ui_category = "Adaptive Radius";
> = 0.5;

uniform int iEdgeDetectionMethod <
    ui_type = "combo";
    ui_label = "Edge Detection Method";
    ui_items = "Sobel 3x3\0Scharr 3x3\0Prewitt 3x3\0Sobel 5x5\0Laplacian of Gaussian\0Structure Tensor\0";
    ui_category = "Adaptive Radius";
> = 5;

uniform float fGradientSensitivity <
    ui_type = "slider";
    ui_label = "Gradient Sensitivity";
    ui_min = 10.0; ui_max = 500.0; ui_step = 1.0;
    ui_category = "Advanced Tuning";
> = 150.0;

uniform float fVarianceWeight <
    ui_type = "slider";
    ui_label = "Variance Weight";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Advanced Tuning";
> = 0.65;

uniform int iColorSpaceOverride <
    ui_type = "combo";
    ui_label = "Color Space Override";
    ui_items = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0";
    ui_tooltip = "Selects the EOTF/OETF used for decoding. \n'Auto' uses BUFFER_COLOR_SPACE definition. \nscRGB assumes 1.0 = 80 nits.";
    ui_category = "System";
> = 0;

uniform bool bGamutMapping <
    ui_label = "Gamut Mapping (Soft Knee Compression)";
    ui_category = "Output Quality";
> = false;

uniform float fGamutKnee <
    ui_type = "slider";
    ui_label = "Gamut Knee Strength";
    ui_min = 0.0; ui_max = 0.1; ui_step = 0.001;
    ui_tooltip = "Threshold for soft rollover to black. Helps prevent aliasing in dark gradients.";
    ui_category = "Output Quality";
> = 0.01;

uniform bool bQuantize10Bit <
    ui_label = "Simulate 10-bit Interface";
    ui_tooltip = "Rounds output to nearest 10-bit step (0-1023). Useful for verifying HDR banding.";
    ui_category = "Output Quality";
> = false;

uniform int iDebugMode <
    ui_type = "combo";
    ui_label = "Debug Visualization";
    ui_items = "Off\0Weights\0Variance\0Dynamic Range\0Enhancement Map\0Adaptive Radius\0Edge Detection\0Black Pixels\0Chroma Edges\0Entropy\0Zone Map\0Negative Values\0Signed Luminance\0";
    ui_category = "Debug";
> = 0;

// ==============================================================================
// 4. True Math Utilities (Bit-Exact Safety)
// ==============================================================================

// Safe Sqrt (guards against domain error)
float TrueSqrt(float x) { return sqrt(max(x, 0.0)); }

// PowSafe: Guards against NaN/Inf but clamps base to FLT_MIN. 
// Use this for general computations where 0.0 inputs are invalid or unimportant.
float PowSafe(float base, float exponent)
{
    // Guard against both underflow and overflow
    float safe_base = max(abs(base), FLT_MIN);
    float result = pow(safe_base, exponent);
    // Clamp result to prevent Inf propagation in negative exponent cases
    return (exponent < 0.0) ? min(result, 1e38) : result;
}

// PowNonNegPreserveZero: Critical for PQ/sRGB EOTFs.
// Preserves exact 0.0 input mapping to 0.0 output, preventing lifted blacks.
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

float GetMinComponent(float3 lin) { return min(min(lin.r, lin.g), lin.b); }

// Robust Smoothstep with precise boundary checks
// Prevents division by zero when edges are extremely close
float TrueSmoothstep(float edge0, float edge1, float x)
{
    float diff = edge1 - edge0;
    if (abs(diff) < FLT_MIN) return step(edge0, x);
    float t = saturate((x - edge0) / diff);
    return t * t * (3.0 - 2.0 * t);
}

// Float Classification
bool IsNanVal(float x) { return (asuint(x) & 0x7FFFFFFF) > 0x7F800000; }
bool IsInfVal(float x) { return (asuint(x) & 0x7FFFFFFF) == 0x7F800000; }
bool3 IsNan3(float3 v) { return bool3(IsNanVal(v.x), IsNanVal(v.y), IsNanVal(v.z)); }
bool3 IsInf3(float3 v) { return bool3(IsInfVal(v.x), IsInfVal(v.y), IsInfVal(v.z)); }

// ==============================================================================
// 5. Color Science (Exact Standard Definitions)
// ==============================================================================

float3 sRGB_EOTF(float3 V)
{
    float3 abs_V = abs(V);
    // Exact IEC 61966-2-1 logic
    float3 linear_lo = abs_V / 12.92;
    // Use PreserveZero to keep perfect black
    float3 linear_hi = PowNonNegPreserveZero3((abs_V + 0.055) / 1.055, 2.4);
    
    // Strict branching to avoid interpolation errors around threshold
    float3 out_lin;
    out_lin.r = (abs_V.r <= 0.04045) ? linear_lo.r : linear_hi.r;
    out_lin.g = (abs_V.g <= 0.04045) ? linear_lo.g : linear_hi.g;
    out_lin.b = (abs_V.b <= 0.04045) ? linear_lo.b : linear_hi.b;
    return sign(V) * out_lin;
}

float3 sRGB_OETF(float3 L)
{
    float3 abs_L = abs(L);
    float3 encoded_lo = abs_L * 12.92;
    // Use PreserveZero to keep perfect black
    float3 encoded_hi = 1.055 * PowNonNegPreserveZero3(abs_L, 1.0/2.4) - 0.055;
    
    float3 out_enc;
    // Use derived threshold for exact round-trip
    out_enc.r = (abs_L.r <= SRGB_THRESHOLD_OETF) ? encoded_lo.r : encoded_hi.r;
    out_enc.g = (abs_L.g <= SRGB_THRESHOLD_OETF) ? encoded_lo.g : encoded_hi.g;
    out_enc.b = (abs_L.b <= SRGB_THRESHOLD_OETF) ? encoded_lo.b : encoded_hi.b;
    return sign(L) * out_enc;
}

float3 PQ_EOTF(float3 N)
{
    float3 N_safe = max(N, 0.0);
    // Use PreserveZero to ensure N=0 -> Np=0
    float3 Np = PowNonNegPreserveZero3(N_safe, 1.0 / PQ_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    // Use PreserveZero to ensure num=0 -> result=0
    return PowNonNegPreserveZero3(num / den, 1.0 / PQ_M1) * PQ_PEAK_LUMINANCE;
}

float3 PQ_InverseEOTF(float3 L)
{
    float3 L_safe = max(L, 0.0) / PQ_PEAK_LUMINANCE;
    // Use PreserveZero to ensure L=0 -> Lp=0
    float3 Lp = PowNonNegPreserveZero3(L_safe, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    // Use PreserveZero to ensure result is stable
    return PowNonNegPreserveZero3(num / den, PQ_M2);
}

float3 DecodeToLinear(float3 encoded)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    if (space == 3) return PQ_EOTF(encoded);
    if (space == 2) return encoded * SCRGB_WHITE_NITS;
    return sRGB_EOTF(encoded) * SCRGB_WHITE_NITS;
}

float3 EncodeFromLinear(float3 lin)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    if (space == 3) return PQ_InverseEOTF(lin);
    if (space == 2) return lin / SCRGB_WHITE_NITS;
    return sRGB_OETF(lin / SCRGB_WHITE_NITS);
}

float GetLuminanceCS(float3 lin)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    return dot(lin, (space >= 3) ? Luma2020 : Luma709);
}

// Resolves the effective white point in linear-light nits for the active color space.
float GetResolvedWhitePoint()
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    return (space <= 1) ? SCRGB_WHITE_NITS : fZoneWhitePoint;
}

float3 SoftClipGamut(float3 lin, float knee)
{
    float minComponent = min(min(lin.r, lin.g), lin.b);
    
    // BT.2408-inspired Soft Knee Compression
    // Starts compressing when values drop below 'knee' threshold
    // Smoothly preserves Luma while bringing Chroma back into gamut.
    if (minComponent < knee)
    {
        float luma = GetLuminanceCS(lin);
        
        // Calculate compression factor
        // If knee is 0, this reverts to hard clip.
        // If knee > 0, we use a smooth Hermite interpolation to 0.
        float t = (knee > FLT_MIN) ? saturate((knee - minComponent) / (knee + FLT_MIN)) : step(minComponent, 0.0);
        
        // Smooth scaling factor
        float3 chroma = lin - luma;
        // FLT_MIN added to prevent divide-by-zero
        float scale = luma / (luma - minComponent + FLT_MIN);
        
        // Blend between original and projected based on knee proximity
        // t=0 (above knee) -> keep original
        // t=1 (deeply negative) -> fully project to luma
        scale = lerp(1.0, min(scale, 1.0), t * t * (3.0 - 2.0 * t)); // Hermite smooth
        
        lin = luma + chroma * scale;
    }
    return lin;
}

// ==============================================================================
// 6. Zone Logic (Stop-Domain)
// ==============================================================================

int GetZone(float normalizedLuma)
{
    if (normalizedLuma < 0.0) return 0;
    if (normalizedLuma < ZONE_I) return 1;
    if (normalizedLuma < ZONE_II) return 2;
    if (normalizedLuma < ZONE_III) return 3;
    if (normalizedLuma < ZONE_IV) return 4;
    if (normalizedLuma < ZONE_V) return 5;
    if (normalizedLuma < ZONE_VI) return 6;
    if (normalizedLuma < ZONE_VII) return 7;
    if (normalizedLuma < ZONE_VIII) return 8;
    if (normalizedLuma < ZONE_IX) return 9;
    if (normalizedLuma < ZONE_X) return 10;
    if (normalizedLuma < ZONE_XI) return 11;
    return 12;
}

float GetZoneProtection(float nl, float minCompNorm, float shadowProt, float midProt, float highProt, float negProt)
{
    if (shadowProt + midProt + highProt + negProt < FLT_MIN) return 1.0;

    float negW = TrueSmoothstep(0.0, -0.001, minCompNorm);
    float s = log2(max(nl, FLT_MIN));

    // Protection curves defined in pure Stop domain
    float blackW = 1.0 - TrueSmoothstep(-20.0, -14.0, s);
    float shadowProtEff = lerp(shadowProt, 1.0, blackW);

    // Shadows: Ramp down from -3.0 (Zone IV) to -2.5 (Zone V boundary)
    float shadowW = (1.0 - negW) * (1.0 - TrueSmoothstep(-3.0, -2.5, s));
    
    // Highlights: Covers everything above transition
    float highW = (1.0 - negW) * TrueSmoothstep(-1.0, 0.0, s);
    
    // Midtones: The remainder between Shadows and Highlights
    float midW = (1.0 - negW) * (1.0 - shadowW) * (1.0 - highW);

    float protection = negW * negProt + shadowW * shadowProtEff + midW * midProt + highW * highProt;
    return 1.0 - saturate(protection);
}

// ==============================================================================
// 7. Analysis & Edge Detection
// ==============================================================================

float3 FetchLinear(int2 pos)
{
    // Clamp is MANDATORY here because tex2Dfetch (integer coordinates) ignores
    // sampler address modes (AddressU/V = CLAMP) in HLSL.
    // We must clamp to valid buffer range to emulate "Clamp To Edge" behavior.
    // Failing to clamp would result in 0 (black) or undefined values at screen edges.
    pos = clamp(pos, int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
    return DecodeToLinear(tex2Dfetch(SamplerBackBuffer, pos).rgb);
}

float FetchPerceptualLuma(int2 pos)
{
    float3 lin = FetchLinear(pos);
    float luma = max(GetLuminanceCS(lin), 0.0);
    // Pure log2 perceptual metric
    return (log2(max(luma, FLT_MIN)) + 20.0) * 0.06;
}

float Sobel3x3(int2 center)
{
    float tl = FetchPerceptualLuma(center + int2(-1, -1)); 
    float tc = FetchPerceptualLuma(center + int2( 0, -1)); 
    float tr = FetchPerceptualLuma(center + int2( 1, -1));
    float ml = FetchPerceptualLuma(center + int2(-1,  0));                                                              
    float mr = FetchPerceptualLuma(center + int2( 1,  0));
    float bl = FetchPerceptualLuma(center + int2(-1,  1)); 
    float bc = FetchPerceptualLuma(center + int2( 0,  1)); 
    float br = FetchPerceptualLuma(center + int2( 1,  1));
    float gx = (tr + 2.0*mr + br) - (tl + 2.0*ml + bl);
    float gy = (bl + 2.0*bc + br) - (tl + 2.0*tc + tr);
    return (gx*gx + gy*gy) * 0.0625;
}

float Scharr3x3(int2 center)
{
    float tl = FetchPerceptualLuma(center + int2(-1, -1)); 
    float tc = FetchPerceptualLuma(center + int2( 0, -1)); 
    float tr = FetchPerceptualLuma(center + int2( 1, -1));
    float ml = FetchPerceptualLuma(center + int2(-1,  0));                                                              
    float mr = FetchPerceptualLuma(center + int2( 1,  0));
    float bl = FetchPerceptualLuma(center + int2(-1,  1)); 
    float bc = FetchPerceptualLuma(center + int2( 0,  1)); 
    float br = FetchPerceptualLuma(center + int2( 1,  1));
    float gx = (3.0*tr + 10.0*mr + 3.0*br) - (3.0*tl + 10.0*ml + 3.0*bl);
    float gy = (3.0*bl + 10.0*bc + 3.0*br) - (3.0*tl + 10.0*tc + 3.0*tr);
    // [v8.2.9 FIX] Normalization: 1/(16^2) = 1/256 = 0.00390625.
    return (gx*gx + gy*gy) * 0.00390625;
}

float Prewitt3x3(int2 center)
{
    float tl = FetchPerceptualLuma(center + int2(-1, -1)); 
    float tc = FetchPerceptualLuma(center + int2( 0, -1)); 
    float tr = FetchPerceptualLuma(center + int2( 1, -1));
    float ml = FetchPerceptualLuma(center + int2(-1,  0));                                                              
    float mr = FetchPerceptualLuma(center + int2( 1,  0));
    float bl = FetchPerceptualLuma(center + int2(-1,  1)); 
    float bc = FetchPerceptualLuma(center + int2( 0,  1)); 
    float br = FetchPerceptualLuma(center + int2( 1,  1));
    float gx = (tr + mr + br) - (tl + ml + bl);
    float gy = (bl + bc + br) - (tl + tc + tr);
    return (gx*gx + gy*gy) * 0.111111111;
}

float Sobel5x5(int2 center)
{
    float sum_gx = 0.0; float sum_gy = 0.0;
    [unroll] for (int y = -2; y <= 2; y++) {
        [unroll] for (int x = -2; x <= 2; x++) {
            float luma = FetchPerceptualLuma(center + int2(x, y));
            int idx = (y + 2) * 5 + (x + 2);
            sum_gx += luma * Sobel5x5_Gx[idx];
            sum_gy += luma * Sobel5x5_Gy[idx];
        }
    }
    // [v8.2.9 FIX] Normalization: 1/(48^2) = 1/2304
    return (sum_gx*sum_gx + sum_gy*sum_gy) * 0.00043402778;
}

float LaplacianOfGaussian(int2 center)
{
    float response = 0.0;
    [unroll] for (int y = -2; y <= 2; y++) {
        [unroll] for (int x = -2; x <= 2; x++) {
            float luma = FetchPerceptualLuma(center + int2(x, y));
            response += luma * LoG_Kernel[(y + 2) * 5 + (x + 2)];
        }
    }
    return response * response;
}

float StructureTensor(int2 center)
{
    float Ixx = 0.0, Iyy = 0.0, Ixy = 0.0;
    [unroll] for (int wy = -1; wy <= 1; wy++) {
        [unroll] for (int wx = -1; wx <= 1; wx++) {
            int2 pos = center + int2(wx, wy);
            
            float tl = FetchPerceptualLuma(pos + int2(-1, -1)); 
            float tc = FetchPerceptualLuma(pos + int2( 0, -1)); 
            float tr = FetchPerceptualLuma(pos + int2( 1, -1));
            float ml = FetchPerceptualLuma(pos + int2(-1,  0));                                                         
            float mr = FetchPerceptualLuma(pos + int2( 1,  0));
            float bl = FetchPerceptualLuma(pos + int2(-1,  1)); 
            float bc = FetchPerceptualLuma(pos + int2( 0,  1)); 
            float br = FetchPerceptualLuma(pos + int2( 1,  1));
            
            float gx = (tr + 2.0*mr + br) - (tl + 2.0*ml + bl);
            float gy = (bl + 2.0*bc + br) - (tl + 2.0*tc + tr);
            
            float w = Structure_Gauss[wy + 1][wx + 1];
            Ixx += gx * gx * w;
            Iyy += gy * gy * w;
            Ixy += gx * gy * w;
        }
    }
    float trace = Ixx + Iyy;
    
    // Numerically stable discriminant: (Ixx-Iyy)^2 + 4*Ixy^2
    // Prevents cancellation errors inherent in trace^2 - 4*det
    float diff = Ixx - Iyy;
    float disc = TrueSqrt(max(diff * diff + 4.0 * Ixy * Ixy, 0.0));
    
    float lambda1 = (trace + disc) * 0.5;
    float lambda2 = (trace - disc) * 0.5;
    float coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + FLT_MIN);
    return TrueSqrt(lambda1) * (1.0 + coherence) * 0.5;
}

float ChromaEdge(int2 center)
{
    float3 c = FetchLinear(center);
    float luma = max(GetLuminanceCS(c), FLT_MIN);
    float3 chroma = c / luma;
    float maxChromaDiff = 0.0;
    [unroll] for (int y = -1; y <= 1; y++) {
        [unroll] for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            float3 nc = FetchLinear(center + int2(x, y));
            float nLuma = max(GetLuminanceCS(nc), FLT_MIN);
            float3 diff = chroma - (nc / nLuma);
            maxChromaDiff = max(maxChromaDiff, dot(diff, diff));
        }
    }
    return TrueSqrt(maxChromaDiff);
}

float GetEdgeStrength(int2 center, int method)
{
    if (method == 0) return Sobel3x3(center);
    if (method == 1) return Scharr3x3(center);
    if (method == 2) return Prewitt3x3(center);
    if (method == 3) return Sobel5x5(center);
    if (method == 4) return LaplacianOfGaussian(center);
    if (method == 5) return StructureTensor(center);
    return Sobel3x3(center);
}

// ==============================================================================
// 8. Bilateral Processing
// ==============================================================================

int GetAdaptiveRadius(int2 center, int base_radius, float strength, float sigma_spatial)
{
    float edge = GetEdgeStrength(center, iEdgeDetectionMethod);
    
    // Branch optimization: Check if Chroma Aware is enabled before fetching/calculating
    [branch]
    if (bChromaAwareBilateral && fChromaEdgeStrength > 0.0) {
        float chromaEdge = ChromaEdge(center);
        // Weighted blend based on UI setting: 0.0 = Luma Only, 1.0 = Max(Luma, Chroma)
        // This preserves edges in isoluminant but colored areas (like shadows) 
        // without forcing hard radius reduction on noisy chroma.
        edge = lerp(edge, max(edge, chromaEdge), fChromaEdgeStrength);
    }
    
    float scale = TrueSmoothstep(0.0, 1.0, edge * (fGradientSensitivity * 0.01));
    // Fixed: Flat areas (scale ~ 0) should use 100% radius (factor 1.0), not 90%.
    // Edges (scale ~ 1) reduce radius to 15%.
    float factor = lerp(1.0, lerp(1.0, 0.15, scale), strength);
    int sigma_max = (int)(sigma_spatial * 3.0 + 0.5);
    return clamp(min((int)(base_radius * factor + 0.5), sigma_max), 1, base_radius);
}

float CalculateAdaptiveStrength(float sum_log, float sum_log_sq, float sum_weight, float min_log, float max_log, float base_strength, int mode)
{
    if (sum_weight < FLT_MIN) return base_strength;
    float inv_weight = 1.0 / sum_weight;
    float range = max_log - min_log;
    float mean = sum_log * inv_weight;
    float var = max(0.0, sum_log_sq * inv_weight - mean * mean);
    float metric;

    if (mode == 0) metric = saturate(range * 0.166666667);
    else if (mode == 1) metric = saturate(var * 0.5);
    else if (mode == 2) metric = PowSafe(max(saturate(range * 0.166666667), FLT_MIN), 1.0 - fVarianceWeight) * PowSafe(max(saturate(var * 0.5), FLT_MIN), fVarianceWeight);
    else metric = saturate((log2(1.0 + var) * (1.0 + range * 0.1)) * 0.25);

    return base_strength * lerp(1.0, PowSafe(metric, fAdaptiveCurve) * 2.0, fAdaptiveAmount);
}

float3 ProcessPixel(int2 center_pos)
{
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;

    float whitePt = GetResolvedWhitePoint();

    float3 color_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, center_pos).rgb);
    float luma_lin = dot(color_lin, lumaCoeffs);

    // Passthrough for non-positive luma (Log domain singularity).
    if (iDebugMode == 0 && luma_lin <= FLT_MIN) return color_lin;

    float log2_center = log2(max(luma_lin, FLT_MIN));

    // Radius determination
    int base_radius; 
    float sigma_s;

    // Strict branching to help compiler dead-strip unused paths
    [branch]
    if (iQualityPreset == 1) {
        base_radius = 24;
        sigma_s = 12.0;
    } else {
        base_radius = iRadius;
        sigma_s = fSigmaSpatial;
    }

    int radius = base_radius;
    
    // Branch optimization: Skip edge detection completely if adaptive radius is off
    [branch]
    if (bAdaptiveRadius && base_radius > 2)
        radius = GetAdaptiveRadius(center_pos, base_radius, fAdaptiveRadiusStrength, sigma_s);

    // Debug Early Exits
    if (iDebugMode == 5) return lerp(float3(0,0,1), float3(1,0,0), float(radius) / float(base_radius));
    if (iDebugMode == 6) { float e = GetEdgeStrength(center_pos, iEdgeDetectionMethod); return float3(e,e,e) * 10.0; }
    if (iDebugMode == 8) { float c = ChromaEdge(center_pos); return float3(c,c,c) * 5.0; }
    if (iDebugMode == 10) return GetZoneColor(GetZone(luma_lin / whitePt));
    if (iDebugMode == 11) return (GetMinComponent(color_lin) < 0.0) ? float3(1,0,1) : float3(0,0.1,0);
    if (iDebugMode == 12) {
        float norm = luma_lin / max(whitePt, FLT_MIN);
        float stops = log2(max(abs(norm), FLT_MIN));
        float t = saturate((stops + 6.0) / 8.0);
        return (norm < 0.0) ? float3(0, t, 0) : float3(t, 0, 0);
    }
    if (iDebugMode == 7) return (luma_lin <= FLT_MIN) ? float3(1, 0, 1) : float3(0, 0, 0);

    // Bilateral Constants
    float inv_2_sigma_s_sq = 0.5 / (sigma_s * sigma_s);
    float inv_2_sigma_r_sq = 0.5 / (fSigmaRange * fSigmaRange);
    float inv_2_sigma_c_sq = 0.5 / (fSigmaChroma * fSigmaChroma);
    
    // Precompute log constant to avoid per-pixel log()
    float cutoff_r_sq = NEG_LN_SPATIAL_CUTOFF / inv_2_sigma_s_sq;

    int cutoff_int = (int)TrueSqrt(cutoff_r_sq);
    int safe_radius = min(cutoff_int + 1, radius);
    int max_r = min(safe_radius, MAX_LOOP_RADIUS);
    
    int y_start = max(-max_r, -center_pos.y);
    int y_end   = min( max_r, (BUFFER_HEIGHT - 1) - center_pos.y);

    float r_limit_sq = float(max_r * max_r);

    // Chroma-Aware Setup
    float3 center_chroma = float3(0,0,0);
    float chroma_weight_factor = 0.0;
    if (bChromaAwareBilateral) {
        chroma_weight_factor = TrueSmoothstep(FLT_MIN, CHROMA_STABILITY_THRESH, luma_lin);
        center_chroma = color_lin / luma_lin - 1.0; 
    }

    // Neumaier Accumulators
    float2 stats_log = 0.0;
    float2 stats_sq  = 0.0;
    float2 stats_w   = 0.0;
    float min_log = log2_center, max_log = log2_center;
    
    // Precompute constant loop bounds
    int x_min_off = -center_pos.x;
    int x_max_off = (BUFFER_WIDTH - 1) - center_pos.x;

    [loop]
    for (int y = y_start; y <= y_end; ++y)
    {
        float y_f = float(y);
        float y_sq = y_f * y_f;
        float spatial_y = y_sq * inv_2_sigma_s_sq;

        int x_limit_circ = (int)TrueSqrt(max(0.0, r_limit_sq - y_sq));
        
        int x_start = max(-x_limit_circ, x_min_off);
        int x_end   = min( x_limit_circ, x_max_off);

        int sample_y = center_pos.y + y;

        [loop]
        for (int x = x_start; x <= x_end; ++x)
        {
            int sample_x = center_pos.x + x;
            
            float3 n_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, int2(sample_x, sample_y)).rgb);
            float n_luma = max(dot(n_lin, lumaCoeffs), 0.0);
            float n_log  = log2(max(n_luma, FLT_MIN));

            float d_luma = log2_center - n_log;
            
            float exponent = -(float(x * x) * inv_2_sigma_s_sq + spatial_y) - (d_luma * d_luma * inv_2_sigma_r_sq);

            if (chroma_weight_factor > 0.0 && n_luma > FLT_MIN) {
                float3 n_chroma = n_lin / n_luma - 1.0;
                float3 d_chroma = center_chroma - n_chroma;
                exponent -= (dot(d_chroma, d_chroma) * inv_2_sigma_c_sq) * chroma_weight_factor;
            }

            // Optimization using log check instead of exp+branch
            if (exponent <= LN_FLT_MIN) continue;

            float weight = exp(exponent); 
            
            // Neumaier Summation
            float val = n_log * weight;
            float t = stats_log.x + val;
            float err = (abs(stats_log.x) >= abs(val)) ? ((stats_log.x - t) + val) : ((val - t) + stats_log.x);
            stats_log.x = t; stats_log.y += err;

            val = n_log * n_log * weight;
            t = stats_sq.x + val;
            err = (abs(stats_sq.x) >= abs(val)) ? ((stats_sq.x - t) + val) : ((val - t) + stats_sq.x);
            stats_sq.x = t; stats_sq.y += err;

            t = stats_w.x + weight;
            err = (abs(stats_w.x) >= abs(weight)) ? ((stats_w.x - t) + weight) : ((weight - t) + stats_w.x);
            stats_w.x = t; stats_w.y += err;

            min_log = min(min_log, n_log);
            max_log = max(max_log, n_log);
        }
    }

    float total_w = stats_w.x + stats_w.y;
    if (total_w < FLT_MIN) return color_lin;

    float total_log = stats_log.x + stats_log.y;
    float total_sq  = stats_sq.x  + stats_sq.y;
    float blurred = total_log / total_w;
    float diff = log2_center - blurred;

    float strength = fStrength;
    [branch]
    if (bAdaptiveStrength)
        strength = CalculateAdaptiveStrength(total_log, total_sq, total_w, min_log, max_log, fStrength, iAdaptiveMode);

    float norm_luma = luma_lin / whitePt;
    float minCompNorm = 0.0;
    if (space >= 2) minCompNorm = GetMinComponent(color_lin) / whitePt;

    strength *= GetZoneProtection(norm_luma, minCompNorm, fShadowProtection, fMidtoneProtection, fHighlightProtection, fNegativeProtection);

    // Apply Contrast
    // Bit-Exact Neutrality Check
    if (abs(strength) < FLT_MIN) return color_lin;

    float enhanced_log = log2_center + strength * diff;
    float enhanced_luma = exp2(enhanced_log);
    float ratio = enhanced_luma / max(luma_lin, FLT_MIN);
    // Safety clamp to prevent FP explosion
    ratio = clamp(ratio, RATIO_MIN, RATIO_MAX); // Safety clamp

    // Post-Loop Debugs
    if (iDebugMode == 1) return saturate(log2(total_w + 1.0) * 0.1).xxx;
    if (iDebugMode == 2) { float m = blurred; float v = max(0.0, (total_sq / total_w) - m * m); return float3(v * 2.0, v, 0.0); }
    if (iDebugMode == 3) return float3((max_log - min_log) * 0.2, 0, 0);
    if (iDebugMode == 4) return lerp(float3(0,0,1), float3(1,0,0), saturate(abs(diff) * strength * 2.0));
    if (iDebugMode == 9) { float m = total_log / total_w; float v = max(0.0, (total_sq / total_w) - m * m); float r = max_log - min_log; float e = log2(1.0 + v) * (1.0 + r * 0.1); return float3(e * 0.25, e * 0.125, 0.0); }

    float3 final = color_lin * ratio;

    // Gamut Mapping (BT.2408-inspired Soft Knee)
    [branch]
    if (bGamutMapping) final = SoftClipGamut(final, fGamutKnee);
    
    if (any(IsNan3(final)) || any(IsInf3(final))) return color_lin;

    return final;
}

// ==============================================================================
// 9. Shader Entry Point
// ==============================================================================

void PS_BilateralContrast(float4 vpos : SV_Position, out float4 fragColor : SV_Target)
{
    // Dead-strip bypass: Entire shader skipped if strength is 0
    [branch]
    if (fStrength <= 0.0 && iDebugMode == 0) {
        fragColor = tex2Dfetch(SamplerBackBuffer, int2(vpos.xy));
        return;
    }

    float3 final = ProcessPixel(int2(vpos.xy));
    float3 encoded = EncodeFromLinear(final);

    // Fix: Runtime output clamping based on override, not compile-time macro.
    // Prevents clipping HDR data if the user overrides SDR buffer to HDR10/scRGB.
    int activeSpace = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    
    if (activeSpace <= 1) {
        encoded = saturate(encoded);
    }

    // 10-bit HDR Quantization (Simulation)
    // Rounds signal to 1023 steps to verify banding in HDR10 pipelines
    // Added max(0) safety to ensure negative values don't cause rounding glitches in unsigned integer emulation
    [branch]
    if (bQuantize10Bit) {
        encoded = round(max(encoded, 0.0) * 1023.0) / 1023.0;
    }

    fragColor = float4(encoded, 1.0);
}

technique BilateralContrast_Reference <
    ui_label = "Bilateral Contrast v8.2.9 (Mastering Edition)";
    ui_tooltip = "MASTERING QUALITY - True Math Processing\n\n"
                 "v8.2.9 Fixes:\n"
                 "- Fix: SDR Zone Protection white point (was 1.0, now 80 nits)\n"
                 "  Shadow/Midtone/Highlight protection now works correctly in SDR\n"
                 "- Fix: Scharr 3x3 normalization (was 1/1024, now 1/256)\n"
                 "  Scharr edge response now matches Sobel/Prewitt magnitude\n"
                 "- Fix: Negative protection extended to PQ (was scRGB only)\n"
                 "- Fix: Debug Zone Map uses resolved white point\n"
                 "  Zone Map visualization now matches actual processing zones\n"
                 "- Refactor: GetResolvedWhitePoint() shared by debug and processing\n"
                 "\nPrevious Fixes (v8.2.7-8):\n"
                 "- Sobel 5x5 normalization\n"
                 "- Native Matrix Kernel Types\n"
                 "- Weighted Chroma Edge Influence\n"
                 "- Soft Knee Gamut Mapping (BT.2408)\n"
                 "- 10-bit Quantization (HDR10 Verification)\n"
                 "- Buffer Safety Checks\n"
                 "- IEEE 754 True Math (PQ Black Fixed)\n"
                 "- Exact IEC/SMPTE Standards Constants\n"
                 "- Stable Near-Black Chroma Processing\n"
                 "- Stop-Domain Luminance Processing\n"
                 "Warning: High ALU usage. Uses Neumaier Summation for precision.\n"
                 "Requires: DirectX 10+ or OpenGL 4.5+";
>
{
    pass Main
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_BilateralContrast;
    }
}