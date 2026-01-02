/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement - MASTERING EDITION
 *
 * Design Philosophy: PRECISION OVER PERFORMANCE
 * - True IEEE 754 Math (No fast intrinsics or approximations)
 * - Exact IEC/SMPTE Standard Constants (derived from rational definitions)
 * - Bit-Exact Neutrality Logic
 * - Pre-computed High-Precision Kernels
 * - True Stop-Domain HDR Processing
 *
 * Version: 8.0.1 (Fix: Missing Constants & Compatibility)
 * Author: startuga
 */

#include "ReShade.fxh"

// ==============================================================================
// 1. High-Precision Constants & Color Science Definitions
// ==============================================================================

// Mathematical Constants (Double Precision derived, truncated to float32)
static const float PI = 3.1415926535897932384626433;
static const float EPSILON = 1e-15;             // High precision epsilon
static const float FLT_MIN = 1.175494351e-38;   // Strict float32 min normalized

// Algorithm Thresholds
static const float SPATIAL_CUTOFF = 1e-4;       // Tight cutoff for high-precision kernel radius
static const int MAX_LOOP_RADIUS = 64;          // Hard loop limit
static const float RATIO_MIN = 0.0001;          // Ratio safety clamp min
static const float RATIO_MAX = 10000.0;         // Ratio safety clamp max

// ITU-R Rec.709 Luma Coefficients (Exact Standard)
static const float3 Luma709 = float3(0.21260000, 0.71520000, 0.07220000);

// ITU-R Rec.2020 Luma Coefficients (Exact Standard)
static const float3 Luma2020 = float3(0.26270000, 0.67800000, 0.05930000);

// ST.2084 (PQ) EOTF Constants
// Derived from exact fractions: m1=2610/16384, m2=2523/32, c1=3424/4096, c2=2413/32, c3=2392/32
static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;
static const float PQ_PEAK_LUMINANCE = 10000.0;
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

// Pre-computed Kernel for Structure Tensor (Gaussian 3x3)
// Weights: Center=0.25 (1/4), Orthogonal=0.125 (1/8), Diagonal=0.0625 (1/16)
static const float Structure_Gauss[9] = {
    0.0625, 0.1250, 0.0625,
    0.1250, 0.2500, 0.1250,
    0.0625, 0.1250, 0.0625
};

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
// Note: Using standard array syntax for broad compatibility
static const float3 ZONE_COLORS[13] = {
    float3(0.5, 0.0, 0.5),      // 0: Negative
    float3(0.02, 0.02, 0.05),   // 1: Zone 0
    float3(0.1, 0.0, 0.1),      // 2: Zone I
    float3(0.2, 0.0, 0.3),      // 3: Zone II
    float3(0.3, 0.0, 0.5),      // 4: Zone III
    float3(0.2, 0.2, 0.8),      // 5: Zone IV
    float3(0.5, 0.5, 0.5),      // 6: Zone V
    float3(0.8, 0.8, 0.2),      // 7: Zone VI
    float3(1.0, 0.8, 0.3),      // 8: Zone VII
    float3(1.0, 0.6, 0.4),      // 9: Zone VIII
    float3(1.0, 0.9, 0.8),      // 10: Zone IX
    float3(1.0, 1.0, 1.0),      // 11: Zone X
    float3(1.0, 1.0, 0.5)       // 12: Zone XI
};

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

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif
#ifndef BUFFER_WIDTH
    #define BUFFER_WIDTH 1920
#endif
#ifndef BUFFER_HEIGHT
    #define BUFFER_HEIGHT 1080
#endif

// ==============================================================================
// 3. UI Parameters
// ==============================================================================

uniform int iQualityPreset <
    ui_type = "combo";
    ui_label = "Quality Preset";
    ui_items = "Custom\0Reference (Mastering)\0";
    ui_category = "Presets";
> = 1;

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
    ui_category = "Protection Zones";
> = 200.0;

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
    ui_items = "Dynamic Range\0Variance\0Hybrid\0Entropy-Based\0";
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
    ui_min = 1; ui_max = 64;
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
    ui_category = "System";
> = 0;

uniform bool bGamutMapping <
    ui_label = "Gamut Mapping (Soft Clip)";
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

// Strict IEEE 754 Math Wrappers
// Used to replace fast intrinsics and guarantee behavior near zero
float TrueSqrt(float x) { return sqrt(max(x, 0.0)); }
float TruePow(float base, float exponent) { return pow(max(base, FLT_MIN), exponent); }
float3 TruePow3(float3 base, float exponent) { return pow(max(base, FLT_MIN), exponent); }

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
bool IsNanVal(float x) { return (asuint(x) & 0x7F800000) == 0x7F800000 && (asuint(x) & 0x7FFFFF) != 0; }
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
    float3 linear_hi = TruePow3((abs_V + 0.055) / 1.055, 2.4);
    
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
    float3 encoded_hi = 1.055 * TruePow3(abs_L, 1.0/2.4) - 0.055;
    
    float3 out_enc;
    out_enc.r = (abs_L.r <= 0.0031308) ? encoded_lo.r : encoded_hi.r;
    out_enc.g = (abs_L.g <= 0.0031308) ? encoded_lo.g : encoded_hi.g;
    out_enc.b = (abs_L.b <= 0.0031308) ? encoded_lo.b : encoded_hi.b;
    return sign(L) * out_enc;
}

float3 PQ_EOTF(float3 N)
{
    float3 N_safe = max(N, 0.0);
    float3 Np = TruePow3(N_safe, 1.0 / PQ_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    return TruePow3(num / den, 1.0 / PQ_M1) * PQ_PEAK_LUMINANCE;
}

float3 PQ_InverseEOTF(float3 L)
{
    float3 L_safe = max(L, 0.0) / PQ_PEAK_LUMINANCE;
    float3 Lp = TruePow3(L_safe, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    return TruePow3(num / den, PQ_M2);
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
    // Use Double-Precision Constants calculated earlier
    return dot(lin, (space >= 2) ? Luma2020 : Luma709);
}

float3 SoftClipGamut(float3 lin)
{
    float minComponent = min(min(lin.r, lin.g), lin.b);
    if (minComponent < 0.0)
    {
        float luma = GetLuminanceCS(lin);
        float3 chroma = lin - luma;
        // FLT_MIN added to prevent divide-by-zero on black pixels
        float scale = luma / (luma - minComponent + FLT_MIN);
        lin = luma + chroma * min(scale, 1.0);
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
    float negW = TrueSmoothstep(0.0, -0.001, minCompNorm);
    float s = log2(max(nl, FLT_MIN));

    // Protection curves defined in pure Stop domain
    float blackW = 1.0 - TrueSmoothstep(-10.0, -6.0, s);
    float shadowProtEff = lerp(shadowProt, 1.0, blackW);
    float shadowW = (1.0 - negW) * (1.0 - TrueSmoothstep(-3.0, -2.5, s));
    float highW = (1.0 - negW) * TrueSmoothstep(-1.0, 0.0, s);
    float specW = TrueSmoothstep(0.0, 1.0, s);
    float highProtEff = lerp(highProt, 1.0, specW);
    float midW = (1.0 - negW) * (1.0 - shadowW) * (1.0 - highW);

    float protection = negW * negProt + shadowW * shadowProtEff + midW * midProt + highW * highProtEff;
    return 1.0 - saturate(protection);
}

// ==============================================================================
// 7. Analysis & Edge Detection
// ==============================================================================

float3 FetchLinear(int2 pos)
{
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
    // Unrolled for explicit precision (no loop counters)
    float tl = FetchPerceptualLuma(center + int2(-1, -1)); float tc = FetchPerceptualLuma(center + int2( 0, -1)); float tr = FetchPerceptualLuma(center + int2( 1, -1));
    float ml = FetchPerceptualLuma(center + int2(-1,  0));                                                         float mr = FetchPerceptualLuma(center + int2( 1,  0));
    float bl = FetchPerceptualLuma(center + int2(-1,  1)); float bc = FetchPerceptualLuma(center + int2( 0,  1)); float br = FetchPerceptualLuma(center + int2( 1,  1));
    float gx = (tr + 2.0*mr + br) - (tl + 2.0*ml + bl);
    float gy = (bl + 2.0*bc + br) - (tl + 2.0*tc + tr);
    return (gx*gx + gy*gy) * 0.0625;
}

float Scharr3x3(int2 center)
{
    float tl = FetchPerceptualLuma(center + int2(-1, -1)); float tc = FetchPerceptualLuma(center + int2( 0, -1)); float tr = FetchPerceptualLuma(center + int2( 1, -1));
    float ml = FetchPerceptualLuma(center + int2(-1,  0));                                                         float mr = FetchPerceptualLuma(center + int2( 1,  0));
    float bl = FetchPerceptualLuma(center + int2(-1,  1)); float bc = FetchPerceptualLuma(center + int2( 0,  1)); float br = FetchPerceptualLuma(center + int2( 1,  1));
    float gx = (3.0*tr + 10.0*mr + 3.0*br) - (3.0*tl + 10.0*ml + 3.0*bl);
    float gy = (3.0*bl + 10.0*bc + 3.0*br) - (3.0*tl + 10.0*tc + 3.0*tr);
    return (gx*gx + gy*gy) * 0.0009765625; // Exact 1/1024
}

float Prewitt3x3(int2 center)
{
    float tl = FetchPerceptualLuma(center + int2(-1, -1)); float tc = FetchPerceptualLuma(center + int2( 0, -1)); float tr = FetchPerceptualLuma(center + int2( 1, -1));
    float ml = FetchPerceptualLuma(center + int2(-1,  0));                                                         float mr = FetchPerceptualLuma(center + int2( 1,  0));
    float bl = FetchPerceptualLuma(center + int2(-1,  1)); float bc = FetchPerceptualLuma(center + int2( 0,  1)); float br = FetchPerceptualLuma(center + int2( 1,  1));
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
    return (sum_gx*sum_gx + sum_gy*sum_gy) * 0.0000152587890625; // Exact 1/65536
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
            float tl = FetchPerceptualLuma(pos + int2(-1, -1)); float tc = FetchPerceptualLuma(pos + int2( 0, -1)); float tr = FetchPerceptualLuma(pos + int2( 1, -1));
            float ml = FetchPerceptualLuma(pos + int2(-1,  0));                                                         float mr = FetchPerceptualLuma(pos + int2( 1,  0));
            float bl = FetchPerceptualLuma(pos + int2(-1,  1)); float bc = FetchPerceptualLuma(pos + int2( 0,  1)); float br = FetchPerceptualLuma(pos + int2( 1,  1));
            float gx = (tr + 2.0*mr + br) - (tl + 2.0*ml + bl);
            float gy = (bl + 2.0*bc + br) - (tl + 2.0*tc + tr);
            
            // Use pre-computed double-precision derived weights
            float w = Structure_Gauss[(wy + 1) * 3 + (wx + 1)];
            Ixx += gx * gx * w;
            Iyy += gy * gy * w;
            Ixy += gx * gy * w;
        }
    }
    float trace = Ixx + Iyy;
    float det = Ixx * Iyy - Ixy * Ixy;
    float disc = TrueSqrt(max(trace * trace - 4.0 * det, 0.0));
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
    switch (method) {
        case 0: return Sobel3x3(center);
        case 1: return Scharr3x3(center);
        case 2: return Prewitt3x3(center);
        case 3: return Sobel5x5(center);
        case 4: return LaplacianOfGaussian(center);
        case 5: return StructureTensor(center);
        default: return Sobel3x3(center);
    }
}

// ==============================================================================
// 8. Bilateral Processing (True Math Loop)
// ==============================================================================

int GetAdaptiveRadius(int2 center, int base_radius, float strength, float sigma_spatial)
{
    float edge = GetEdgeStrength(center, iEdgeDetectionMethod);
    if (bChromaAwareBilateral) edge = max(edge, ChromaEdge(center) * 0.5);
    float scale = smoothstep(0.0, 1.0, edge * (fGradientSensitivity * 0.01));
    float factor = lerp(1.0, lerp(0.90, 0.15, scale), strength);
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

    // Standard Deviation Weighting
    if (mode == 0) metric = saturate(range * 0.166666667);
    else if (mode == 1) metric = saturate(var * 0.5);
    else if (mode == 2) metric = TruePow(max(saturate(range * 0.166666667), FLT_MIN), 1.0 - fVarianceWeight) * TruePow(max(saturate(var * 0.5), FLT_MIN), fVarianceWeight);
    else metric = saturate((log2(1.0 + var) * (1.0 + range * 0.1)) * 0.25);

    return base_strength * lerp(1.0, TruePow(metric, fAdaptiveCurve) * 2.0, fAdaptiveAmount);
}

float3 ProcessPixel(int2 center_pos, float2 texcoord)
{
    float3 color_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, center_pos).rgb);
    float luma_lin = GetLuminanceCS(color_lin);

    // Strict Bit-Exact Passthrough for black
    // If input is absolute zero, output must be absolute zero to avoid FP drift
    if (luma_lin < FLT_MIN) {
        if (iDebugMode == 0) return color_lin; 
    }

    float log2_center = log2(max(luma_lin, FLT_MIN));
    bool is_black = (luma_lin < FLT_MIN) || IsInfVal(log2_center);

    // Allow debug modes to see black pixels
    if (iDebugMode == 0 && is_black) return color_lin;

    int base_radius = (iQualityPreset == 1) ? 48 : iRadius;
    float sigma_s = (iQualityPreset == 1) ? 12.0 : fSigmaSpatial;

    int radius = base_radius;
    if (bAdaptiveRadius && base_radius > 2)
        radius = GetAdaptiveRadius(center_pos, base_radius, fAdaptiveRadiusStrength, sigma_s);

    // Debug Early Exits
    if (iDebugMode == 5) return lerp(float3(0,0,1), float3(1,0,0), float(radius) / float(base_radius));
    if (iDebugMode == 6) { float e = GetEdgeStrength(center_pos, iEdgeDetectionMethod); return float3(e,e,e) * 10.0; }
    if (iDebugMode == 8) { float c = ChromaEdge(center_pos); return float3(c,c,c) * 5.0; }
    if (iDebugMode == 10) return ZONE_COLORS[GetZone(luma_lin / fZoneWhitePoint)];
    if (iDebugMode == 11) return (GetMinComponent(color_lin) < 0.0) ? float3(1,0,1) : float3(0,0.1,0);
    if (iDebugMode == 12) {
        float norm = luma_lin / max(fZoneWhitePoint, FLT_MIN);
        float stops = log2(max(abs(norm), FLT_MIN));
        float t = saturate((stops + 6.0) / 8.0);
        return (norm < 0.0) ? float3(0, t, 0) : float3(t, 0, 0);
    }
    if (iDebugMode == 7) return (luma_lin < FLT_MIN) ? float3(1, 0, 1) : float3(0, 0, 0);

    // Bilateral Constants
    float inv_2_sigma_s_sq = 0.5 / (sigma_s * sigma_s);
    float inv_2_sigma_r_sq = 0.5 / (fSigmaRange * fSigmaRange);
    float inv_2_sigma_c_sq = 0.5 / (fSigmaChroma * fSigmaChroma);
    float cutoff_r_sq = -log(SPATIAL_CUTOFF) / inv_2_sigma_s_sq;

    // Safety Clamp on Loop
    int safe_radius = min((int)TrueSqrt(cutoff_r_sq) + 1, radius);
    int max_r = min(safe_radius, MAX_LOOP_RADIUS);
    float radius_sq_f = float(max_r * max_r);

    float3 center_chroma = float3(0,0,0);
    if (bChromaAwareBilateral && luma_lin > FLT_MIN)
        center_chroma = color_lin / luma_lin - 1.0;

    // Accumulators (Neumaier-Compensated for High Precision Accumulation)
    float2 stats_log = 0.0;
    float2 stats_sq  = 0.0;
    float2 stats_w   = 0.0;
    float min_log = log2_center, max_log = log2_center;
    int count = 0;

    [loop]
    for (int y = -max_r; y <= max_r; ++y)
    {
        float y_sq = float(y * y);
        if (y_sq > cutoff_r_sq) continue;
        int x_limit = (int)TrueSqrt(radius_sq_f - y_sq);

        [loop]
        for (int x = -x_limit; x <= x_limit; ++x)
        {
            float r_sq = float(x * x) + y_sq;
            int2 pos = clamp(center_pos + int2(x, y), int2(0,0), int2(BUFFER_WIDTH-1, BUFFER_HEIGHT-1));
            float3 n_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, pos).rgb);
            float n_luma = max(GetLuminanceCS(n_lin), 0.0);
            float n_log  = log2(max(n_luma, FLT_MIN));

            if (IsInfVal(n_log)) continue;

            float d_luma = log2_center - n_log;
            // True Math: exp() instead of approximations
            float exponent = -(r_sq * inv_2_sigma_s_sq) - (d_luma * d_luma * inv_2_sigma_r_sq);

            if (bChromaAwareBilateral && exponent > -10.0 && n_luma > FLT_MIN) {
                float3 n_chroma = n_lin / n_luma - 1.0;
                float3 d_chroma = center_chroma - n_chroma;
                exponent -= dot(d_chroma, d_chroma) * inv_2_sigma_c_sq;
            }

            float weight = exp(exponent); // IEEE 754 exp

            if (weight > FLT_MIN) {
                // Neumaier Summation
                // Reduces catastrophic cancellation when summing many small weights
                float val = n_log * weight;
                float t = stats_log.x + val;
                stats_log.y += (abs(stats_log.x) >= abs(val)) ? ((stats_log.x - t) + val) : ((val - t) + stats_log.x);
                stats_log.x = t;

                val = n_log * n_log * weight;
                t = stats_sq.x + val;
                stats_sq.y += (abs(stats_sq.x) >= abs(val)) ? ((stats_sq.x - t) + val) : ((val - t) + stats_sq.x);
                stats_sq.x = t;

                t = stats_w.x + weight;
                stats_w.y += (abs(stats_w.x) >= abs(weight)) ? ((stats_w.x - t) + weight) : ((weight - t) + stats_w.x);
                stats_w.x = t;

                min_log = min(min_log, n_log);
                max_log = max(max_log, n_log);
                count++;
            }
        }
    }

    float total_w = stats_w.x + stats_w.y;
    if (total_w < FLT_MIN || count < 1) return color_lin;

    float total_log = stats_log.x + stats_log.y;
    float total_sq  = stats_sq.x  + stats_sq.y;
    float blurred = total_log / total_w;
    float diff = log2_center - blurred;

    float strength = fStrength;
    if (bAdaptiveStrength)
        strength = CalculateAdaptiveStrength(total_log, total_sq, total_w, min_log, max_log, fStrength, iAdaptiveMode);

    float norm_luma = luma_lin / fZoneWhitePoint;
    float minCompNorm = 0.0;
    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    if (space == 2) minCompNorm = GetMinComponent(color_lin) / fZoneWhitePoint;

    strength *= GetZoneProtection(norm_luma, minCompNorm, fShadowProtection, fMidtoneProtection, fHighlightProtection, fNegativeProtection);

    float enhanced_log = log2_center + strength * diff;

    // Post-Loop Debug
    if (iDebugMode == 1) return saturate(log2(total_w + 1.0) * 0.1).xxx;
    if (iDebugMode == 2) { float m = blurred; float v = max(0.0, (total_sq / total_w) - m * m); return float3(v * 2.0, v, 0.0); }
    if (iDebugMode == 3) return float3((max_log - min_log) * 0.2, 0, 0);
    if (iDebugMode == 4) return lerp(float3(0,0,1), float3(1,0,0), saturate(abs(diff) * strength * 2.0));
    if (iDebugMode == 9) { float m = total_log / total_w; float v = max(0.0, (total_sq / total_w) - m * m); float r = max_log - min_log; float e = log2(1.0 + v) * (1.0 + r * 0.1); return float3(e * 0.25, e * 0.125, 0.0); }

    if (IsInfVal(enhanced_log)) return color_lin;

    // Bit-Exact Neutrality Check
    if (abs(strength) < FLT_MIN) return color_lin;

    float enhanced_luma = exp2(enhanced_log);
    float ratio = enhanced_luma / max(luma_lin, FLT_MIN);
    
    // Safety clamp to prevent FP explosion
    ratio = clamp(ratio, RATIO_MIN, RATIO_MAX);

    float3 final = color_lin * ratio;
    if (bGamutMapping) final = SoftClipGamut(final);
    if (any(IsNan3(final)) || any(IsInf3(final))) return color_lin;

    return final;
}

// ==============================================================================
// 9. Shader Entry Point
// ==============================================================================

void PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    // Strict bypass for Bit-Exact output if disabled
    if (fStrength <= 0.0 && iDebugMode == 0) {
        fragColor = tex2Dfetch(SamplerBackBuffer, int2(vpos.xy));
        return;
    }

    float3 final = ProcessPixel(int2(vpos.xy), texcoord);
    float3 encoded = EncodeFromLinear(final);

    #if (BUFFER_COLOR_SPACE <= 1)
        encoded = saturate(encoded);
    #endif

    fragColor = float4(encoded, 1.0);
}

technique BilateralContrast_Reference <
    ui_label = "Bilateral Contrast v8.0.1 (Mastering Edition)";
    ui_tooltip = "MASTERING QUALITY - True Math Processing\n\n"
                 "v8.0.1 Features:\n"
                 "- IEEE 754 True Math (No approximations)\n"
                 "- Exact IEC/SMPTE Standards Constants\n"
                 "- Pre-computed High-Precision Kernels\n"
                 "- Bit-Exact Neutrality Logic\n"
                 "- Stop-Domain Luminance Processing\n\n"
                 "Warning: High ALU usage. Designed for offline rendering.\n"
                 "Requires: DirectX 10+ or OpenGL 4.5+ (uses tex2Dfetch).";
>
{
    pass Main
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_BilateralContrast;
    }
}