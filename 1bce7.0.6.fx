/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement - REFERENCE EDITION
 *
 * Design Philosophy: ZERO COMPROMISES
 * - Maximum numerical precision at every stage
 * - Physically accurate color science
 * - Research-grade algorithms
 * - Performance optimizations that preserve quality
 *
 * Version: 7.0.6 (Reference / Polished) - ReShade 6.x Compatible
 * Author: startuga
 * 
 * Changelog v7.0.6:
 * - FIX: Restored missing Debug Modes (Range, Entropy)
 * - PERF: Replaced inner-loop pow() with multiplication
 * - PERF: Converted Bit Depth logic to preprocessor
 * - FIX: Consistent Zone Map normalization
 * 
 * Changelog v7.0.5:
 * - PERF: Fully vectorized Color Science
 * - PERF: Optimized Neumaier & Loops
 */

#include "ReShade.fxh"

// ==============================================================================
// Texture Configuration
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
// Automatic Color Space Detection
// ==============================================================================

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
// Reference Timing
// ==============================================================================

uniform float frame_time < source = "frametime"; >;
uniform int frame_count < source = "framecount"; >;

// ==============================================================================
// UI Configuration
// ==============================================================================

uniform int iQualityPreset <
    ui_type = "combo";
    ui_label = "Quality Preset";
    ui_items = "Custom\0Reference (Maximum)\0";
    ui_category = "Presets";
> = 1;

uniform float fStrength <
    ui_type = "slider";
    ui_label = "Contrast Strength";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.001;
    ui_category = "Core Settings";
> = 2.5;

uniform float fMidtoneProtection <
    ui_type = "slider";
    ui_label = "Midtone Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Core Settings";
> = 0.3;

uniform float fHighlightProtection <
    ui_type = "slider";
    ui_label = "Highlight Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Core Settings";
> = 0.5;

uniform float fShadowProtection <
    ui_type = "slider";
    ui_label = "Shadow Protection";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Core Settings";
> = 0.4;

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
    ui_label = "Adaptive Strength Amount";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Adaptive Processing";
> = 0.6;

uniform float fAdaptiveCurve <
    ui_type = "slider";
    ui_label = "Adaptive Response Curve";
    ui_min = 0.1; ui_max = 4.0; ui_step = 0.01;
    ui_category = "Adaptive Processing";
> = 1.8;

uniform int iRadius <
    ui_type = "slider";
    ui_label = "Filter Radius";
    ui_tooltip = "Maximum sampling radius.\nReference preset uses 48.";
    ui_min = 1;
    ui_max = 64; 
    ui_category = "Filter Parameters";
> = 32;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_min = 0.1; ui_max = 32.0; ui_step = 0.01;
    ui_category = "Filter Parameters";
> = 12.0;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Range Sigma (Stops)";
    ui_min = 0.01; ui_max = 4.0; ui_step = 0.001;
    ui_category = "Filter Parameters";
> = 0.35;

uniform float fSigmaChroma <
    ui_type = "slider";
    ui_label = "Chroma Sigma";
    ui_tooltip = "Controls color similarity weighting.";
    ui_min = 0.01; ui_max = 1.0; ui_step = 0.001;
    ui_category = "Filter Parameters";
> = 0.15;

uniform bool bChromaAwareBilateral <
    ui_label = "Enable Chroma-Aware Filtering";
    ui_tooltip = "Uses full color information for edge detection.";
    ui_category = "Filter Parameters";
> = true;

uniform bool bAdaptiveRadius <
    ui_label = "Enable Adaptive Radius";
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
    ui_min = 10.0; ui_max = 500.0; ui_step = 5.0;
    ui_category = "Advanced Tuning";
> = 150.0;

uniform float fVarianceWeight <
    ui_type = "slider";
    ui_label = "Variance Weight (Hybrid Mode)";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_category = "Advanced Tuning";
> = 0.65;

uniform bool bEnableDithering <
    ui_label = "Enable Output Dithering";
    ui_category = "Output Quality";
> = true;

uniform int iDitherMethod <
    ui_type = "combo";
    ui_label = "Dither Method";
    ui_items = "Triangular (Fast)\0Blue Noise Approximation\0Bayer 8x8\0Temporal Blue Noise\0";
    ui_category = "Output Quality";
> = 3;

uniform float fDitherStrength <
    ui_type = "slider";
    ui_label = "Dither Strength";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_category = "Output Quality";
> = 1.0;

uniform bool bGamutMapping <
    ui_label = "Enable Gamut Mapping";
    ui_category = "Output Quality";
> = true;

uniform int iDebugMode <
    ui_type = "combo";
    ui_label = "Debug Visualization";
    ui_items = "Off\0Weights\0Variance\0Dynamic Range\0Enhancement Map\0Adaptive Radius\0Edge Detection\0Black Pixels\0Chroma Edges\0Entropy\0Zone Map\0";
    ui_category = "Debug";
> = 0;

// ==============================================================================
// Constants
// ==============================================================================

static const float PI = 3.14159265358979323846;
static const float EPSILON = 1e-10;
static const float SAFE_LOG_MIN = 1e-10;
static const float WEIGHT_THRESHOLD = 1e-12;
static const float SPATIAL_CUTOFF = 0.001; 
static const float RATIO_MIN = 0.0625;
static const float RATIO_MAX = 16.0;
static const int MAX_LOOP_RADIUS = 64;

// Zone system
static const float ZONE_I = 0.032;
static const float ZONE_II = 0.063;
static const float ZONE_III = 0.125;
static const float ZONE_IV = 0.25;
static const float ZONE_V = 0.5;
static const float ZONE_VI = 0.707;
static const float ZONE_VII = 0.84;
static const float ZONE_VIII = 0.94;
static const float ZONE_IX = 0.98;
static const float ZONE_X = 1.0;

// Luminance coefficients
static const float3 Luma709 = float3(0.2126729, 0.7151522, 0.0721750);
static const float3 Luma2020 = float3(0.2627002, 0.6779981, 0.0593017);

// PQ Constants
static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;
static const float PQ_PEAK_LUMINANCE = 10000.0;

// Debug colors
static const float3 ZONE_COLORS[11] = {
    float3(0, 0, 0), float3(0.1, 0, 0.1), float3(0.2, 0, 0.3), float3(0.3, 0, 0.5),
    float3(0.2, 0.2, 0.8), float3(0.5, 0.5, 0.5), float3(0.8, 0.8, 0.2),
    float3(1.0, 0.8, 0.3), float3(1.0, 0.6, 0.4), float3(1.0, 0.9, 0.8), float3(1, 1, 1)
};

// ==============================================================================
// Optimized Vector Utilities
// ==============================================================================

float SafeSqrt(float x) { return sqrt(max(x, 0.0)); }
float SafePow(float base, float exponent) { return pow(max(base, EPSILON), exponent); }
float3 SafePow3(float3 base, float exponent) { return pow(max(base, EPSILON), exponent); }

bool IsNanVal(float x) { return (asuint(x) & 0x7F800000) == 0x7F800000 && (asuint(x) & 0x7FFFFF) != 0; }
bool IsInfVal(float x) { return (asuint(x) & 0x7FFFFFFF) == 0x7F800000; }
bool3 IsNan3(float3 v) { return bool3(IsNanVal(v.x), IsNanVal(v.y), IsNanVal(v.z)); }
bool3 IsInf3(float3 v) { return bool3(IsInfVal(v.x), IsInfVal(v.y), IsInfVal(v.z)); }

// ==============================================================================
// Vectorized Color Science
// ==============================================================================

float3 sRGB_EOTF(float3 V)
{
    float3 sign_V = sign(V);
    float3 abs_V = abs(V);
    float3 linear_lo = abs_V / 12.92;
    float3 linear_hi = SafePow3((abs_V + 0.055) / 1.055, 2.4);
    float3 selector = step(0.04045, abs_V);
    return sign_V * lerp(linear_lo, linear_hi, selector);
}

float3 PQ_EOTF(float3 N)
{
    float3 N_safe = max(N, 0.0);
    float3 Np = SafePow3(N_safe, 1.0 / PQ_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, EPSILON);
    return SafePow3(num / den, 1.0 / PQ_M1) * PQ_PEAK_LUMINANCE;
}

float3 sRGB_OETF(float3 L)
{
    float3 sign_L = sign(L);
    float3 abs_L = abs(L);
    float3 encoded_lo = abs_L * 12.92;
    float3 encoded_hi = 1.055 * SafePow3(abs_L, 1.0/2.4) - 0.055;
    float3 selector = step(0.0031308, abs_L);
    return sign_L * lerp(encoded_lo, encoded_hi, selector);
}

float3 PQ_InverseEOTF(float3 L)
{
    float3 L_safe = max(L, 0.0) / PQ_PEAK_LUMINANCE;
    float3 Lp = SafePow3(L_safe, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    return SafePow3(num / den, PQ_M2);
}

float3 DecodeToLinear(float3 encoded)
{
    #if (BUFFER_COLOR_SPACE == 3)
        return PQ_EOTF(encoded);
    #elif (BUFFER_COLOR_SPACE == 2)
        return encoded * 80.0;
    #else
        return sRGB_EOTF(encoded) * 80.0;
    #endif
}

float3 EncodeFromLinear(float3 lin)
{
    #if (BUFFER_COLOR_SPACE == 3)
        return PQ_InverseEOTF(lin);
    #elif (BUFFER_COLOR_SPACE == 2)
        return lin / 80.0;
    #else
        return sRGB_OETF(lin / 80.0);
    #endif
}

// Inlined Luma for hot path
#if (BUFFER_COLOR_SPACE >= 2)
    #define GET_LUMINANCE(c) dot((c), Luma2020)
#else
    #define GET_LUMINANCE(c) dot((c), Luma709)
#endif

float3 SoftClipGamut(float3 lin)
{
    float minComponent = min(min(lin.r, lin.g), lin.b);
    if (minComponent < 0.0)
    {
        float luma = GET_LUMINANCE(lin);
        float3 chroma = lin - luma;
        float scale = luma / (luma - minComponent + EPSILON);
        lin = luma + chroma * min(scale, 1.0);
    }
    return lin;
}

int GetZone(float normalizedLuma)
{
    if (normalizedLuma < ZONE_I) return 0;
    if (normalizedLuma < ZONE_II) return 1;
    if (normalizedLuma < ZONE_III) return 2;
    if (normalizedLuma < ZONE_IV) return 3;
    if (normalizedLuma < ZONE_V) return 4;
    if (normalizedLuma < ZONE_VI) return 5;
    if (normalizedLuma < ZONE_VII) return 6;
    if (normalizedLuma < ZONE_VIII) return 7;
    if (normalizedLuma < ZONE_IX) return 8;
    if (normalizedLuma < ZONE_X) return 9;
    return 10;
}

float GetZoneProtection(float normalizedLuma, float shadowProt, float midProt, float highProt)
{
    float shadowWeight = smoothstep(ZONE_IV, ZONE_II, normalizedLuma);
    float midWeight = saturate(1.0 - abs(normalizedLuma - ZONE_V) * 4.0);
    float highWeight = smoothstep(ZONE_VI, ZONE_VIII, normalizedLuma);
    return 1.0 - saturate(shadowWeight * shadowProt + midWeight * midProt + highWeight * highProt);
}

// ==============================================================================
// Edge Detection
// ==============================================================================

float3 FetchLinear(int2 pos)
{
    pos = clamp(pos, int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
    return DecodeToLinear(tex2Dfetch(SamplerBackBuffer, pos).rgb);
}

float FetchPerceptualLuma(int2 pos)
{
    pos = clamp(pos, int2(0, 0), int2(BUFFER_WIDTH - 1, BUFFER_HEIGHT - 1));
    float3 lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, pos).rgb);
    float luma = max(GET_LUMINANCE(lin), 0.0);
    return (log2(max(luma, 1e-6)) + 20.0) * 0.06;
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
    
    return (gx*gx + gy*gy) * 0.0009765625; 
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
    static const float Gx[25] = { -1, -2, 0, 2, 1, -4, -8, 0, 8, 4, -6, -12, 0, 12, 6, -4, -8, 0, 8, 4, -1, -2, 0, 2, 1 };
    static const float Gy[25] = { -1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0, 2, 8, 12, 8, 2, 1, 4, 6, 4, 1 };
    float sum_gx = 0.0; float sum_gy = 0.0;
    
    [unroll]
    for (int y = -2; y <= 2; y++) {
        [unroll]
        for (int x = -2; x <= 2; x++) {
            float luma = FetchPerceptualLuma(center + int2(x, y));
            int idx = (y + 2) * 5 + (x + 2);
            sum_gx += luma * Gx[idx];
            sum_gy += luma * Gy[idx];
        }
    }
    return (sum_gx*sum_gx + sum_gy*sum_gy) * 0.0000152587890625;
}

float LaplacianOfGaussian(int2 center)
{
    static const float LoG[25] = { 0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0 };
    float response = 0.0;
    [unroll]
    for (int y = -2; y <= 2; y++) {
        [unroll]
        for (int x = -2; x <= 2; x++) {
            float luma = FetchPerceptualLuma(center + int2(x, y));
            response += luma * LoG[(y + 2) * 5 + (x + 2)];
        }
    }
    return response * response;
}

float StructureTensor(int2 center)
{
    float Ixx = 0.0, Iyy = 0.0, Ixy = 0.0;
    static const float gauss[9] = { 0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625 };
    [unroll]
    for (int wy = -1; wy <= 1; wy++) {
        [unroll]
        for (int wx = -1; wx <= 1; wx++) {
            int2 pos = center + int2(wx, wy);
            float tl = FetchPerceptualLuma(pos + int2(-1, -1)); float tc = FetchPerceptualLuma(pos + int2( 0, -1));
            float tr = FetchPerceptualLuma(pos + int2( 1, -1)); float ml = FetchPerceptualLuma(pos + int2(-1,  0));
            float mr = FetchPerceptualLuma(pos + int2( 1,  0)); float bl = FetchPerceptualLuma(pos + int2(-1,  1));
            float bc = FetchPerceptualLuma(pos + int2( 0,  1)); float br = FetchPerceptualLuma(pos + int2( 1,  1));
            float gx = (tr + 2.0*mr + br) - (tl + 2.0*ml + bl);
            float gy = (bl + 2.0*bc + br) - (tl + 2.0*tc + tr);
            float w = gauss[(wy + 1) * 3 + (wx + 1)];
            Ixx += gx * gx * w; Iyy += gy * gy * w; Ixy += gx * gy * w;
        }
    }
    float trace = Ixx + Iyy;
    float det = Ixx * Iyy - Ixy * Ixy;
    float disc = SafeSqrt(max(trace * trace - 4.0 * det, 0.0));
    float lambda1 = (trace + disc) * 0.5;
    float lambda2 = (trace - disc) * 0.5;
    float coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + EPSILON);
    return SafeSqrt(lambda1) * (1.0 + coherence) * 0.5;
}

float ChromaEdge(int2 center)
{
    float3 c = FetchLinear(center);
    float luma = max(GET_LUMINANCE(c), EPSILON);
    float3 chroma = c / luma;
    float maxChromaDiff = 0.0;
    [unroll]
    for (int y = -1; y <= 1; y++) {
        [unroll]
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            float3 nc = FetchLinear(center + int2(x, y));
            float nLuma = max(GET_LUMINANCE(nc), EPSILON);
            float3 diff = chroma - (nc / nLuma);
            maxChromaDiff = max(maxChromaDiff, dot(diff, diff));
        }
    }
    return SafeSqrt(maxChromaDiff);
}

float GetEdgeStrength(int2 center, int method)
{
    switch(method) {
        case 0:  return Sobel3x3(center);
        case 1:  return Scharr3x3(center);
        case 2:  return Prewitt3x3(center);
        case 3:  return Sobel5x5(center);
        case 4:  return LaplacianOfGaussian(center);
        case 5:  return StructureTensor(center);
        default: return Sobel3x3(center);
    }
}

// ==============================================================================
// Dithering
// ==============================================================================

float3 TriangularDither(float2 uv, float time_seed) {
    float2 seed = uv + frac(time_seed);
    float3 noise = float3(frac(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453),
                          frac(sin(dot(seed, float2(12.9898, 78.233) + 0.5)) * 43758.5453),
                          frac(sin(dot(seed, float2(12.9898, 78.233) + 1.0)) * 43758.5453));
    noise = noise * 2.0 - 1.0;
    return sign(noise) * (1.0 - sqrt(1.0 - abs(noise)));
}

float InterleavedGradientNoise(float2 pos) {
    return frac(52.9829189 * frac(dot(pos, float2(0.06711056, 0.00583715))));
}

float3 BlueNoiseApprox(float2 pos, float time_seed) {
    float2 offset = float2(InterleavedGradientNoise(pos + float2(0.0, time_seed)),
                           InterleavedGradientNoise(pos + float2(time_seed, 0.0)));
    return float3(InterleavedGradientNoise(pos + offset * 100.0),
                  InterleavedGradientNoise(pos + offset * 100.0 + float2(47.5, 13.2)),
                  InterleavedGradientNoise(pos + offset * 100.0 + float2(23.1, 89.7))) * 2.0 - 1.0;
}

float Bayer8x8(int2 pos) {
    static const float bayer[64] = { 0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21 };
    return bayer[(pos.y & 7) * 8 + (pos.x & 7)] * 0.03125 - 1.0;
}

float3 TemporalBlueNoise(float2 pos, int frame) {
    float angle = float(frame & 63) * PI * 0.0625;
    float2 rotated = float2(pos.x * cos(angle) - pos.y * sin(angle), pos.x * sin(angle) + pos.y * cos(angle));
    float3 noise = BlueNoiseApprox(rotated, float(frame) * 0.1);
    float temporal = frac(float(frame) * 1.6180339887 - 1.0);
    return frac(noise + temporal) * 2.0 - 1.0;
}

float3 ApplyDither(float3 color, float2 uv, int2 pos, float bit_depth, int method, float strength) {
    // REMOVED: if (bit_depth >= 16.0) return color;
    float time_seed = frac(frame_time * 0.001);
    float3 noise;
    switch(method) {
        case 0: noise = TriangularDither(uv, time_seed); break;
        case 1: noise = BlueNoiseApprox(float2(pos), time_seed); break;
        case 2: noise = float3(Bayer8x8(pos), Bayer8x8(pos + int2(3, 7)), Bayer8x8(pos + int2(5, 2))); break;
        case 3: noise = TemporalBlueNoise(float2(pos), frame_count); break;
        default: noise = TriangularDither(uv, time_seed); break;
    }
    return color + noise * (1.0 / (exp2(bit_depth) - 1.0)) * strength;
}

// ==============================================================================
// Optimized Main Processing
// ==============================================================================

int GetAdaptiveRadius(int2 center, int base_radius, float strength, float sigma_spatial)
{
    float edge = GetEdgeStrength(center, iEdgeDetectionMethod);
    if (bChromaAwareBilateral) edge = max(edge, ChromaEdge(center) * 0.5);
    
    float scale = smoothstep(0.0, 1.0, edge * (fGradientSensitivity * 0.01));
    float factor = lerp(1.0, lerp(0.75, 0.05, scale), strength);
    
    int sigma_max = (int)(sigma_spatial * 3.0 + 0.5);
    return clamp(min((int)(base_radius * factor + 0.5), sigma_max), 1, base_radius);
}

float CalculateAdaptiveStrength(float sum_log, float sum_log_sq, float sum_weight, float min_log, float max_log, float base_strength, int mode)
{
    if (sum_weight < WEIGHT_THRESHOLD) return base_strength;
    
    float inv_weight = 1.0 / sum_weight;
    float range = max_log - min_log;
    float mean = sum_log * inv_weight;
    float var = max(0.0, sum_log_sq * inv_weight - mean * mean);
    float metric;
    
    switch(mode) {
        case 0: metric = saturate(range * 0.166666667); break;
        case 1: metric = saturate(var * 0.5); break;
        case 2: metric = SafePow(max(saturate(range * 0.166666667), EPSILON), 1.0 - fVarianceWeight) * SafePow(max(saturate(var * 0.5), EPSILON), fVarianceWeight); break;
        case 3: default: metric = saturate((log2(1.0 + var) * (1.0 + range * 0.1)) * 0.25); break;
    }
    return base_strength * lerp(1.0, SafePow(metric, fAdaptiveCurve) * 2.0, fAdaptiveAmount);
}

float3 ProcessPixel(int2 center_pos, float2 texcoord)
{
    float3 color_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, center_pos).rgb);
    float luma_safe = max(GET_LUMINANCE(color_lin), 0.0);
    float log2_center = log2(max(luma_safe, SAFE_LOG_MIN));
    
    if (IsInfVal(log2_center) || luma_safe < EPSILON) return (iDebugMode == 7) ? float3(1, 0, 1) : color_lin;
    
    int base_radius = (iQualityPreset == 1) ? 48 : iRadius;
    float sigma_s = (iQualityPreset == 1) ? 12.0 : fSigmaSpatial;
    
    int radius = base_radius;
    if (bAdaptiveRadius && base_radius > 2)
        radius = GetAdaptiveRadius(center_pos, base_radius, fAdaptiveRadiusStrength, sigma_s);
    
    // Debug modes early exit
    if (iDebugMode == 5) return lerp(float3(0,0,1), float3(1,0,0), float(radius)/float(base_radius));
    if (iDebugMode == 6) { float e = GetEdgeStrength(center_pos, iEdgeDetectionMethod); return float3(e,e,e)*10.0; }
    if (iDebugMode == 8) { float c = ChromaEdge(center_pos); return float3(c,c,c)*5.0; }
    if (iDebugMode == 10) {
        #if (BUFFER_COLOR_SPACE == 3)
            float normalized = saturate(luma_safe * 0.0001);
        #elif (BUFFER_COLOR_SPACE == 2)
            float normalized = saturate(luma_safe * 0.0001);
        #else
            float normalized = saturate(luma_safe * 0.0125);
        #endif
        return ZONE_COLORS[GetZone(normalized)];
    }
    
    float inv_2_sigma_s_sq = 0.5 / (sigma_s * sigma_s);
    float inv_2_sigma_r_sq = 0.5 / (fSigmaRange * fSigmaRange);
    float inv_2_sigma_c_sq = 0.5 / (fSigmaChroma * fSigmaChroma);
    
    // Spatial cutoff optimization
    float cutoff_r_sq = -log(SPATIAL_CUTOFF) / inv_2_sigma_s_sq;
    int safe_radius = min((int)SafeSqrt(cutoff_r_sq) + 1, radius);
    int max_r = min(safe_radius, MAX_LOOP_RADIUS);
    float radius_sq_f = float(max_r * max_r);
    
    float3 center_chroma = float3(0,0,0);
    if (bChromaAwareBilateral && luma_safe > EPSILON) center_chroma = color_lin / luma_safe - 1.0;
    
    float sum_log = 0.0, sum_log_c = 0.0;
    float sum_log_sq = 0.0, sum_log_sq_c = 0.0;
    float sum_weight = 0.0, sum_weight_c = 0.0;
    float min_log = log2_center, max_log = log2_center;
    int count = 0;
    
    [loop]
    for (int y = -max_r; y <= max_r; ++y)
    {
        float y_sq = float(y * y);
        if (y_sq > cutoff_r_sq) continue;
        int x_limit = (int)SafeSqrt(radius_sq_f - y_sq);
        
        [loop]
        for (int x = -x_limit; x <= x_limit; ++x)
        {
            float r_sq = float(x * x) + y_sq;
            int2 pos = clamp(center_pos + int2(x, y), int2(0,0), int2(BUFFER_WIDTH-1, BUFFER_HEIGHT-1));
            
            float3 n_lin = DecodeToLinear(tex2Dfetch(SamplerBackBuffer, pos).rgb);
            float n_luma = max(GET_LUMINANCE(n_lin), 0.0);
            float n_log = log2(max(n_luma, SAFE_LOG_MIN));
            
            if (IsInfVal(n_log)) continue;
            
            float d_luma = log2_center - n_log;
            float weight = exp(-r_sq * inv_2_sigma_s_sq) * exp(-(d_luma * d_luma) * inv_2_sigma_r_sq);
            
            if (bChromaAwareBilateral && weight > 1e-5 && n_luma > EPSILON) {
                float3 n_chroma = n_lin / n_luma - 1.0;
                float3 d_chroma = center_chroma - n_chroma;
                float dist = dot(d_chroma, d_chroma);
                weight *= exp(-dist * inv_2_sigma_c_sq);
            }
            
            if (weight > WEIGHT_THRESHOLD) {
                float val = n_log * weight;
                float t = sum_log + val;
                sum_log_c += (abs(sum_log) >= abs(val)) ? ((sum_log - t) + val) : ((val - t) + sum_log);
                sum_log = t;
                
                val = n_log * n_log * weight;
                t = sum_log_sq + val;
                sum_log_sq_c += (abs(sum_log_sq) >= abs(val)) ? ((sum_log_sq - t) + val) : ((val - t) + sum_log_sq);
                sum_log_sq = t;
                
                t = sum_weight + weight;
                sum_weight_c += (abs(sum_weight) >= abs(weight)) ? ((sum_weight - t) + weight) : ((weight - t) + sum_weight);
                sum_weight = t;
                
                min_log = min(min_log, n_log);
                max_log = max(max_log, n_log);
                count++;
            }
        }
    }
    
    float total_weight = sum_weight + sum_weight_c;
    if (total_weight < WEIGHT_THRESHOLD || count < 1) return color_lin;
    
    float total_log = sum_log + sum_log_c;
    float total_sq = sum_log_sq + sum_log_sq_c;
    
    float blurred = total_log / total_weight;
    float diff = log2_center - blurred;
    
    float strength = fStrength;
    if (bAdaptiveStrength) strength = CalculateAdaptiveStrength(total_log, total_sq, total_weight, min_log, max_log, fStrength, iAdaptiveMode);
    
    #if (BUFFER_COLOR_SPACE == 3)
        strength *= GetZoneProtection(saturate(luma_safe * 0.0001), fShadowProtection, fMidtoneProtection, fHighlightProtection);
    #elif (BUFFER_COLOR_SPACE == 2)
        strength *= GetZoneProtection(saturate(luma_safe * 0.0001), fShadowProtection, fMidtoneProtection, fHighlightProtection);
    #else
        strength *= GetZoneProtection(saturate(luma_safe * 0.0125), fShadowProtection, fMidtoneProtection, fHighlightProtection);
    #endif
    
    float enhanced = log2_center + strength * diff;
    
    if (iDebugMode == 1) return saturate(log2(total_weight + 1.0) * 0.1).xxx;
    if (iDebugMode == 2) { float i = 1.0/total_weight; float v = max(0.0, total_sq*i - (total_log*i)*(total_log*i)); return float3(v*5.0, v*2.0, 0.0); }
    if (iDebugMode == 3) { float range = max_log - min_log; return float3(range * 0.2, range * 0.1, 0.0); }
    if (iDebugMode == 4) return lerp(float3(0,0,1), float3(1,0,0), saturate(abs(diff) * strength * 2.0));
    if (iDebugMode == 9) {
        float inv_w = 1.0 / total_weight;
        float mean = total_log * inv_w;
        float var = max(0.0, total_sq * inv_w - mean * mean);
        float range = max_log - min_log;
        float entropy = log2(1.0 + var) * (1.0 + range * 0.1);
        return float3(entropy * 0.25, entropy * 0.125, 0.0);
    }
    
    if (IsInfVal(enhanced)) return color_lin;
    
    float ratio = clamp(exp2(enhanced) / max(luma_safe, EPSILON), RATIO_MIN, RATIO_MAX);
    float3 final = color_lin * ratio;
    
    if (bGamutMapping) final = SoftClipGamut(final);
    return (any(IsNan3(final)) || any(IsInf3(final))) ? color_lin : final;
}

// ==============================================================================
// Entry Point
// ==============================================================================

void PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    if (fStrength <= 0.0 && iDebugMode == 0) {
        fragColor = tex2Dfetch(SamplerBackBuffer, int2(vpos.xy));
        return;
    }
    
    float3 final = ProcessPixel(int2(vpos.xy), texcoord);
    float3 encoded = EncodeFromLinear(final);
    
    if (bEnableDithering) {
        float bd = 8.0;
        #ifdef BUFFER_COLOR_BIT_DEPTH
            bd = float(BUFFER_COLOR_BIT_DEPTH);
        #else
            #if (BUFFER_COLOR_SPACE == 2)
                bd = 16.0;
            #elif (BUFFER_COLOR_SPACE == 3)
                bd = 10.0;
            #endif
        #endif
        encoded = ApplyDither(encoded, texcoord, int2(vpos.xy), bd, iDitherMethod, fDitherStrength);
    }
    
    #if (BUFFER_COLOR_SPACE <= 1)
        encoded = saturate(encoded);
    #endif
    
    fragColor = float4(encoded, 1.0);
}

technique BilateralContrast_Reference <
    ui_label = "Bilateral Contrast v7.0.6 (Polished Reference)";
    ui_tooltip = "REFERENCE QUALITY - Polished & Optimized\n\n"
                 "v7.0.6 Changes:\n"
                 "- Fixed missing Debug Modes (Range, Entropy)\n"
                 "- Optimized inner loop math (pow -> mult)\n"
                 "- Zero-overhead bit depth check\n"
                 "- Consistent Zone Map visualization";
>
{
    pass { VertexShader = PostProcessVS; PixelShader = PS_BilateralContrast; }
}