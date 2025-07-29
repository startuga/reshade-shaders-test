/**
 * Perceptually Uniform Detail Enhancer v2.5 (Compiler-Fixed, Production Ready)
 *
 * This version corrects the matrix initialization syntax to use the float3x3(...)
 * constructor, as required by the user's compiler.
 *
 * Version 2.5:
 *  - [COMPILER FIX] Reverted matrix initialization to the constructor syntax, e.g.,
 *    `float3x3(f1, f2, ..., f9)`, as the brace-based syntax `{...}` was explicitly
 *    rejected by the compiler with error X3017. This should resolve compilation issues.
 *
 * Version 2.4:
 *  - Fixed a critical bug in the from_oklab color space conversion function.
 */

//----------------------------------------------------------------------------------------------------------------------
// UI Uniforms
//----------------------------------------------------------------------------------------------------------------------
uniform float fStrength <
    ui_type = "slider"; ui_label = "Detail Strength";
    ui_tooltip = "Overall strength of the detail enhancement effect.";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.05;
> = 1.0;

uniform int iRadiusPixels <
    ui_type = "slider"; ui_label = "Filter Radius (Pixels)";
    ui_tooltip = "Radius of the bilateral filter kernel. Larger values capture larger-scale local contrast but are significantly more performance-intensive.";
    ui_min = 1; ui_max = 20; ui_step = 1;
> = 7;

uniform float fSigmaSpatial <
    ui_type = "slider"; ui_label = "Sigma Spatial";
    ui_tooltip = "Spatial sigma for the bilateral filter. Controls influence based on distance.";
    ui_min = 0.1; ui_max = 10.0; ui_step = 0.1;
> = 3.5;

uniform float fSigmaLightness <
    ui_type = "slider"; ui_label = "Sigma Lightness (Log, Range)";
    ui_tooltip = "Lightness sigma for the bilateral filter. Operates on LOG-ENCODED lightness. Smaller values preserve edges more strongly.";
    ui_min = 0.001; ui_max = 2.0; ui_step = 0.001;
> = 0.2;

uniform float fOvershootProtection <
    ui_type = "slider"; ui_label = "Overshoot Protection";
    ui_tooltip = "Limits the maximum enhancement to prevent extreme brights/darks and halos. Lower values provide stronger protection. A value of 1.0 limits deviation to ~1 log stop.";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.05;
> = 1.0;

//----------------------------------------------------------------------------------------------------------------------
// Includes and Global Constants
//----------------------------------------------------------------------------------------------------------------------
#include "ReShade.fxh"

static const float EPSILON_DIV = 1e-7f;
static const float EPSILON_LOG = 1e-6f;

//----------------------------------------------------------------------------------------------------------------------
// Oklab Color Space Conversion Functions
//----------------------------------------------------------------------------------------------------------------------
// Reverting to constructor syntax for matrices as required by the compiler.
static const float3x3 M1 = float3x3(
    0.4121656120, 0.5362752080, 0.0514575653,
    0.2118591070, 0.6807189584, 0.1074065790,
    0.0883097947, 0.2818474174, 0.6302613616
);
static const float3x3 M2 = float3x3(
    0.2104542553,  0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050,  0.4505937099,
    0.0259040371,  0.7827717662, -0.8086757660
);
static const float3x3 M2_inv = float3x3(
    1.0,  0.3963377774,  0.2158037573,
    1.0, -0.1055613458, -0.0638541728,
    1.0, -0.0894841775, -1.2914855480
);
static const float3x3 M1_inv = float3x3(
     4.0767245293, -3.3072168827,  0.2307590544,
    -1.2681437731,  2.6093323231, -0.3411344290,
    -0.0041119855, -0.7034763098,  1.7068625689
);

float3 to_oklab(float3 rgb)
{
    float3 lms = mul(M1, rgb);
    lms = pow(abs(lms), 1.0 / 3.0) * sign(lms);
    return mul(M2, lms);
}

float3 from_oklab(float3 oklab)
{
    float3 lms = mul(M2_inv, oklab);
    lms = pow(abs(lms), 3.0) * sign(lms);
    return mul(M1_inv, lms);
}

//----------------------------------------------------------------------------------------------------------------------
// Pixel Shader
//----------------------------------------------------------------------------------------------------------------------
float4 OklabHDRDetailEnhancerPS(float4 vpos : SV_POSITION, float2 texcoord : TEXCOORD) : SV_TARGET
{
    float3 originalColorRGB = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float3 logOriginalColor = log(originalColorRGB + EPSILON_LOG);
    float3 originalColorOklab = to_oklab(logOriginalColor);
    float originalLightness = originalColorOklab.x;

    // --- Bilateral Filter on Log-Oklab Lightness (L) ---
    float sumWeightedLightness = 0.0;
    float sumWeights = 0.0;
    float twoSigmaSpatialSq = 2.0 * fSigmaSpatial * fSigmaSpatial + EPSILON_DIV;
    float twoSigmaLightnessSq = 2.0 * fSigmaLightness * fSigmaLightness + EPSILON_DIV;

    for (int y = -iRadiusPixels; y <= iRadiusPixels; ++y)
    {
        for (int x = -iRadiusPixels; x <= iRadiusPixels; ++x)
        {
            float2 pixelOffset = float2(x, y);
            float2 sampleTexcoord = texcoord + pixelOffset * ReShade::PixelSize;
            
            float3 neighborColorRGB = tex2D(ReShade::BackBuffer, sampleTexcoord).rgb;
            float neighborLightness = to_oklab(log(neighborColorRGB + EPSILON_LOG)).x;

            float spatialWeight = exp(-dot(pixelOffset, pixelOffset) / twoSigmaSpatialSq);
            float lightnessWeight = exp(-pow(originalLightness - neighborLightness, 2) / twoSigmaLightnessSq);
            float totalWeight = spatialWeight * lightnessWeight;

            sumWeightedLightness += neighborLightness * totalWeight;
            sumWeights += totalWeight;
        }
    }
    float blurredLightness = (sumWeights > EPSILON_DIV) ? (sumWeightedLightness / sumWeights) : originalLightness;

    // --- Detail Extraction ---
    float detailLightness = originalLightness - blurredLightness;

    // --- Controlled Reconstruction ---
    float enhancedDetail = detailLightness * fStrength;
    float deviation = detailLightness + enhancedDetail;
    float safe_deviation = fOvershootProtection * tanh(deviation / (fOvershootProtection + EPSILON_DIV));
    float newLuma = blurredLightness + safe_deviation;
    
    // --- Final Color Conversion ---
    float3 finalColorOklab = float3(newLuma, originalColorOklab.y, originalColorOklab.z);
    float3 finalLogRGB = from_oklab(finalColorOklab);
    float3 finalColorRGB = exp(finalLogRGB) - EPSILON_LOG;
    
    return float4(max(finalColorRGB, 0.0), tex2D(ReShade::BackBuffer, texcoord).a);
}

//----------------------------------------------------------------------------------------------------------------------
// Technique Definition
//----------------------------------------------------------------------------------------------------------------------
technique OklabHDRDetailEnhancer <
    ui_name = "Perceptual Detail Enhancer (Oklab, HDR-Safe)";
    ui_tooltip = "State-of-the-art detail enhancement using a fully HDR-safe, color-accurate pipeline.\n"
                 "V2.5 - Fixed matrix syntax for compiler compatibility.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = OklabHDRDetailEnhancerPS;
    }
}