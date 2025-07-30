/**
 * Universal Detail Enhancer v3.1 (Production Ready, Adaptive Sigma)
 *
 * This definitive version incorporates an Adaptive Luminance Sigma, making the shader
 * intuitive and effective to use with High Dynamic Range (HDR) content.
 *
 * Version 3.1:
 *  - [FINAL USABILITY FIX] Implemented an Adaptive Luminance Sigma. The `fSigmaLuma`
 *    parameter is now proportional to the pixel's own brightness. This ensures the
 *    filter behaves consistently and predictably across the entire dynamic range,
 *    from dark shadows to bright highlights.
 *
 * Version 3.0:
 *  - Replaced Oklab pipeline with a gamut-agnostic Luminance Ratio method to ensure
 *    full compatibility with scRGB and other wide-gamut color spaces.
 */

//----------------------------------------------------------------------------------------------------------------------
// UI Uniforms
//----------------------------------------------------------------------------------------------------------------------
uniform float fStrength <
    ui_type = "slider"; ui_label = "Detail Strength";
    ui_tooltip = "Strength of the detail enhancement effect.";
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

uniform float fSigmaLuma <
    ui_type = "slider"; ui_label = "Proportional Luminance Sigma";
    ui_tooltip = "The tolerance for luminance differences, as a proportion of the pixel's own brightness. E.g., 0.1 = 10% tolerance. This allows the filter to adapt to HDR values.";
    ui_min = 0.01; ui_max = 2.0; ui_step = 0.01;
> = 0.2;

uniform float fDetailCompression <
    ui_type = "slider"; ui_label = "Detail Compression";
    ui_tooltip = "Compresses the extracted detail signal to prevent harshness and artifacts in high-contrast (HDR) scenes. Higher values mean more compression.";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.05;
> = 1.0;

//----------------------------------------------------------------------------------------------------------------------
// Includes and Global Constants
//----------------------------------------------------------------------------------------------------------------------
#include "ReShade.fxh"

static const float3 LUMA_WEIGHTS = float3(0.2126, 0.7152, 0.0722);
static const float EPSILON = 1e-6f;

//----------------------------------------------------------------------------------------------------------------------
// Pixel Shader
//----------------------------------------------------------------------------------------------------------------------
float4 UniversalDetailEnhancerPS(float4 vpos : SV_POSITION, float2 texcoord : TEXCOORD) : SV_TARGET
{
    float3 originalColor = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float originalLuma = dot(originalColor, LUMA_WEIGHTS);

    // --- Adaptive Bilateral Filter on Linear Luminance ---
    // The key improvement: sigma is scaled by the pixel's own luminance.
    // This makes the filter's behavior consistent across the entire dynamic range.
    float effectiveLumaSigma = fSigmaLuma * originalLuma + EPSILON; // Add epsilon to avoid zero sigma for black pixels.
    float twoSigmaSpatialSq = 2.0 * fSigmaSpatial * fSigmaSpatial + EPSILON;
    float twoEffectiveLumaSigmaSq = 2.0 * effectiveLumaSigma * effectiveLumaSigma;

    float sumWeightedLuma = 0.0;
    float sumWeights = 0.0;

    for (int y = -iRadiusPixels; y <= iRadiusPixels; ++y)
    {
        for (int x = -iRadiusPixels; x <= iRadiusPixels; ++x)
        {
            float2 pixelOffset = float2(x, y);
            float2 sampleTexcoord = texcoord + pixelOffset * ReShade::PixelSize;
            
            float3 neighborColor = tex2D(ReShade::BackBuffer, sampleTexcoord).rgb;
            float neighborLuma = dot(neighborColor, LUMA_WEIGHTS);

            float spatialWeight = exp(-dot(pixelOffset, pixelOffset) / twoSigmaSpatialSq);
            float lumaWeight = exp(-pow(originalLuma - neighborLuma, 2) / twoEffectiveLumaSigmaSq);
            float totalWeight = spatialWeight * lumaWeight;

            sumWeightedLuma += neighborLuma * totalWeight;
            sumWeights += totalWeight;
        }
    }
    float blurredLuma = (sumWeights > EPSILON) ? (sumWeightedLuma / sumWeights) : originalLuma;

    // --- Detail Extraction and Compression ---
    float detailLuma = originalLuma - blurredLuma;
    float compressedDetail = fDetailCompression * tanh(detailLuma / (fDetailCompression + EPSILON));
    float enhancedDetail = compressedDetail * fStrength;

    // --- Luminance and Color Reconstruction ---
    float newLuma = originalLuma + enhancedDetail;
    newLuma = max(newLuma, 0.0);

    // --- Gamut-Agnostic Luminance Ratio Method ---
    float3 finalColor = originalColor;
    if (abs(originalLuma) > EPSILON)
    {
        finalColor = originalColor * (newLuma / originalLuma);
    }
    else
    {
        finalColor = float3(newLuma, newLuma, newLuma);
    }

    // No final clamp to preserve wide-gamut (scRGB) colors.
    return float4(finalColor, tex2D(ReShade::BackBuffer, texcoord).a);
}

//----------------------------------------------------------------------------------------------------------------------
// Technique Definition
//----------------------------------------------------------------------------------------------------------------------
technique UniversalDetailEnhancer <
    ui_name = "Universal Detail Enhancer";
    ui_tooltip = "A fully gamut-agnostic and HDR-safe detail enhancer with an adaptive filter.\n"
                 "Compatible with SDR, HDR10, and scRGB without clamping wide-gamut colors.\n"
                 "V3.1 - Final, Production-Ready Version.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = UniversalDetailEnhancerPS;
    }
}