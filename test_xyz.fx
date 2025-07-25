/**
 * Complex Detail Enhancer v1.3 by AI
 *
 * Enhances local contrast using an edge-aware bilateral filter.
 * Designed to minimize halos and preserve original colors.
 * Compatible with SDR, HDR10, and scRGB. Performance is not a primary concern for this version.
 *
 * Version 1.3:
 *  - Removed faulty [unroll] attributes from loops with variable (uniform) bounds,
 *    which caused a compilation error (X3511). Loops now compile correctly.
 *
 * Version 1.2:
 *  - Replaced fixed EPSILON check on originalLuma with a "> 0.0" check for HDR safety.
 */

//----------------------------------------------------------------------------------------------------------------------
// UI Uniforms
//----------------------------------------------------------------------------------------------------------------------
uniform float fStrength <
    ui_type = "slider"; ui_label = "Overall Strength";
    ui_tooltip = "Overall strength of the detail enhancement effect. Higher values apply more of the extracted detail.";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 0.5;

uniform int iRadiusPixels <
    ui_type = "slider"; ui_label = "Bilateral Filter Radius (Pixels)";
    ui_tooltip = "Radius of the bilateral filter kernel. Larger values capture larger-scale local contrast but are significantly more performance-intensive.";
    ui_min = 1; ui_max = 20; ui_step = 1;
> = 7;

uniform float fSigmaSpatial <
    ui_type = "slider"; ui_label = "Bilateral Sigma Spatial";
    ui_tooltip = "Spatial sigma for the bilateral filter. Typically set relative to Radius (e.g., Radius / 2).";
    ui_min = 0.1; ui_max = 10.0; ui_step = 0.1;
> = 3.5;

uniform float fSigmaLuma <
    ui_type = "slider"; ui_label = "Bilateral Sigma Luma (Range Sigma)";
    ui_tooltip = "Luminance sigma for the bilateral filter. Key for edge preservation. Smaller values preserve edges more strongly. May need larger values for HDR.";
    ui_min = 0.01; ui_max = 2.0; ui_step = 0.01;
> = 0.2;

uniform float fDetailBoost <
    ui_type = "slider"; ui_label = "Detail Boost";
    ui_tooltip = "Multiplier for the extracted detail signal (after compression). Increases the intensity of details.";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 1.5;

uniform float fDetailCompress <
    ui_type = "slider"; ui_label = "Detail Compression Factor";
    ui_tooltip = "Controls hyperbolic tangent compression of the detail signal. Higher values mean stronger compression of extreme details, reducing artifacts.";
    ui_min = 0.1; ui_max = 10.0; ui_step = 0.05;
> = 2.0;

uniform bool bDebugView <
    ui_label = "Debug View: Enhanced Detail Signal";
    ui_tooltip = "Shows the raw enhanced detail signal (scaled for visibility) before it's added to the original luminance.";
> = false;


//----------------------------------------------------------------------------------------------------------------------
// Includes and Global Constants
//----------------------------------------------------------------------------------------------------------------------
#include "ReShade.fxh"

static const float3 LUMA_WEIGHTS = float3(0.2126, 0.7152, 0.0722);
static const float EPSILON_DIV = 1e-7f;

//----------------------------------------------------------------------------------------------------------------------
// Color Space Conversion Functions (Add these after the #includes)
//----------------------------------------------------------------------------------------------------------------------
// Note: These conversions assume the sRGB color space, which is standard for ReShade's backbuffer.
// Conversion matrix from sRGB (BT.709 primaries) to CIE XYZ
static const float3x3 RGB_TO_XYZ_MAT = float3x3(
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
);

// Conversion matrix from CIE XYZ to sRGB (BT.709 primaries)
static const float3x3 XYZ_TO_RGB_MAT = float3x3(
    3.2404542, -1.5371385, -0.4985314,
   -0.9692660,  1.8760108,  0.0415560,
    0.0556434, -0.2040259,  1.0572252
);

// Converts RGB to Yxy. The 'Y' component is luminance.
float3 RGBtoYxy(float3 color)
{
    float3 xyz = mul(RGB_TO_XYZ_MAT, color);
    float sum = dot(xyz, float3(1.0, 1.0, 1.0));

    if (sum < EPSILON_DIV)
    {
        // For pure black, return (0, 0.3127, 0.3290) which is Y=0 at the D65 white point.
        // This prevents division by zero and provides a neutral chromaticity.
        return float3(0.0, 0.3127, 0.3290);
    }
    
    return float3(xyz.y, xyz.x / sum, xyz.y / sum);
}

// Converts Yxy to RGB.
float3 YxytoRGB(float3 color)
{
    float Y = color.x;
    float x = color.y;
    float y = color.z;

    if (y < EPSILON_DIV) 
    {
        // If chromaticity 'y' is zero, we can't reconstruct. Return pure luminance.
        return float3(Y, Y, Y);
    }

    float3 xyz = float3(
        x * (Y / y),
        Y,
        (1.0 - x - y) * (Y / y)
    );

    return mul(XYZ_TO_RGB_MAT, xyz);
}

//----------------------------------------------------------------------------------------------------------------------
// Pixel Shader
//----------------------------------------------------------------------------------------------------------------------
float4 ComplexDetailEnhancerPS(float4 vpos : SV_POSITION, float2 texcoord : TEXCOORD) : SV_TARGET
{
    float3 originalColor = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float originalLuma = dot(originalColor, LUMA_WEIGHTS);

    // --- Bilateral Filter ---
    float sumWeightedLuma = 0.0;
    float sumWeights = 0.0;
    float twoSigmaSpatialSq = 2.0 * fSigmaSpatial * fSigmaSpatial + EPSILON_DIV;
    float twoSigmaLumaSq = 2.0 * fSigmaLuma * fSigmaLuma + EPSILON_DIV;

    // Loops must be standard loops because their bounds depend on a runtime uniform (iRadiusPixels).
    // The [unroll] attribute requires compile-time constant bounds and will fail otherwise.
    for (int y = -iRadiusPixels; y <= iRadiusPixels; ++y)
    {
        for (int x = -iRadiusPixels; x <= iRadiusPixels; ++x)
        {
            float2 pixelOffset = float2(x, y);
            float2 sampleTexcoord = texcoord + pixelOffset * ReShade::PixelSize;
            float3 neighborColor = tex2D(ReShade::BackBuffer, sampleTexcoord).rgb;
            
            float neighborLuma = dot(neighborColor, LUMA_WEIGHTS);

            float spatialWeight = exp(-dot(pixelOffset, pixelOffset) / twoSigmaSpatialSq);
            float lumaWeight = exp(-pow(originalLuma - neighborLuma, 2) / twoSigmaLumaSq);
            float totalWeight = spatialWeight * lumaWeight;

            sumWeightedLuma += neighborLuma * totalWeight;
            sumWeights += totalWeight;
        }
    }

    float blurredLuma = (sumWeights > EPSILON_DIV) ? (sumWeightedLuma / sumWeights) : originalLuma;

    // --- Detail Extraction and Enhancement ---
    float detailLuma = originalLuma - blurredLuma;
    float compressedDetail = (abs(fDetailCompress) > EPSILON_DIV) ? (tanh(detailLuma * fDetailCompress) / fDetailCompress) : detailLuma;
    float boostedDetail = compressedDetail * fDetailBoost;

    // --- Debug View ---
    if (bDebugView)
    {
        return float4((boostedDetail * 2.0 + 0.5).xxx, 1.0);
    }

    // --- Luminance and Color Reconstruction (USING Yxy) ---
    float newLuma = originalLuma + boostedDetail * fStrength;
    newLuma = max(newLuma, 0.0); // Negative-luminance guard

    // Convert original color to Yxy space
    float3 originalYxy = RGBtoYxy(originalColor);
    
    // Replace the old luminance (Y) with the new one
    originalYxy.x = newLuma;

    // Convert back to RGB for the final output
    float3 finalColor = YxytoRGB(originalYxy);

    return float4(finalColor, tex2D(ReShade::BackBuffer, texcoord).a);
}

//----------------------------------------------------------------------------------------------------------------------
// Technique Definition
//----------------------------------------------------------------------------------------------------------------------
technique ComplexDetailEnhancer <
    ui_name = "Complex Detail Enhancer";
    ui_tooltip = "Enhances local contrast using an edge-aware bilateral filter.\n"
                 "Designed to minimize halos and preserve colors across SDR/HDR.\n"
                 "V1.3 - Compiler fix for dynamic loops.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = ComplexDetailEnhancerPS;
    }
}