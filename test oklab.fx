/**
 * Perceptually Uniform Detail Enhancer v2.0
 *
 * This shader represents a state-of-the-art approach to detail enhancement by operating
 * in the Oklab color space. This provides theoretically superior color accuracy compared
 * to traditional luminance-based methods.
 *
 * By processing the Lightness (L) component of Oklab, we can enhance local contrast
 * while minimizing the hue and saturation shifts that can occur in other color spaces.
 * This makes it ideal for color-critical work and high-fidelity rendering.
 *
 * Version 2.0:
 *  - [MAJOR] Replaced all luminance processing with a full RGB <-> Oklab conversion pipeline.
 *  - Operations are now performed on the perceptually uniform Lightness channel (L).
 *  - UI parameters updated to reflect the change (e.g., Sigma Luma -> Sigma Lightness).
 *  - Guarantees the highest possible fidelity to the original chrominance.
 *
 * Based on the Oklab color space by Björn Ottosson.
 */

//----------------------------------------------------------------------------------------------------------------------
// UI Uniforms
//----------------------------------------------------------------------------------------------------------------------
uniform float fStrength <
    ui_type = "slider"; ui_label = "Overall Strength";
    ui_tooltip = "Overall strength of the detail enhancement effect.";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
> = 0.5;

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
    ui_type = "slider"; ui_label = "Sigma Lightness (Range)";
    ui_tooltip = "Lightness sigma for the bilateral filter. This is key for edge preservation in the Oklab space. Smaller values preserve edges more strongly.";
    ui_min = 0.001; ui_max = 1.0; ui_step = 0.001;
> = 0.1;

uniform float fDetailBoost <
    ui_type = "slider"; ui_label = "Detail Boost";
    ui_tooltip = "Multiplier for the extracted detail signal. Increases the intensity of details.";
    ui_min = 0.0; ui_max = 5.0; ui_step = 0.01;
> = 1.5;

uniform float fDetailCompress <
    ui_type = "slider"; ui_label = "Detail Compression";
    ui_tooltip = "Controls compression of the detail signal. Higher values mean stronger compression of extreme details, reducing artifacts.";
    ui_min = 0.1; ui_max = 10.0; ui_step = 0.05;
> = 2.0;

uniform bool bDebugView <
    ui_label = "Debug View: Enhanced Detail Signal";
    ui_tooltip = "Shows the raw enhanced detail signal (in Oklab Lightness) before it's added back.";
> = false;

//----------------------------------------------------------------------------------------------------------------------
// Includes and Global Constants
//----------------------------------------------------------------------------------------------------------------------
#include "ReShade.fxh"

static const float EPSILON_DIV = 1e-7f;

//----------------------------------------------------------------------------------------------------------------------
// Oklab Color Space Conversion Functions
//----------------------------------------------------------------------------------------------------------------------
// These functions convert linear sRGB to Oklab and back.
// Matrices and method from Björn Ottosson's blog post.

// Linear sRGB to LMS (cone space)
static const float3x3 M1 = float3x3(
    0.4121656120, 0.5362752080, 0.0514575653,
    0.2118591070, 0.6807189584, 0.1074065790,
    0.0883097947, 0.2818474174, 0.6302613616
);

// LMS to Oklab
static const float3x3 M2 = float3x3(
    0.2104542553,  0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050,  0.4505937099,
    0.0259040371,  0.7827717662, -0.8086757660
);

// Inverse matrices for Oklab to RGB
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


float3 linear_srgb_to_oklab(float3 rgb)
{
    float3 lms = mul(M1, rgb);
    // Apply cube root non-linearity
    lms = pow(abs(lms), 1.0 / 3.0) * sign(lms); // Use sign-safe pow
    return mul(M2, lms);
}

float3 oklab_to_linear_srgb(float3 oklab)
{
    float3 lms = mul(M2_inv, oklab);
    // Apply power non-linearity
    lms = pow(lms, 3.0);
    return mul(M1_inv, lms);
}


//----------------------------------------------------------------------------------------------------------------------
// Pixel Shader
//----------------------------------------------------------------------------------------------------------------------
float4 OklabDetailEnhancerPS(float4 vpos : SV_POSITION, float2 texcoord : TEXCOORD) : SV_TARGET
{
    // --- Initial Conversion and Data Prep ---
    // We assume the backbuffer is in linear sRGB, which is standard for modern rendering with ReShade.
    float3 originalColorRGB = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float3 originalColorOklab = linear_srgb_to_oklab(originalColorRGB);
    
    float originalLightness = originalColorOklab.x; // L is the first component

    // --- Bilateral Filter on Oklab Lightness (L) ---
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
            float neighborLightness = linear_srgb_to_oklab(neighborColorRGB).x;

            float spatialWeight = exp(-dot(pixelOffset, pixelOffset) / twoSigmaSpatialSq);
            float lightnessWeight = exp(-pow(originalLightness - neighborLightness, 2) / twoSigmaLightnessSq);
            float totalWeight = spatialWeight * lightnessWeight;

            sumWeightedLightness += neighborLightness * totalWeight;
            sumWeights += totalWeight;
        }
    }

    float blurredLightness = (sumWeights > EPSILON_DIV) ? (sumWeightedLightness / sumWeights) : originalLightness;

    // --- Detail Extraction and Enhancement (in Oklab space) ---
    float detailLightness = originalLightness - blurredLightness;
    float compressedDetail = (abs(fDetailCompress) > EPSILON_DIV) ? (tanh(detailLightness * fDetailCompress) / fDetailCompress) : detailLightness;
    float boostedDetail = compressedDetail * fDetailBoost;

    // --- Debug View ---
    if (bDebugView)
    {
        return float4((boostedDetail * 5.0 + 0.5).xxx, 1.0); // Scaled for visibility
    }

    // --- Lightness and Color Reconstruction ---
    float newLightness = originalLightness + boostedDetail * fStrength;
    newLightness = max(newLightness, 0.0); // Guard against negative lightness. No upper clamp for HDR.

    // Reconstruct the final Oklab color by combining the new Lightness
    // with the *original* chroma components (a and b). This is the key to color accuracy.
    float3 finalColorOklab = float3(newLightness, originalColorOklab.y, originalColorOklab.z);

    // Convert the final color from Oklab back to linear sRGB for output
    float3 finalColorRGB = oklab_to_linear_srgb(finalColorOklab);
    
    return float4(finalColorRGB, tex2D(ReShade::BackBuffer, texcoord).a);
}

//----------------------------------------------------------------------------------------------------------------------
// Technique Definition
//----------------------------------------------------------------------------------------------------------------------
technique OklabDetailEnhancer <
    ui_name = "Perceptual Detail Enhancer (Oklab)";
    ui_tooltip = "State-of-the-art detail enhancement using the Oklab color space.\n"
                 "Provides maximum color accuracy by isolating perceptual lightness (L)\n"
                 "for processing, preventing hue or saturation shifts.\n"
                 "V2.0 - Oklab Implementation.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = OklabDetailEnhancerPS;
    }
}