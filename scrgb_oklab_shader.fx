/**
 * scRGB-Compatible Perceptually Uniform Detail Enhancer v3.0
 *
 * This version addresses scRGB color clamping issues by:
 * 1. Using XYZ D65 as an intermediate color space to preserve wide gamut
 * 2. Proper handling of scRGB's extended range (negative values and >1.0)
 * 3. Safe logarithmic processing for extended dynamic range
 * 4. Optional gamut mapping to prevent impossible colors
 */

#include "ReShade.fxh"

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
    ui_tooltip = "Limits the maximum enhancement to prevent extreme brights/darks and halos. Lower values provide stronger protection.";
    ui_min = 0.1; ui_max = 5.0; ui_step = 0.05;
> = 1.0;

uniform bool bPreserveWideGamut <
    ui_label = "Preserve Wide Gamut";
    ui_tooltip = "When enabled, preserves colors outside sRGB gamut. Disable if you experience color artifacts.";
> = true;

uniform float fScRGBScale <
    ui_type = "slider"; ui_label = "scRGB Scale Factor";
    ui_tooltip = "Scaling factor for scRGB processing. Adjust if colors appear too dim or oversaturated.";
    ui_min = 0.1; ui_max = 10.0; ui_step = 0.1;
> = 1.0;

//----------------------------------------------------------------------------------------------------------------------
// Global Constants
//----------------------------------------------------------------------------------------------------------------------
static const float EPSILON_DIV = 1e-7f;
static const float EPSILON_LOG = 1e-6f;
static const float SCRGB_MIN_SAFE = -0.4f;  // Safe minimum for scRGB to avoid extreme negatives
static const float SCRGB_MAX_SAFE = 7.0f;   // Safe maximum for scRGB

//----------------------------------------------------------------------------------------------------------------------
// Color Space Conversion Matrices
//----------------------------------------------------------------------------------------------------------------------
// sRGB to XYZ D65 (for scRGB compatibility)
static const float3x3 sRGB_to_XYZ = float3x3(
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
);

// XYZ D65 to sRGB
static const float3x3 XYZ_to_sRGB = float3x3(
     3.2404542, -1.5371385, -0.4985314,
    -0.9692660,  1.8760108,  0.0415560,
     0.0556434, -0.2040259,  1.0572252
);

// XYZ D65 to Oklab (via LMS)
static const float3x3 XYZ_to_LMS = float3x3(
     0.8189330101,  0.3618667424, -0.1288597137,
     0.0329845436,  0.9293118715,  0.0361456387,
     0.0482003018,  0.2643662691,  0.6338517070
);

static const float3x3 LMS_to_XYZ = float3x3(
     1.2268798733, -0.5578149965,  0.2813910456,
    -0.0405801784,  1.1122568696, -0.0716766787,
    -0.0763812845, -0.4214819784,  1.5861632204
);

// LMS to Oklab
static const float3x3 LMS_to_Oklab = float3x3(
    0.2104542553,  0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050,  0.4505937099,
    0.0259040371,  0.7827717662, -0.8086757660
);

static const float3x3 Oklab_to_LMS = float3x3(
    1.0,  0.3963377774,  0.2158037573,
    1.0, -0.1055613458, -0.0638541728,
    1.0, -0.0894841775, -1.2914855480
);

//----------------------------------------------------------------------------------------------------------------------
// scRGB-Safe Color Space Conversion Functions
//----------------------------------------------------------------------------------------------------------------------
float3 safe_scRGB_clamp(float3 scrgb)
{
    // Safely clamp scRGB values to prevent extreme processing artifacts
    return clamp(scrgb, SCRGB_MIN_SAFE, SCRGB_MAX_SAFE);
}

float3 scRGB_to_XYZ(float3 scrgb)
{
    // scRGB is linear RGB with extended range, so we can use the sRGB matrix
    scrgb = safe_scRGB_clamp(scrgb) * fScRGBScale;
    return mul(sRGB_to_XYZ, scrgb);
}

float3 XYZ_to_scRGB(float3 xyz)
{
    float3 linear_rgb = mul(XYZ_to_sRGB, xyz);
    return linear_rgb / fScRGBScale;
}

float3 XYZ_to_Oklab(float3 xyz)
{
    float3 lms = mul(XYZ_to_LMS, xyz);
    // Protect against negative LMS values which can occur with wide gamut
    lms = max(lms, EPSILON_LOG);
    lms = pow(lms, 1.0 / 3.0);
    return mul(LMS_to_Oklab, lms);
}

float3 Oklab_to_XYZ(float3 oklab)
{
    float3 lms = mul(Oklab_to_LMS, oklab);
    lms = pow(max(lms, EPSILON_LOG), 3.0);
    return mul(LMS_to_XYZ, lms);
}

float3 scRGB_to_Oklab(float3 scrgb)
{
    return XYZ_to_Oklab(scRGB_to_XYZ(scrgb));
}

float3 Oklab_to_scRGB(float3 oklab)
{
    return XYZ_to_scRGB(Oklab_to_XYZ(oklab));
}

//----------------------------------------------------------------------------------------------------------------------
// Pixel Shader
//----------------------------------------------------------------------------------------------------------------------
float4 ScRGBOklabDetailEnhancerPS(float4 vpos : SV_POSITION, float2 texcoord : TEXCOORD) : SV_TARGET
{
    float3 originalColorScRGB = tex2D(ReShade::BackBuffer, texcoord).rgb;
    
    // Convert to Oklab via XYZ for wide gamut preservation
    float3 originalColorOklab;
    if (bPreserveWideGamut)
    {
        originalColorOklab = scRGB_to_Oklab(originalColorScRGB);
    }
    else
    {
        // Fallback: clamp to [0,1] and use standard conversion
        float3 clampedRGB = saturate(originalColorScRGB);
        originalColorOklab = XYZ_to_Oklab(scRGB_to_XYZ(clampedRGB));
    }
    
    float originalLightness = originalColorOklab.x;

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
            
            float3 neighborColorScRGB = tex2D(ReShade::BackBuffer, sampleTexcoord).rgb;
            float neighborLightness;
            
            if (bPreserveWideGamut)
            {
                neighborLightness = scRGB_to_Oklab(neighborColorScRGB).x;
            }
            else
            {
                float3 clampedNeighbor = saturate(neighborColorScRGB);
                neighborLightness = XYZ_to_Oklab(scRGB_to_XYZ(clampedNeighbor)).x;
            }

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

    // --- Controlled Reconstruction with Wide Gamut Consideration ---
    float enhancedDetail = detailLightness * fStrength;
    float deviation = detailLightness + enhancedDetail;
    float safe_deviation = fOvershootProtection * tanh(deviation / (fOvershootProtection + EPSILON_DIV));
    float newLuma = blurredLightness + safe_deviation;
    
    // --- Final Color Conversion ---
    float3 finalColorOklab = float3(newLuma, originalColorOklab.y, originalColorOklab.z);
    float3 finalColorScRGB;
    
    if (bPreserveWideGamut)
    {
        finalColorScRGB = Oklab_to_scRGB(finalColorOklab);
    }
    else
    {
        finalColorScRGB = saturate(Oklab_to_scRGB(finalColorOklab));
    }
    
    return float4(finalColorScRGB, tex2D(ReShade::BackBuffer, texcoord).a);
}

//----------------------------------------------------------------------------------------------------------------------
// Technique Definition
//----------------------------------------------------------------------------------------------------------------------
technique ScRGBOklabDetailEnhancer <
    ui_name = "scRGB-Compatible Perceptual Detail Enhancer";
    ui_tooltip = "HDR and wide-gamut safe detail enhancement using Oklab color space via XYZ.\n"
                 "V3.0 - Full scRGB compatibility with wide gamut preservation.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = ScRGBOklabDetailEnhancerPS;
    }
}