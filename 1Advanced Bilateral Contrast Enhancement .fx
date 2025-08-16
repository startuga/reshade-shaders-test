/**
 * Advanced Bilateral Contrast Enhancement for ReShade
 *
 * This shader enhances micro-contrast and local contrast by applying a high-precision bilateral filter
 * to the luminance channel. It is meticulously engineered to preserve edges and maintain perfect color fidelity
 * (hue and saturation) across a wide range of display standards.
 *
 * Core Principles for Maximum Quality:
 * 1.  Unified Working Space: All calculations are performed in a linear Rec. 2020 color space to ensure
 *     consistent and accurate results regardless of the input signal.
 * 2.  Gamut-Aware Conversions: Utilizes a comprehensive color science library to correctly handle
 *     conversions for SDR (sRGB), scRGB, HDR10 (PQ), and HLG, including essential gamut mapping.
 * 3.  Luminance-Only Operation: The contrast enhancement is applied strictly to the luminance component,
 *     and the result is applied as a ratio back to the linear RGB data, perfectly preserving the original chrominance.
 * 4.  Numerical Stability: Employs Kahan summation to minimize floating-point errors during the filter
 *     accumulation, which is vital for high-quality processing, especially in HDR.
 *
 * Author: Your AI Assistant
 * Version: 2.0
 */

#include "lilium__include/colour_space.fxh"

// ==============================================================================
// UI Configuration
// ==============================================================================

uniform float fStrength <
    ui_type = "slider";
    ui_label = "Contrast Strength";
    ui_tooltip = "Controls the intensity of the micro-contrast enhancement.";
    ui_min = 0.0;
    ui_max = 3.0;
    ui_step = 0.05;
> = 0.5;

uniform int iRadius <
    ui_type = "slider";
    ui_label = "Filter Radius";
    ui_tooltip = "Pixel radius of the bilateral filter. Larger values affect broader details but are more performance-intensive.";
    ui_min = 1;
    ui_max = 10;
> = 3;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_tooltip = "Standard deviation of the spatial Gaussian kernel. Controls the 'spread' of the filter.";
    ui_min = 0.1;
    ui_max = 10.0;
    ui_step = 0.1;
> = 1.5;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Luminance Sigma (Edge Detection)";
    ui_tooltip = "Controls edge preservation. Lower values are more sensitive to luminance differences, preserving more edges.";
    ui_min = 0.01;
    ui_max = 0.5;
    ui_step = 0.01;
> = 0.1;

// ==============================================================================
// Color and Luminance Helpers
// ==============================================================================

namespace Bilateral
{
    /**
     * Decodes the input color from the backbuffer into the common working space: linear Rec. 2020.
     * This ensures all subsequent calculations are performed consistently and accurately.
     */
    float3 DecodeToLinearBT2020(float3 color)
    {
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
        color = DECODE_SDR(color);                // sRGB EOTF -> Linear Rec.709
        color = Csp::Mat::Bt709To::Bt2020(color); // Gamut Rec.709 -> Rec.2020
    #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        color = Csp::Mat::Bt709To::Bt2020(color); // Gamut Rec.709 -> Rec.2020 (scRGB is already linear)
    #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        color = Csp::Trc::PqTo::Linear(color);    // PQ EOTF -> Linear Rec.2020
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        color = Csp::Trc::HlgTo::Linear(color);   // HLG OETF -> Linear Rec.2020
    #endif
        return color;
    }

    /**
     * Encodes the processed linear Rec. 2020 color back to the original format of the backbuffer.
     */
    float3 EncodeFromLinearBT2020(float3 color)
    {
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
        color = Csp::Mat::Bt2020To::Bt709(color); // Gamut Rec.2020 -> Rec.709
        color = ENCODE_SDR(color);                // Linear Rec.709 -> sRGB EOTF
    #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        color = Csp::Mat::Bt2020To::Bt709(color); // Gamut Rec.2020 -> Rec.709
    #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        color = Csp::Trc::LinearTo::Pq(color);    // Linear Rec.2020 -> PQ EOTF
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        color = Csp::Trc::Linear_To::Hlg(color);  // Linear Rec.2020 -> HLG OETF
    #endif
        return color;
    }

    /**
     * Calculates the perceptual luminance of a linear color in the Rec. 2020 color space.
     */
    float GetLuminance(float3 linearBT2020)
    {
        return dot(linearBT2020, Csp::Mat::Bt2020ToXYZ[1]);
    }
}

// ==============================================================================
// Pixel Shader
// ==============================================================================

void PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    // --- 1. Fetch and Decode Center Pixel ---
    const int2 center_pos = int2(vpos.xy);
    const float3 color_encoded = tex2Dfetch(SamplerBackBuffer, center_pos).rgb;
    const float3 color_linear = Bilateral::DecodeToLinearBT2020(color_encoded);
    const float luma = Bilateral::GetLuminance(color_linear);

    // --- 2. Bilateral Filter with Kahan Summation ---
    float sum_luma = 0.0, c_luma = 0.0;
    float sum_weight = 0.0, c_weight = 0.0;
    const float epsilon = 1e-5;

    const float sigma_spatial_sq = 2.0 * fSigmaSpatial * fSigmaSpatial;
    const float sigma_range_sq = 2.0 * fSigmaRange * fSigmaRange;

    [loop]
    for (int y = -iRadius; y <= iRadius; ++y)
    {
        [loop]
        for (int x = -iRadius; x <= iRadius; ++x)
        {
            const int2 offset = int2(x, y);
            const int2 sample_pos = center_pos + offset;

            // --- Fetch and Decode Neighbor Pixel ---
            const float3 neighbor_encoded = tex2Dfetch(SamplerBackBuffer, sample_pos).rgb;
            const float3 neighbor_linear = Bilateral::DecodeToLinearBT2020(neighbor_encoded);
            const float neighbor_luma = Bilateral::GetLuminance(neighbor_linear);

            // --- Calculate Weights ---
            // Spatial Weight (Gaussian)
            const float dist_sq_spatial = dot(offset, offset);
            const float weight_spatial = exp(-dist_sq_spatial / sigma_spatial_sq);

            // Range Weight (relative for HDR compatibility)
            const float avg_luma = (luma + neighbor_luma) * 0.5 + epsilon;
            const float dist_sq_range = pow((luma - neighbor_luma) / avg_luma, 2.0);
            const float weight_range = exp(-dist_sq_range / sigma_range_sq);
            
            const float weight = weight_spatial * weight_range;

            // --- Kahan Summation for weighted luminance ---
            const float input_luma = neighbor_luma * weight;
            const float y_l = input_luma - c_luma;
            const float t_l = sum_luma + y_l;
            c_luma = (t_l - sum_luma) - y_l;
            sum_luma = t_l;

            // --- Kahan Summation for total weight ---
            const float y_w = weight - c_weight;
            const float t_w = sum_weight + y_w;
            c_weight = (t_w - sum_weight) - y_w;
            sum_weight = t_w;
        }
    }

    const float blurred_luma = sum_luma / (sum_weight + epsilon);

    // --- 3. Apply Contrast Enhancement ---
    const float enhanced_luma = luma + fStrength * (luma - blurred_luma);

    // Apply enhancement as a ratio to preserve hue and saturation
    const float ratio = (luma > epsilon) ? enhanced_luma / luma : 1.0;
    const float3 enhanced_linear = color_linear * ratio;

    // --- 4. Encode Final Color ---
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
        enhanced_linear = saturate(enhanced_linear);
    #endif

    fragColor.rgb = Bilateral::EncodeFromLinearBT2020(enhanced_linear);
    fragColor.a = 1.0;
}

// ==============================================================================
// Technique
// ==============================================================================

technique BilateralContrast <
    ui_label = "Advanced Bilateral Contrast";
    ui_tooltip = "High-precision bilateral filter for micro-contrast enhancement.\n"
                 "Color-accurate for SDR and HDR. Quality-focused.";
>
{
    pass
    {
        VertexShader = VS_PostProcess;
        PixelShader = PS_BilateralContrast;
    }
}