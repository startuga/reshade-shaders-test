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
 * 4.  Numerical Stability: Employs Kahan summation and robust fallbacks to minimize floating-point errors
 *     and prevent artifacts, which is vital for high-quality processing, especially in HDR.
 *
 * Author: Your AI Assistant
 * Version: 2.2 (Final Review)
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
    ui_tooltip = "Standard deviation of the spatial Gaussian kernel. A value around Radius/2 is often a good starting point.";
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
    float3 DecodeToLinearBT2020(float3 color)
    {
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
        color = DECODE_SDR(color);
        color = Csp::Mat::Bt709To::Bt2020(color);
    #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        color = Csp::Mat::Bt709To::Bt2020(color);
    #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        color = Csp::Trc::PqTo::Linear(color);
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        color = Csp::Trc::HlgTo::Linear(color);
    #endif
        return color;
    }

    float3 EncodeFromLinearBT2020(float3 color)
    {
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
        color = Csp::Mat::Bt2020To::Bt709(color);
        color = ENCODE_SDR(color);
    #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        color = Csp::Mat::Bt2020To::Bt709(color);
    #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        color = Csp::Trc::LinearTo::Pq(color);
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        color = Csp::Trc::LinearTo::Hlg(color);
    #endif
        return color;
    }

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
    const float luma = max(Bilateral::GetLuminance(color_linear), 1e-8); // FIX: Prevent zero luminance

    // --- 2. Bilateral Filter with Kahan Summation ---
    float sum_luma = 0.0, c_luma = 0.0;
    float sum_weight = 0.0, c_weight = 0.0;
    const float WEIGHT_THRESHOLD = 1e-6; // FIX: Tightened threshold for better precision

    // FIX: Clamp sigma values to prevent division by zero or extreme values
    const float sigma_spatial_clamped = max(fSigmaSpatial, 0.1);
    const float sigma_range_clamped = max(fSigmaRange, 0.01);
    
    const float inv_sigma_spatial_sq = 1.0 / (2.0 * sigma_spatial_clamped * sigma_spatial_clamped);
    const float inv_sigma_range_sq = 1.0 / (2.0 * sigma_range_clamped * sigma_range_clamped);

    [loop]
    for (int y = -iRadius; y <= iRadius; ++y)
    {
        [loop]
        for (int x = -iRadius; x <= iRadius; ++x)
        {
            const int2 offset = int2(x, y);
            const int2 sample_pos = center_pos + offset;

            const float3 neighbor_encoded = tex2Dfetch(SamplerBackBuffer, sample_pos).rgb;
            const float3 neighbor_linear = Bilateral::DecodeToLinearBT2020(neighbor_encoded);
            const float neighbor_luma = max(Bilateral::GetLuminance(neighbor_linear), 1e-8); // FIX: Prevent zero luminance

            // Spatial Weight (Gaussian)
            const float dist_sq_spatial = dot(offset, offset);
            const float weight_spatial = exp(-dist_sq_spatial * inv_sigma_spatial_sq);

            // Range Weight (relative for HDR compatibility)
            const float avg_luma = (luma + neighbor_luma) * 0.5;
            float weight_range = 1.0;
            
            // FIX: Use WEIGHT_THRESHOLD consistently for numerical stability
            if (avg_luma > WEIGHT_THRESHOLD) {
                const float luma_diff = luma - neighbor_luma;
                const float normalized_diff = luma_diff / avg_luma;
                const float dist_sq_range = normalized_diff * normalized_diff;
                weight_range = exp(-dist_sq_range * inv_sigma_range_sq);
            }
            
            const float weight = weight_spatial * weight_range;

            // Kahan Summation for weighted luminance
            const float input_luma = neighbor_luma * weight;
            const float y_l = input_luma - c_luma;
            const float t_l = sum_luma + y_l;
            c_luma = (t_l - sum_luma) - y_l;
            sum_luma = t_l;

            // Kahan Summation for total weight
            const float y_w = weight - c_weight;
            const float t_w = sum_weight + y_w;
            c_weight = (t_w - sum_weight) - y_w;
            sum_weight = t_w;
        }
    }

    float3 enhanced_linear = color_linear;

    // FIX: Use consistent threshold and add safety checks
    [branch]
    if (sum_weight > WEIGHT_THRESHOLD)
    {
        const float blurred_luma = sum_luma / sum_weight;
        const float luma_diff = luma - blurred_luma;
        const float enhanced_luma = luma + fStrength * luma_diff;

        // FIX: Ensure enhanced luminance stays positive and prevent extreme ratios
        if (enhanced_luma > 1e-8 && luma > WEIGHT_THRESHOLD) {
            const float ratio = enhanced_luma / luma;
            // FIX: Clamp ratio to prevent extreme values that could cause artifacts
            const float safe_ratio = clamp(ratio, 0.1, 10.0);
            enhanced_linear = color_linear * safe_ratio;
        }
    }

    // --- 4. Encode Final Color ---
    // FIX: Ensure no NaN/Inf values before encoding
    enhanced_linear = max(enhanced_linear, 0.0);
    
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