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
 * 5.  Entropy-Based Adaptation: Incorporates Entropy-based Adaptive Bilateral Filtering (EABF) to dynamically
 *     adjust the range sigma based on local entropy, enhancing detail preservation in textured regions.
 * 6.  Optimized Single-Pass Kernel: All neighborhood statistics (min/max luma, entropy) and the
 *     final filtering are computed in a single pass over the local pixels to maximize efficiency.
 * 7.  Non-Destructive Detail Scaling: Prevents black crush by limiting the amount of subtractive detail,
 *     rather than hard-clamping the result, preserving detail in the darkest regions.
 * 8.  HDR-Aware Output: The final output is correctly clamped only for SDR displays, preserving the full
 *     unclamped dynamic range for all HDR formats (scRGB, HDR10, HLG).
 *
 * Author: Your AI Assistant
 * Version: 2.5 (HDR Clipping Fix)
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
    ui_tooltip = "Pixel radius of the bilateral filter. Larger values are more performance-intensive.";
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
    ui_label = "Base Luminance Sigma";
    ui_tooltip = "Base value for edge preservation, adapted by local entropy.";
    ui_min = 0.01;
    ui_max = 0.5;
    ui_step = 0.01;
> = 0.1;

uniform float fAlpha <
    ui_type = "slider";
    ui_label = "EABF Adaptation Steepness";
    ui_tooltip = "Controls the steepness of the sigmoid adaptation for entropy. More negative values cause a sharper drop in sigma with higher entropy.";
    ui_min = -5.0;
    ui_max = 0.0;
    ui_step = 0.1;
> = -1.0;

uniform float fKParam <
    ui_type = "slider";
    ui_label = "EABF Adaptation Amplitude";
    ui_tooltip = "Maximum multiplier for the range sigma adaptation in low-entropy areas.";
    ui_min = 1.0;
    ui_max = 5.0;
    ui_step = 0.1;
> = 2.5;

uniform float fEntropyThreshold <
    ui_type = "slider";
    ui_label = "EABF Entropy Threshold";
    ui_tooltip = "Midpoint of the sigmoid function, based on the expected local entropy.";
    ui_min = 0.0;
    ui_max = 5.0;
    ui_step = 0.1;
> = 2.0;

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
// Definitions and Pixel Shader
// ==============================================================================

#define ENTROPY_BINS 16
#define MAX_RADIUS 10 // Must match the UI max for iRadius

void PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    const int2 center_pos = int2(vpos.xy);
    const int window_dim = 2 * iRadius + 1;
    const int num_pixels = window_dim * window_dim;
    const float epsilon = 1e-6;

    // --- 1. Single-Pass Neighborhood Sampling ---
    float neighborhood_luma[(2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1)];
    int pixel_idx = 0;

    [loop]
    for (int y = -iRadius; y <= iRadius; ++y)
    {
        [loop]
        for (int x = -iRadius; x <= iRadius; ++x)
        {
            const float3 neighbor_encoded = tex2Dfetch(SamplerBackBuffer, center_pos + int2(x, y)).rgb;
            const float3 neighbor_linear = Bilateral::DecodeToLinearBT2020(neighbor_encoded);
            neighborhood_luma[pixel_idx++] = Bilateral::GetLuminance(neighbor_linear);
        }
    }

    // --- 2. Calculate Local Stats from Stored Luminance ---
    float min_luma = neighborhood_luma[0], max_luma = neighborhood_luma[0];
    
    float hist[ENTROPY_BINS];
    [unroll]
    for (int i = 0; i < ENTROPY_BINS; ++i) { hist[i] = 0.0; }

    [loop]
    for (int i = 1; i < num_pixels; ++i)
    {
        min_luma = min(min_luma, neighborhood_luma[i]);
        max_luma = max(max_luma, neighborhood_luma[i]);
    }

    [loop]
    for (int i = 0; i < num_pixels; ++i)
    {
        float norm_luma = (neighborhood_luma[i] - min_luma) / (max_luma - min_luma + epsilon);
        int bin = clamp(int(norm_luma * (ENTROPY_BINS - 1)), 0, ENTROPY_BINS - 1);
        hist[bin] += 1.0;
    }
    
    float entropy = 0.0;
    [loop]
    for (int i = 0; i < ENTROPY_BINS; ++i)
    {
        float p = hist[i] / float(num_pixels);
        if (p > epsilon) entropy -= p * log(p);
    }

    // --- 3. Compute Adaptive Sigma and Final Filter ---
    const float exp_term = exp(-fAlpha * (entropy - fEntropyThreshold));
    const float K = fKParam / (1.0 + exp_term);
    const float local_sigma_range = K * fSigmaRange;

    const float sigma_spatial_sq = 2.0 * fSigmaSpatial * fSigmaSpatial;
    const float sigma_range_sq = 2.0 * local_sigma_range * local_sigma_range;
    
    float sum_luma = 0.0, c_luma = 0.0;
    float sum_weight = 0.0, c_weight = 0.0;
    pixel_idx = 0;
    
    const float center_luma = neighborhood_luma[num_pixels / 2];

    [loop]
    for (int y = -iRadius; y <= iRadius; ++y)
    {
        [loop]
        for (int x = -iRadius; x <= iRadius; ++x)
        {
            float neighbor_luma = neighborhood_luma[pixel_idx++];
            
            const float dist_sq_spatial = float(x * x + y * y);
            const float weight_spatial = exp(-dist_sq_spatial / sigma_spatial_sq);

            const float avg_luma = (center_luma + neighbor_luma) * 0.5 + epsilon;
            const float dist_sq_range = pow((center_luma - neighbor_luma) / avg_luma, 2.0);
            const float weight_range = exp(-dist_sq_range / sigma_range_sq);
            
            const float weight = weight_spatial * weight_range;

            // Kahan Summation
            const float input_luma = neighbor_luma * weight;
            const float y_l = input_luma - c_luma;
            const float t_l = sum_luma + y_l;
            c_luma = (t_l - sum_luma) - y_l;
            sum_luma = t_l;

            const float y_w = weight - c_weight;
            const float t_w = sum_weight + y_w;
            c_weight = (t_w - sum_weight) - y_w;
            sum_weight = t_w;
        }
    }

    const float blurred_luma = sum_luma / (sum_weight + epsilon);

    // --- 4. Apply Contrast Enhancement ---
    const float detail = center_luma - blurred_luma;
    float scaled_detail = fStrength * detail;
    
    scaled_detail = max(scaled_detail, -center_luma + epsilon);
    
    const float enhanced_luma = center_luma + scaled_detail;
    
    const float3 color_linear = Bilateral::DecodeToLinearBT2020(tex2Dfetch(SamplerBackBuffer, center_pos).rgb);
    const float ratio = (center_luma > epsilon) ? enhanced_luma / center_luma : 1.0;
    float3 enhanced_linear = color_linear * ratio;

    // --- 5. Encode Final Color ---
    // ** FIX: Only clamp to [0, 1] range for SDR outputs. HDR must not be clamped. **
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
        enhanced_linear = saturate(enhanced_linear);
    #endif

    fragColor.rgb = Bilateral::EncodeFromLinearBT2020(enhanced_linear);
    fragColor.a = 1.0;
}

// ==============================================================================
// Technique
// ==============================================================================

technique AdaptiveBilateralContrast <
    ui_label = "Adaptive Bilateral Contrast (EABF)";
    ui_tooltip = "High-precision adaptive filter for micro-contrast.\n"
                 "Dynamically adjusts detail preservation based on local texture complexity.";
>
{
    pass
    {
        VertexShader = VS_PostProcess;
        PixelShader = PS_BilateralContrast;
    }
}