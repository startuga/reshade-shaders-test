/**
 * Advanced Bilateral Contrast Enhancement for ReShade
 *
 * This shader enhances micro-contrast by applying a high-precision bilateral filter to the luminance channel.
 * It is engineered for maximum quality, preserving edges and color fidelity across all display standards.
 *
 * Core Principles for Maximum Quality:
 * 1.  Unified Working Space: All calculations are performed in linear Rec. 2020 space for accuracy.
 * 2.  Gamut-Aware Conversions: Utilizes the Lilium color science framework for SDR, scRGB, HDR10, and HLG.
 * 3.  Luminance-Only Operation: Preserves chrominance by applying enhancement as a luminance ratio.
 * 4.  Numerical Stability: Employs Kahan summation, robust fallbacks, and explicit shadow protection to prevent artifacts.
 *
 * Author: Your AI Assistant
 * Version: 2.5 (Production Ready - Framework Optimized)
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
    ui_tooltip = "Pixel radius of the bilateral filter. Larger values affect broader details.";
    ui_min = 1;
    ui_max = 10;
> = 3;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_tooltip = "Standard deviation of the spatial kernel. A value around Radius/2 is a good start.";
    ui_min = 0.1;
    ui_max = 10.0;
    ui_step = 0.1;
> = 1.5;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Luminance Sigma (Edge Detection)";
    ui_tooltip = "Controls edge preservation. Lower values are more sensitive to luminance differences.";
    ui_min = 0.01;
    ui_max = 0.5;
    ui_step = 0.01;
> = 0.1;

uniform float fDarkProtection <
    ui_type = "slider";
    ui_label = "Dark Area Protection";
    ui_tooltip = "Fades out the contrast effect in dark areas to prevent 'black crush' and preserve shadow detail. Higher values protect brighter shadows.";
    ui_min = 0.0;
    ui_max = 0.1;
    ui_step = 0.001;
> = 0.01;

// ==============================================================================
// Color and Luminance Helpers (Namespace: Bilateral)
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
        // FIX: Use optimized LUT conversion when available
        #ifdef CSP_USE_HDR10_LUT
            color = Csp::Trc::PqTo::LinearLut(color);
        #else
            color = Csp::Trc::PqTo::Linear(color);
        #endif
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
        // FIX: Use optimized LUT conversion when available
        #ifdef CSP_USE_HDR10_LUT
            color = Csp::Trc::LinearTo::PqLut(color);
        #else
            color = Csp::Trc::LinearTo::Pq(color);
        #endif
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
    const int2 center_pos = int2(vpos.xy);
    const float3 color_encoded = tex2Dfetch(SamplerBackBuffer, center_pos).rgb;
    const float3 color_linear = Bilateral::DecodeToLinearBT2020(color_encoded);
    
    // FIX: Framework-consistent epsilon values based on color space precision
    #if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        static const float LUMA_EPSILON = 1e-8;      // Tighter for PQ precision
        static const float WEIGHT_THRESHOLD = 1e-7;
    #else
        static const float LUMA_EPSILON = 1e-7;
        static const float WEIGHT_THRESHOLD = 1e-6;
    #endif
    
    const float luma = max(Bilateral::GetLuminance(color_linear), LUMA_EPSILON);

    float sum_luma = 0.0, c_luma = 0.0;
    float sum_weight = 0.0, c_weight = 0.0;

    // FIX: More robust parameter validation
    const float sigma_spatial_clamped = clamp(fSigmaSpatial, 0.1, 20.0);
    const float sigma_range_clamped = clamp(fSigmaRange, 0.001, 1.0);
    
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
            const float neighbor_luma = max(Bilateral::GetLuminance(neighbor_linear), LUMA_EPSILON);

            const float dist_sq_spatial = dot(offset, offset);
            const float weight_spatial = exp(-dist_sq_spatial * inv_sigma_spatial_sq);

            // FIX: More robust range weight calculation
            const float avg_luma = (luma + neighbor_luma) * 0.5;
            float weight_range = 1.0;
            
            // FIX: Use dynamic threshold for better HDR handling
            #if (ACTUAL_COLOUR_SPACE == CSP_HDR10 || ACTUAL_COLOUR_SPACE == CSP_HLG)
                const float dynamic_threshold = max(WEIGHT_THRESHOLD, avg_luma * 1e-5);
            #else
                const float dynamic_threshold = WEIGHT_THRESHOLD;
            #endif
            
            if (avg_luma > dynamic_threshold) {
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

    [branch]
    if (sum_weight > WEIGHT_THRESHOLD)
    {
        const float blurred_luma = sum_luma / sum_weight;
        float luma_diff = luma - blurred_luma;

        // FIX: Enhanced dark area protection with HDR-aware scaling
        #if (ACTUAL_COLOUR_SPACE == CSP_HDR10 || ACTUAL_COLOUR_SPACE == CSP_HLG)
            // For HDR, scale protection threshold to account for wider luminance range
            const float protection_threshold = fDarkProtection * 10.0; // Allow higher protection values for HDR
            const float protection_factor = smoothstep(0.0, protection_threshold, luma);
        #else
            const float protection_factor = smoothstep(0.0, max(fDarkProtection, LUMA_EPSILON), luma);
        #endif
        
        luma_diff *= protection_factor;
        
        const float enhanced_luma = luma + fStrength * luma_diff;

        // FIX: More robust ratio calculation with HDR-aware limits
        if (enhanced_luma > LUMA_EPSILON && luma > WEIGHT_THRESHOLD) {
            const float ratio = enhanced_luma / luma;
            
            // FIX: Dynamic ratio limits based on color space capabilities
            #if (ACTUAL_COLOUR_SPACE == CSP_HDR10 || ACTUAL_COLOUR_SPACE == CSP_HLG)
                const float ratio_min = 0.01;  // Allow more aggressive darkening in HDR
                const float ratio_max = 100.0; // Allow higher brightness ratios for HDR
            #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
                const float ratio_min = 0.05;
                const float ratio_max = 20.0;  // Intermediate range for scRGB
            #else
                const float ratio_min = 0.1;
                const float ratio_max = 10.0;  // Conservative for SDR
            #endif
            
            const float safe_ratio = clamp(ratio, ratio_min, ratio_max);
            enhanced_linear = color_linear * safe_ratio;
        }
    }

    // FIX: Comprehensive output validation
    enhanced_linear = max(enhanced_linear, 0.0);
    
    // FIX: Color space-appropriate final clamping
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
        enhanced_linear = saturate(enhanced_linear);
    #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        // scRGB can handle values > 1.0, but prevent extreme values
        enhanced_linear = clamp(enhanced_linear, 0.0, 125.0); // ~10,000 nits limit
    #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        // PQ can theoretically handle up to 10,000 nits, but clamp to reasonable values
        enhanced_linear = clamp(enhanced_linear, 0.0, 100.0); // Practical HDR limit
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        // HLG typically targets 1,000 nits but can go higher
        enhanced_linear = clamp(enhanced_linear, 0.0, 10.0); // Conservative HLG limit
    #endif

    fragColor.rgb = Bilateral::EncodeFromLinearBT2020(enhanced_linear);
    fragColor.a = 1.0;
}

// ==============================================================================
// Technique
// ==============================================================================

technique lilium__bilateral_contrast <
    ui_label = "Lilium: Bilateral Contrast Enhancement";
    ui_tooltip = "HDR-aware bilateral filter for micro-contrast enhancement.\n"
                 "Optimized for the Lilium HDR framework with shadow protection.\n"
                 "Color-accurate for SDR and HDR with robust numerical stability.";
>
{
    pass
    {
        VertexShader = VS_PostProcess;
        PixelShader = PS_BilateralContrast;
    }
}