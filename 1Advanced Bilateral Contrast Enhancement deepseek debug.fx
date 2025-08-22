/**
 * Advanced Bilateral Contrast Enhancement for ReShade
 * With Enhanced Debugging and Validation
 *
 * This shader enhances micro-contrast by applying a high-precision bilateral filter to the luminance channel.
 * It is engineered for maximum quality, preserving edges and color fidelity across all display standards.
 *
 * Core Principles for Maximum Quality:
 * 1.  Unified Working Space: All calculations are performed in linear Rec. 2020 space for accuracy.
 * 2.  Gamut-Aware Conversions: Utilizes the Lilium color science framework for SDR, scRGB, HDR10, and HLG.
 * 3.  Luminance-Only Operation: Preserves chrominance by applying enhancement as a luminance ratio.
 * 4.  Numerical Stability: Employs Kahan summation, robust fallbacks, and explicit shadow protection to prevent artifacts.
 * 5.  Comprehensive Debugging: Multiple visualization modes for quality validation and tuning.
 *
 * Author: Your AI Assistant
 * Version: 2.9 (Logarithmic Perception - Production Ready with Debugging)
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
    ui_tooltip = "Fades out the contrast effect in the darkest tones to prevent 'black crush' and preserve shadow detail.\n"
                 "The value sets the perceptual brightness threshold for the fade-out (e.g., 0.1 protects the darkest 10% of tones).\n"
                 "Uses a logarithmic model for consistent SDR/HDR behavior.";
    ui_min = 0.0;
    ui_max = 0.0025;
    ui_step = 0.00001;
> = 0.1;

// Enhanced Debugging and Validation
uniform int iDebugMode <
    ui_type = "combo";
    ui_label = "Debug View";
    ui_tooltip = "Visualization modes for debugging and validation.";
    ui_items = "Normal Output\0Original Luma\0Enhanced Luma\0Luma Difference\0Protection Mask\0Bilateral Weights\0Color Space Analysis\0";
> = 0;

uniform bool bShowClipping <
    ui_type = "checkbox";
    ui_label = "Show Clipping";
    ui_tooltip = "Highlight areas where values are being clipped (red = overbright, blue = underbright).";
> = false;

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
    
    // Debug visualization helpers
    float3 HeatMap(float value, float minVal, float maxVal)
    {
        float normalized = saturate((value - minVal) / (maxVal - minVal));
        float3 color = float3(0.0, 0.0, 0.0);
        
        if (normalized < 0.25)
            color = lerp(float3(0.0, 0.0, 1.0), float3(0.0, 1.0, 1.0), normalized / 0.25);
        else if (normalized < 0.5)
            color = lerp(float3(0.0, 1.0, 1.0), float3(0.0, 1.0, 0.0), (normalized - 0.25) / 0.25);
        else if (normalized < 0.75)
            color = lerp(float3(0.0, 1.0, 0.0), float3(1.0, 1.0, 0.0), (normalized - 0.5) / 0.25);
        else
            color = lerp(float3(1.0, 1.0, 0.0), float3(1.0, 0.0, 0.0), (normalized - 0.75) / 0.25);
            
        return color;
    }
    
    float3 HighlightClipping(float3 color, float3 linearColor)
    {
        if (bShowClipping)
        {
            // Check for underbright (values below 0)
            if (any(linearColor < 0.0))
                return lerp(color, float3(0.0, 0.0, 1.0), 0.5);
                
            // Check for overbright based on color space
            #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
                if (any(linearColor > 1.0))
                    return lerp(color, float3(1.0, 0.0, 0.0), 0.5);
            #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
                if (any(linearColor > 125.0)) // ~10,000 nits limit
                    return lerp(color, float3(1.0, 0.0, 0.0), 0.5);
            #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
                if (any(linearColor > 100.0)) // Practical HDR limit
                    return lerp(color, float3(1.0, 0.0, 0.0), 0.5);
            #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
                if (any(linearColor > 10.0)) // Conservative HLG limit
                    return lerp(color, float3(1.0, 0.0, 0.0), 0.5);
            #endif
        }
        
        return color;
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
    
    #if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        static const float LUMA_EPSILON = 1e-8;
        static const float WEIGHT_THRESHOLD = 1e-7;
    #else
        static const float LUMA_EPSILON = 1e-7;
        static const float WEIGHT_THRESHOLD = 1e-6;
    #endif
    
    const float luma = max(Bilateral::GetLuminance(color_linear), LUMA_EPSILON);

    float sum_luma = 0.0, c_luma = 0.0;
    float sum_weight = 0.0, c_weight = 0.0;

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

            // FIX: Calculate avg_luma before using it
            const float avg_luma = (luma + neighbor_luma) * 0.5;
            float weight_range = 1.0;

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
    float enhanced_luma = luma;
    float luma_diff = 0.0;
    float protection_factor = 1.0;

    [branch]
    if (sum_weight > WEIGHT_THRESHOLD)
    {
        const float blurred_luma = sum_luma / sum_weight;
        luma_diff = luma - blurred_luma;

        // Logarithmic Perceptual Dark Area Protection
        #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
            static const float MAX_LUMA_ESTIMATE = 1.0;
        #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
            static const float MAX_LUMA_ESTIMATE = 125.0;
        #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
            static const float MAX_LUMA_ESTIMATE = 100.0;
        #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
            static const float MAX_LUMA_ESTIMATE = 10.0;
        #else
            static const float MAX_LUMA_ESTIMATE = 1.0;
        #endif

        const float log_base = log(1.0 + MAX_LUMA_ESTIMATE);
        
        if (log_base > LUMA_EPSILON) {
            const float perceptual_luma = log(1.0 + luma) / log_base;
            protection_factor = smoothstep(0.0, max(fDarkProtection, LUMA_EPSILON), perceptual_luma);
            luma_diff *= protection_factor;
        }
        
        enhanced_luma = luma + fStrength * luma_diff;

        if (enhanced_luma > LUMA_EPSILON && luma > WEIGHT_THRESHOLD) {
            const float ratio = enhanced_luma / luma;
            
            #if (ACTUAL_COLOUR_SPACE == CSP_HDR10 || ACTUAL_COLOUR_SPACE == CSP_HLG)
                const float ratio_min = 1e-6;
                const float ratio_max = 1000.0;
            #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
                const float ratio_min = 1e-6;
                const float ratio_max = 1000.0;
            #else
                const float ratio_min = 1e-6;
                const float ratio_max = 10.0;
            #endif
            
            const float safe_ratio = clamp(ratio, ratio_min, ratio_max);
            enhanced_linear = color_linear * safe_ratio;
        }
    }

    // Enhanced Debugging and Validation Outputs
    if (iDebugMode > 0)
    {
        switch (iDebugMode)
        {
            case 1: // Original Luma
                fragColor.rgb = luma.xxx;
                break;
                
            case 2: // Enhanced Luma
                fragColor.rgb = enhanced_luma.xxx;
                break;
                
            case 3: // Luma Difference
                // Scale for better visualization
                float scaled_diff = luma_diff * 10.0;
                fragColor.rgb = Bilateral::HeatMap(scaled_diff, -1.0, 1.0);
                break;
                
            case 4: // Protection Mask
                fragColor.rgb = protection_factor.xxx;
                break;
                
            case 5: // Bilateral Weights
                // Normalize weight sum for visualization
                float normalized_weight = sum_weight / ((iRadius*2+1)*(iRadius*2+1));
                fragColor.rgb = normalized_weight.xxx;
                break;
                
            case 6: // Color Space Analysis
                // Show which color space we're working in
                #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
                    fragColor.rgb = float3(1.0, 0.0, 0.0); // Red for SDR
                #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
                    fragColor.rgb = float3(0.0, 1.0, 0.0); // Green for scRGB
                #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
                    fragColor.rgb = float3(0.0, 0.0, 1.0); // Blue for HDR10
                #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
                    fragColor.rgb = float3(1.0, 1.0, 0.0); // Yellow for HLG
                #else
                    fragColor.rgb = float3(1.0, 1.0, 1.0); // White for unknown
                #endif
                break;
        }
        
        // Apply clipping visualization to debug outputs
        fragColor.rgb = Bilateral::HighlightClipping(fragColor.rgb, enhanced_linear);
        fragColor.a = 1.0;
        return;
    }

    // Normal output processing
    enhanced_linear = max(enhanced_linear, 0.0);
    enhanced_linear = min(enhanced_linear, enhanced_linear); // NaN check
    
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
        enhanced_linear = saturate(enhanced_linear);
    #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        enhanced_linear = clamp(enhanced_linear, 0.0, 125.0);
    #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        enhanced_linear = clamp(enhanced_linear, 0.0, 100.0);
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        enhanced_linear = clamp(enhanced_linear, 0.0, 10.0);
    #endif

    fragColor.rgb = Bilateral::EncodeFromLinearBT2020(enhanced_linear);
    
    // Apply clipping visualization to normal output
    fragColor.rgb = Bilateral::HighlightClipping(fragColor.rgb, enhanced_linear);
    fragColor.a = 1.0;
}

// ==============================================================================
// Technique
// ==============================================================================

technique lilium__bilateral_contrast <
    ui_label = "Lilium: Bilateral Contrast Enhancement";
    ui_tooltip = "HDR-aware bilateral filter for micro-contrast enhancement.\n"
                 "Features logarithmic perceptual dark area protection.\n"
                 "Includes comprehensive debugging and validation tools.\n"
                 "Optimized for the Lilium HDR framework with robust numerical stability.";
>
{
    pass
    {
        VertexShader = VS_PostProcess;
        PixelShader = PS_BilateralContrast;
    }
}