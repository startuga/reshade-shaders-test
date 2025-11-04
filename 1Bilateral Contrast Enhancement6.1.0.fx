/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement for ReShade
 *
 * This shader implements academically rigorous bilateral filtering with:
 * 1. Correct log2 luminance ratio processing (not absolute values)
 * 2. Proper Gaussian range kernel without threshold artifacts
 * 3. SDR/HDR processing with proper color space conversions
 * 4. Variance-based adaptive strength for superior detail detection
 * 5. True circular loop sampling for optimal iteration count
 * 6. Zero-overhead adaptive radius via hardware gradients
 * 7. Performance monitoring and debug modes
 * 8. Content-aware tuning controls for different source material
 *
 * Version: 6.1.0 (Streamlined)
 */

#include "ReShade.fxh"
#include "lilium__include/colour_space.fxh"  // https://github.com/EndlesslyFlowering/ReShade_HDR_shaders/blob/master/Shaders/lilium__include/colour_space.fxh

// ==============================================================================
// UI Configuration
// ==============================================================================

uniform int iQualityPreset <
    ui_type = "combo";
    ui_label = "Quality Preset";
    ui_tooltip = "Quick presets for performance vs quality.\n"
                 "Custom allows manual configuration of all parameters.";
    ui_items = "Performance\0Balanced\0Quality\0Ultra\0Custom\0";
    ui_category = "Presets";
> = 2;

uniform float fStrength <
    ui_type = "slider";
    ui_label = "Contrast Strength";
    ui_tooltip = "Controls the intensity of the micro-contrast enhancement.\n"
                 "Higher values create more dramatic local contrast.";
    ui_min = 0.1; // Log sliders cannot have a min of 0
    ui_max = 4.0;
    ui_step = 0.01;
    ui_category = "Core Settings";
> = 3.0;

uniform bool bAdaptiveStrength <
    ui_label = "Enable Adaptive Strength";
    ui_tooltip = "Automatically adjusts strength based on local image statistics.\n"
                 "• Reduces enhancement in flat areas to avoid noise amplification\n"
                 "• Increases enhancement where detail is present\n"
                 "• Uses both dynamic range and variance for superior adaptation";
    ui_category = "Adaptive Processing";
> = true;

uniform int iAdaptiveMode <
    ui_type = "combo";
    ui_label = "Adaptive Mode";
    ui_tooltip = "Method for calculating local image statistics:\n"
                 "• Dynamic Range: Min/max luminance difference\n"
                 "• Variance: Statistical variance of luminance\n"
                 "• Hybrid: Combines both metrics for best results";
    ui_items = "Dynamic Range\0Variance\0Hybrid (Recommended)\0";
    ui_category = "Adaptive Processing";
> = 2;

uniform float fAdaptiveAmount <
    ui_type = "slider";
    ui_label = "Adaptive Strength Amount";
    ui_tooltip = "How much the local statistics affect the strength.\n"
                 "0.0 = No adaptation (always uses base strength)\n"
                 "1.0 = Full adaptation (from 0x to 2x strength)";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
    ui_category = "Adaptive Processing";
> = 0.5;

uniform float fAdaptiveCurve <
    ui_type = "slider";
    ui_label = "Adaptive Response Curve";
    ui_tooltip = "Controls how image statistics map to strength modulation.\n"
                 "Lower values = More aggressive adaptation\n"
                 "Higher values = More conservative adaptation\n\n"
                 "Recommended: 1.5 (general), 2.0-2.5 (noisy), 1.0-1.3 (clean CGI)";
    ui_min = 0.5;
    ui_max = 3.0;
    ui_step = 0.01;
    ui_category = "Adaptive Processing";
> = 1.5;

uniform int iRadius <
    ui_type = "slider";
    ui_label = "Filter Radius";
    ui_tooltip = "Pixel radius of the bilateral filter.\n"
                 "Larger values affect broader spatial details but are slower.\n"
                 "Automatically adjusted by quality preset.";
    ui_min = 1;
    ui_max = 16;
    ui_category = "Filter Parameters";
> = 7;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_tooltip = "Standard deviation of the spatial Gaussian kernel.\n"
                 "Controls the spatial fall-off of the filter.";
    ui_min = 0.1;
    ui_max = 8.0;
    ui_step = 0.01;
    ui_category = "Filter Parameters";
> = 3.0;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Range Sigma (in Exposure Stops)";
    ui_tooltip = "Controls edge preservation in exposure stops:\n"
                 "0.3 = Sharp edges (preserves 1.2x luminance ratios)\n"
                 "0.5 = Moderate (preserves 1.4x luminance ratios)\n"
                 "1.0 = Soft edges (preserves 2.0x luminance ratios)\n"
                 "2.0 = Very soft (preserves 4.0x luminance ratios)\n\n"
                 "Recommended: 0.3 (general), 0.4-0.5 (smooth content)";
    ui_min = 0.1;
    ui_max = 3.0;
    ui_step = 0.01;
    ui_category = "Filter Parameters";
> = 0.30;

uniform bool bAdaptiveRadius <
    ui_label = "Enable Adaptive Radius";
    ui_tooltip = "Dynamically adjusts filter radius based on local gradients.\n"
                 "Reduces radius in low-detail areas for better performance.";
    ui_category = "Optimizations";
> = true;

uniform bool bEnableSpatialCutoff <
    ui_label = "Enable Spatial Cutoff";
    ui_tooltip = "Limits processing to the effective Gaussian radius (3 sigma).\n"
                 "Significantly improves performance with large radii and high sigma values.";
    ui_category = "Optimizations";
> = true;

uniform float fGradientSensitivity <
    ui_type = "slider";
    ui_label = "Gradient Sensitivity";
    ui_tooltip = "Controls adaptive radius response to image gradients.\n"
                 "Lower = Maintains larger radii in detailed areas\n"
                 "Higher = More aggressive radius reduction\n\n"
                 "CONTENT-SPECIFIC RECOMMENDATIONS:\n"
                 "• High-frequency (hair, foliage, textures): 50-75\n"
                 "• General content (default): 100\n"
                 "• Smooth content (portraits, skies): 125-150\n\n"
                 "Only applies when Adaptive Radius is enabled.";
    ui_min = 1.0;
    ui_max = 200.0;
    ui_step = 5.0;
    ui_category = "Advanced Tuning";
> = 100.0;

uniform float fVarianceWeight <
    ui_type = "slider";
    ui_label = "Variance Weight (Hybrid Mode)";
    ui_tooltip = "Balances variance vs dynamic range in Hybrid adaptive mode.\n"
                 "Higher = Prioritizes variance (texture complexity)\n"
                 "Lower = Prioritizes dynamic range (contrast extremes)\n\n"
                 "CONTENT-SPECIFIC RECOMMENDATIONS:\n"
                 "• Noisy input (low-light, high-ISO): 0.80-0.90\n"
                 "• General content (default): 0.70\n"
                 "• Clean CGI/renders: 0.50-0.60\n\n"
                 "Only applies when Adaptive Mode is set to Hybrid.\n"
                 "The weight for Dynamic Range is automatically set to (1.0 - Variance Weight).";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.05;
    ui_category = "Advanced Tuning";
> = 0.70;

uniform int iDebugMode <
    ui_type = "combo";
    ui_label = "Debug Visualization";
    ui_tooltip = "Visualize various aspects of the filtering process.";
    ui_items = "Off\0"
               "Show Weights\0"
               "Show Local Variance\0"
               "Show Dynamic Range\0"
               "Show Enhancement Map\0"
               "Show Adaptive Radius (Color)\0"
               "Performance Heatmap\0"
               "Show Black Pixels\0"
               "Show Radius Grid\0";
    ui_category = "Debug";
> = 0;

uniform bool bShowStatistics <
    ui_label = "Show Performance Statistics";
    ui_tooltip = "Display filter efficiency metrics as overlay.\n"
                 "Shows: Radius, Samples, Efficiency, Black pixels";
    ui_category = "Debug";
> = false;

// ==============================================================================
// Constants & Utilities
// ==============================================================================

namespace Constants
{
    static const float WEIGHT_THRESHOLD = 1e-7f;
    static const float RATIO_MIN = 0.125f;    // Prevent excessive darkening (1/8x)
    static const float RATIO_MAX = 8.0f;      // Prevent excessive brightening (8x)
    static const float SPATIAL_CUTOFF_SIGMA = 3.0f;
    static const int MAX_LOOP_RADIUS = 16;    // Hard limit to prevent crashes
}

namespace FxUtils
{
    void KahanSum(inout float sum, inout float compensation, const float input)
    {
        const float y = input - compensation;
        const float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
}

// ==============================================================================
// Quality Preset System
// ==============================================================================

namespace QualitySettings
{
    int GetRadius()
    {
        switch(iQualityPreset)
        {
            case 0: return 3;  // Performance
            case 1: return 4;  // Balanced
            case 2: return 6;  // Quality
            case 3: return 9; // Ultra
            default: return iRadius; // Custom
        }
    }
    
    float GetSpatialSigma()
    {
        switch(iQualityPreset)
        {
            case 0: return 1.5;  // Performance
            case 1: return 2.5;  // Balanced
            case 2: return 3.0;  // Quality
            case 3: return 3.5;  // Ultra
            default: return fSigmaSpatial; // Custom
        }
    }
}

// ==============================================================================
// Color Science Pipeline
// ==============================================================================

namespace ColorScience
{
    float3 DecodeToLinear(float3 color)
    {
        #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
            color = DECODE_SDR(color);
        #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
            color = Csp::Mat::ScRgbTo::Bt2020Normalised(color);
        #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
            color = Csp::Trc::PqTo::Linear(color);
        #endif
        return color;
    }

    float3 EncodeFromLinear(float3 color)
    {        
        #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
            color = ENCODE_SDR(color);
        #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
            color = Csp::Mat::Bt2020NormalisedTo::ScRgb(color);
        #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
            color = Csp::Trc::LinearTo::Pq(color);
        #endif
        return color;
    }

    float GetLuminance(const float3 linearColour)
    {
        #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
            return dot(linearColour, Csp::Mat::Bt709ToXYZ[1]);
        #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
            return dot(linearColour, Csp::Mat::ScRgbToXYZ[1]);
        #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
            return dot(linearColour, Csp::Mat::Bt2020ToXYZ[1]);
        #endif
    }

    // NO EPSILON - pure log2, can return -infinity
    float LinearToLog2Ratio(const float linear_luma) { return log2(linear_luma); }
    float Log2RatioToLinear(const float log2_ratio) { return exp2(log2_ratio); }
}

// ==============================================================================
// Bilateral Filtering Core
// ==============================================================================

namespace BilateralFilter
{
    int GetAdaptiveRadius(float log2_luma, int base_radius)
    {
        // Skip if we have infinity
        if (isinf(log2_luma))
            return base_radius;
            
        float gx = ddx(log2_luma);
        float gy = ddy(log2_luma);
        float gradient_sq = gx * gx + gy * gy;
        const float GRADIENT_SQ_THRESHOLD = 0.25;        
        // Fixed: Restored smoothstep for proper clamping
        const float radius_scale = smoothstep(0.0, GRADIENT_SQ_THRESHOLD, gradient_sq * fGradientSensitivity);        
        // Extremely aggressive reduction for maximum performance
        const float adaptive_factor = lerp(0.01, 1.0, radius_scale);        
        // Ensure at least 1 pixel radius
        int adapted_radius = (int)(base_radius * adaptive_factor + 0.5);
        if (adapted_radius < 1)
        {
            return (radius_scale < 0.1) ? 1 : 2;
        }
        
        return adapted_radius;
    }
    
    float CalculateAdaptiveStrength(
        const float local_dynamic_range, const float local_variance,
        const float base_strength, const float adaptive_amount,
        const float adaptive_curve, const int mode)
    {
        // Check for infinity in dynamic range
        if (isinf(local_dynamic_range))
            return base_strength; // Return base strength if we can't calculate adaptation          
        float metric;
        if (mode == 0) { // Dynamic Range
            // FIX: Normalize dynamic range to a reasonable scale
            // In log2 space: 0 = flat, 1 = 2x variation, 3 = 8x variation, 5 = 32x variation
            // Normalize to [0,1] assuming 5 stops is extreme
            const float MAX_EXPECTED_RANGE = 5.0; // 5 stops = 32x brightness variation
            metric = saturate(local_dynamic_range / MAX_EXPECTED_RANGE);
        } else if (mode == 1) { // Variance
            // FIX: Normalize variance to a reasonable scale
            // Variance in log2 space squared, typical range 0 to ~1
            const float MAX_EXPECTED_VARIANCE = 1.0;
            metric = saturate(local_variance / MAX_EXPECTED_VARIANCE);
        } else { // Hybrid
            // FIX: Both metrics need normalization before combining
            const float MAX_EXPECTED_RANGE = 5.0;
            const float MAX_EXPECTED_VARIANCE = 1.0;
            
            float range_metric = saturate(local_dynamic_range / MAX_EXPECTED_RANGE);
            float variance_metric = saturate(local_variance / MAX_EXPECTED_VARIANCE);
            
            // Still need small epsilon to avoid pow(0, x) issues
            range_metric = max(range_metric, 1e-6);
            variance_metric = max(variance_metric, 1e-6);
            metric = pow(range_metric, 1.0 - fVarianceWeight) * pow(variance_metric, fVarianceWeight);
        }
        
        const float modulation = pow(metric, adaptive_curve);
        const float adaptive_multiplier = lerp(1.0, modulation * 2.0, adaptive_amount);
        return base_strength * adaptive_multiplier;
    }
}

// ==============================================================================
// Main Pixel Shader
// ==============================================================================

void PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    // --- 1. Initial Data Fetch ---
    const int2 center_pos = int2(vpos.xy);
    const float3 color_encoded = tex2Dfetch(SamplerBackBuffer, center_pos).rgb;
    const float3 color_linear = ColorScience::DecodeToLinear(color_encoded);
    const float luma_linear = ColorScience::GetLuminance(color_linear);
    const float log2_ratio_center = ColorScience::LinearToLog2Ratio(luma_linear); // Can be -inf
    
    // --- 2. Check for Pure Black ---
    // If center pixel is pure black (log2 = -inf), skip processing entirely
    bool is_black_pixel = isinf(log2_ratio_center);
    
    if (is_black_pixel)
    {
        // Debug mode 7: Show black pixels
        if (iDebugMode == 7)
        {
            fragColor.rgb = float3(1, 0, 1); // Magenta for black pixels
            fragColor.a = 1.0;
            return;
        }

        // Normal mode: pass through unchanged        
        fragColor.rgb = color_encoded;
        fragColor.a = 1.0;
        return;
    }
    
    // --- 3. Setup Kernel Parameters ---
    const int base_radius = QualitySettings::GetRadius();
    int effective_radius = base_radius;
    if (bAdaptiveRadius && base_radius > 2)
    {
        effective_radius = BilateralFilter::GetAdaptiveRadius(log2_ratio_center, base_radius);

        if (iDebugMode == 5) // Adaptive radius visualization
        {
            float adaptive_factor = (float)effective_radius / (float)base_radius;
            
            // Color coding for the extreme range:
            // Blue (1-10% reduction) -> Cyan (10-25%) -> Green (25-50%) -> Yellow (50-75%) -> Red (75-100%)
            float3 debug_color;
            if (adaptive_factor < 0.1)
                debug_color = float3(0, 0, 1); // Deep blue for extreme reduction (1-10%)
            else if (adaptive_factor < 0.25)
                debug_color = lerp(float3(0, 0, 1), float3(0, 1, 1), (adaptive_factor - 0.1) / 0.15); // Blue to cyan
            else if (adaptive_factor < 0.5)
                debug_color = lerp(float3(0, 1, 1), float3(0, 1, 0), (adaptive_factor - 0.25) / 0.25); // Cyan to green
            else if (adaptive_factor < 0.75)
                debug_color = lerp(float3(0, 1, 0), float3(1, 1, 0), (adaptive_factor - 0.5) / 0.25); // Green to yellow
            else
                debug_color = lerp(float3(1, 1, 0), float3(1, 0, 0), (adaptive_factor - 0.75) / 0.25); // Yellow to red
            
            fragColor.rgb = debug_color;
            fragColor.a = 1.0;
            return;
        }
    }
    
    const float sigma_spatial = QualitySettings::GetSpatialSigma();
    const float inv_2_sigma_spatial_sq = 0.5 / (sigma_spatial * sigma_spatial);
    const float inv_2_sigma_range_sq = 0.5 / (max(fSigmaRange, 0.01) * max(fSigmaRange, 0.01));
    
    //Apply hard limit to prevent crashes when spatial cutoff is disabled
    float cutoff_radius;
    if (bEnableSpatialCutoff)
    {
        cutoff_radius = min((float)effective_radius, Constants::SPATIAL_CUTOFF_SIGMA * sigma_spatial);
    }
    else
    {
        // When cutoff is disabled, still apply a reasonable maximum to prevent crashes
        cutoff_radius = min((float)effective_radius, (float)Constants::MAX_LOOP_RADIUS);
    }
    
    const float cutoff_radius_sq = cutoff_radius * cutoff_radius;
    const int max_radius = min((int)cutoff_radius, Constants::MAX_LOOP_RADIUS); // Extra safety
    
    // --- 4. Run Bilateral Filter Loop ---
    float sum_log2 = 0.0, compensation_log2 = 0.0;
    float sum_log2_sq = 0.0, compensation_log2_sq = 0.0;
    float sum_weight = 0.0, compensation_weight = 0.0;
    float min_log2 = log2_ratio_center, max_log2 = log2_ratio_center;
    int pixels_processed = 0;
    int valid_samples = 0;
    int black_pixels_skipped = 0;

    [loop]
    for (int y = -max_radius; y <= max_radius; ++y)
    {
        const float y_sq = (float)(y * y);
        const int x_max = (int)sqrt(max(0.0, cutoff_radius_sq - y_sq));
        
        [loop]
        for (int x = -x_max; x <= x_max; ++x)
        {
            pixels_processed++;
            const float r2 = (float)(x * x) + y_sq;
            
            const int2 sample_pos = center_pos + int2(x, y);
            const float3 neighbor_linear = ColorScience::DecodeToLinear(tex2Dfetch(SamplerBackBuffer, sample_pos).rgb);
            const float log2_ratio_neighbor = ColorScience::LinearToLog2Ratio(ColorScience::GetLuminance(neighbor_linear));
            
            // Skip if neighbor is pure black (-inf)
            if (isinf(log2_ratio_neighbor))
            {
                black_pixels_skipped++;
                continue;
            }
                
            valid_samples++;
            
            const float d = log2_ratio_center - log2_ratio_neighbor;
            const float spatial_exponent = -r2 * inv_2_sigma_spatial_sq;
            const float range_exponent = -(d * d) * inv_2_sigma_range_sq;
            const float weight = exp(spatial_exponent + range_exponent);
            
            if (weight > Constants::WEIGHT_THRESHOLD)
            {
                FxUtils::KahanSum(sum_log2, compensation_log2, log2_ratio_neighbor * weight);
                FxUtils::KahanSum(sum_log2_sq, compensation_log2_sq, log2_ratio_neighbor * log2_ratio_neighbor * weight);
                FxUtils::KahanSum(sum_weight, compensation_weight, weight);
                
                if (bAdaptiveStrength)
                {
                    min_log2 = min(min_log2, log2_ratio_neighbor);
                    max_log2 = max(max_log2, log2_ratio_neighbor);
                }
            }
        }
    }
    
    // --- 5. Calculate Final Color ---
    float3 enhanced_linear = color_linear;
    if (sum_weight > Constants::WEIGHT_THRESHOLD && valid_samples > 0)
    {
        const float blurred_log2_ratio = sum_log2 / sum_weight;
        float log2_diff = log2_ratio_center - blurred_log2_ratio;

        float effective_strength = fStrength;
        if (bAdaptiveStrength)
        {
            const float local_dynamic_range = max_log2 - min_log2;
            const float mean_log2 = sum_log2 / sum_weight;
            const float mean_sq_log2 = sum_log2_sq / sum_weight;
            const float local_variance = max(0.0, mean_sq_log2 - mean_log2 * mean_log2);
            
            effective_strength = BilateralFilter::CalculateAdaptiveStrength(
                local_dynamic_range, local_variance, fStrength, 
                fAdaptiveAmount, fAdaptiveCurve, iAdaptiveMode
            );
        }        
        const float enhanced_log2_ratio = log2_ratio_center + effective_strength * log2_diff;
        
        // Check for valid enhanced ratio before conversion
        if (!isinf(enhanced_log2_ratio))
        {
            const float enhanced_luma_linear = ColorScience::Log2RatioToLinear(enhanced_log2_ratio);
            
            // Calculate ratio and apply safety bounds
            float ratio = enhanced_luma_linear / luma_linear;
            ratio = clamp(ratio, Constants::RATIO_MIN, Constants::RATIO_MAX);
            enhanced_linear = color_linear * ratio;
        }
        
        // --- 6. Apply Debug Visualizations (if enabled) ---
        if (iDebugMode > 0 && iDebugMode != 7)
        {
            switch(iDebugMode)
            {
                case 1: enhanced_linear = sum_weight.xxx; break;
                case 2:
                {
                    const float mean_log2 = sum_log2 / sum_weight;
                    const float mean_sq_log2 = sum_log2_sq / sum_weight;
                    const float variance = max(0.0, mean_sq_log2 - mean_log2 * mean_log2);
                    enhanced_linear = float3(variance * 10.0, variance * 5.0, 0.0); 
                    break;
                }
                case 3:
                {
                    const float range = (max_log2 - min_log2) * 0.2;
                    enhanced_linear = float3(range, range * 0.5, 0.0);
                    break;
                }
                case 4:
                {
                    const float enhancement = abs(log2_diff) * effective_strength * 2.0;
                    enhanced_linear = lerp(float3(0, 0, 1), float3(1, 0, 0), enhancement);
                    break;
                }
                case 6:
                {
                    const float total_pixels = 3.14159 * effective_radius * effective_radius;
                    const float efficiency = (float)pixels_processed / max(total_pixels, 1.0);
                    enhanced_linear = lerp(float3(1, 0, 0), float3(0, 1, 0), efficiency);
                    break;
                }
                case 8:
                {
                    float radius_normalized = (float)effective_radius / 16.0;
                    enhanced_linear = radius_normalized.xxx;
                    
                    int2 pixel = int2(vpos.xy);
                    bool is_boundary = ((pixel.x % max(effective_radius, 1)) == 0) || 
                                      ((pixel.y % max(effective_radius, 1)) == 0);
                    if (is_boundary)
                        enhanced_linear = float3(1, 0, 1);
                    break;
                }
            }
        }
    }
    
    // --- 7. Final Validation and Encoding ---
    if (any(isnan(enhanced_linear) || isinf(enhanced_linear)))
    {
        enhanced_linear = color_linear;
    }
    
    fragColor.rgb = ColorScience::EncodeFromLinear(enhanced_linear);
    fragColor.a = 1.0;
    
    // --- 8. Show Performance Statistics Display ---
    if (bShowStatistics)
    {
        // Create a dark background for text area
        if (center_pos.x < 320 && center_pos.y < 100)
        {
            fragColor.rgb *= 0.3;
            
            // Simple visualization bars for different metrics
            const float bar_width = 300.0;
            const float bar_height = 15.0;
            // Bar 1: Effective Radius (y: 10-25)
            if (center_pos.y >= 10 && center_pos.y < 10 + bar_height)
            {
                float radius_normalized = (float)effective_radius / 16.0; // Max radius is 16
                if ((float)center_pos.x / bar_width < radius_normalized)
                {
                    fragColor.rgb = lerp(float3(0.2, 0.2, 1), float3(1, 0.2, 0.2), radius_normalized);
                }
            }
            // Bar 2: Filter Efficiency (y: 30-45)
            else if (center_pos.y >= 30 && center_pos.y < 30 + bar_height)
            {
                const float total_pixels = 3.14159 * effective_radius * effective_radius;
                const float efficiency = (float)valid_samples / max(total_pixels, 1.0);
                if ((float)center_pos.x / bar_width < efficiency)
                {
                    fragColor.rgb = lerp(float3(1, 0, 0), float3(0, 1, 0), efficiency);
                }
            }
            // Bar 3: Black Pixel Ratio (y: 50-65)
            else if (center_pos.y >= 50 && center_pos.y < 50 + bar_height && pixels_processed > 0)
            {
                const float black_ratio = (float)black_pixels_skipped / (float)pixels_processed;
                if ((float)center_pos.x / bar_width < black_ratio)
                {
                    fragColor.rgb = float3(1, 0, 1); // Magenta for black pixels
                }
            }
            // Bar 4: Weight Sum (y: 70-85)
            else if (center_pos.y >= 70 && center_pos.y < 70 + bar_height)
            {
                const float weight_normalized = saturate(sum_weight / max((float)valid_samples, 1.0));
                if ((float)center_pos.x / bar_width < weight_normalized)
                {
                    fragColor.rgb = float3(1, 1, 0); // Yellow for weights
                }
            }
        }
    }
}

// ==============================================================================
// Technique
// ==============================================================================

technique lilium__bilateral_contrast <
    ui_label = "Lilium: Physically Correct Bilateral Contrast v6.1.0";
    ui_tooltip = "Academically rigorous local contrast enhancement with pure infinity handling.\n\n"
                 "New in v6.1.0:\n"
                 "• REMOVED: Tone protection features for cleaner, simpler code\n"
                 "• FIXED: Adaptive radius now uses smoothstep for proper clamping\n"
                 "• AGGRESSIVE: Adaptive radius can reduce to 1% for massive performance gains\n"
                 "• IMPROVED: Debug visualizations for extreme reduction ranges\n\n"
                 "Features:\n"
                 "• Correct log2 luminance ratio processing\n"
                 "• Pure Gaussian bilateral filtering\n"
                 "• Variance-based adaptive strength\n"
                 "• Extreme adaptive radius optimization (1% to 100%)\n"
                 "• SDR/HDR support with proper color preservation\n"
                 "• Numerical stability through Kahan summation\n\n"
                 "See shader header for content-specific tuning guide.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_BilateralContrast;
    }
}