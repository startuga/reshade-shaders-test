/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement for ReShade
 *
 * This shader implements academically rigorous bilateral filtering with:
 * 1. Correct log2 luminance ratio processing (not absolute values)
 * 2. Proper Gaussian range kernel without threshold artifacts
 * 3. Smoothed, perception-aligned shadow protection
 * 4. SDR/HDR processing with proper color space conversions
 * 5. Variance-based adaptive strength for superior detail detection
 * 6. True circular loop sampling for optimal iteration count
 * 7. Zero-overhead adaptive radius via hardware gradients
 * 8. Performance monitoring and debug modes
 * 9. Content-aware tuning controls for different source material
 *
 * Version: 5.0.6
 * 
 * Changelog:
 * 5.0.6 - FIXED: Adaptive Strength calculation logic to be more intuitive and powerful.
 *         OPTIMIZED: Bilateral filter loop bounds are now tighter, preventing wasted iterations.
 *         FIXED: Minor typo in technique UI label.
 * 5.0.5 - FIXED CRITICAL BUG: Center pixel was sampled twice (once in init, once in loop).
 *         FIXED: Loop now correctly skips center pixel.
 *         IMPROVED: GetLuminance logic clarified for all color spaces.
 *         IMPROVED: Adaptive Strength calculation for intuitive 'fAdaptiveAmount' range.
 * 5.0.4 - Fixed adaptive strength hybrid mode to use weighted average for smooth transitions.
 *         Added advanced exposed controls for content-specific tuning.
 *         Added comprehensive content-type tuning documentation.
 * 5.0.3 - Code quality improvements and constant organization.
 * 5.0.2 - Replaced gradient estimation with zero-cost hardware derivatives (ddx/ddy).
 *         Implemented true circular loop sampling to reduce total iterations by ~21.46%.
 * 5.0.1 - Branchless optimizations.
 * 5.0.0 - Initial feature-rich implementation with variance-based adaptation.
 * 
 * CONTENT-SPECIFIC TUNING GUIDE
 * ==============================
 * 
 * For optimal results, adjust advanced tuning parameters based on content type:
 * 
 * HIGH-FREQUENCY CONTENT (Hair, Foliage, Fabric, Fine Textures):
 *   - Gradient Sensitivity: 50-75 (maintains larger radii in detailed areas)
 *   - Variance Weight: 0.7 (keep default)
 *   - Effect: More natural enhancement of fine structures without over-sharpening
 * 
 * NOISY INPUT (Low-Light, High-ISO, Compressed Video, Film Grain):
 *   - Gradient Sensitivity: 100 (keep default)
 *   - Variance Weight: 0.8-0.9 (conservative, resists noise amplification)
 *   - Adaptive Curve: 2.0-2.5 (reduces sensitivity to noise-inflated variance)
 *   - Effect: Cleaner results without enhancing noise artifacts
 * 
 * CLEAN CGI (Renders, Vector Graphics, Game Screenshots):
 *   - Gradient Sensitivity: 100 (keep default)
 *   - Variance Weight: 0.5-0.6 (balanced toward dynamic range)
 *   - Adaptive Curve: 1.0-1.3 (more aggressive, no noise concerns)
 *   - Effect: Better detection of both geometric and textural detail
 * 
 * SMOOTH CONTENT (Portraits, Skies, Gradients, Soft Lighting):
 *   - Gradient Sensitivity: 125-150 (aggressive radius reduction in flat areas)
 *   - Variance Weight: 0.7 (keep default)
 *   - Range Sigma: 0.4-0.5 (softer edge preservation)
 *   - Effect: Gentler enhancement that preserves smooth transitions
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

uniform float fDarkProtection <
    ui_type = "slider";
    ui_label = "Shadow Protection (Linear Threshold)";
    ui_tooltip = "Linear luminance threshold below which contrast is smoothly reduced.\n"
                 "Prevents crushing of shadow details.";
    ui_min = 0.0;
    ui_max = 0.2;
    ui_step = 0.001;
    ui_category = "Tone Protection";
> = 0.0;

uniform float fHighlightProtection <
    ui_type = "slider";
    ui_label = "Highlight Protection (Linear Threshold)";
    ui_tooltip = "Linear luminance threshold above which contrast is smoothly reduced.\n"
                 "Prevents blowing out highlights. 0 = disabled.";
    ui_min = 0.0;
    ui_max = 10.0;
    ui_step = 0.01;
    ui_category = "Tone Protection";
> = 0.0;

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
    ui_min = 25.0;
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
    ui_items = "Off\0Show Weights\0Show Local Variance\0Show Dynamic Range\0Show Enhancement Map\0Show Adaptive Radius\0Performance Heatmap\0";
    ui_category = "Debug";
> = 0;

uniform bool bShowStatistics <
    ui_label = "Show Performance Statistics";
    ui_tooltip = "Display filter efficiency metrics in top-left corner.";
    ui_category = "Debug";
> = false;

// ==============================================================================
// Constants & Utilities
// ==============================================================================

namespace Constants
{
    // Precision and stability constants
    static const float LUMA_EPSILON = 1e-8f;
    static const float WEIGHT_THRESHOLD = 1e-7f;
    static const float RATIO_MAX = 8.0f;   // exp2(3.0)
    static const float RATIO_MIN = 0.125f; // exp2(-3.0)
    
    // Adaptive strength constants
    static const float MIN_DYNAMIC_RANGE = 0.05f;  // Minimum dynamic range in stops
    static const float MAX_DYNAMIC_RANGE = 10.0f;  // Maximum expected dynamic range in stops
    static const float MIN_VARIANCE_SQ = 0.001f * 0.001f;      // Minimum variance threshold
    static const float MAX_VARIANCE_SQ = 2.0f * 2.0f;          // Maximum expected variance in log2 space
    
    // Optimization constants
    static const float SPATIAL_CUTOFF_SIGMA = 3.0f; // Process within 3 sigma (99.7% of Gaussian)
}

namespace FxUtils
{
    // Kahan summation for improved numerical stability
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
            case 1: return 5;  // Balanced
            case 2: return 7;  // Quality
            case 3: return 10; // Ultra
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
            case 3: return 4.0;  // Ultra
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
            // Convert to linear BT.2020 normalised for unified processing
            color = Csp::Mat::ScRgbTo::Bt2020Normalised(color);
        #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
            // Decode PQ to linear BT.2020 normalised
            color = Csp::Trc::PqTo::Linear(color);
        #endif
        
        return color;
    }

    float3 EncodeFromLinear(float3 color)
    {        
        #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
            color = ENCODE_SDR(color);
        #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
            // Convert back from linear BT.2020 normalised to scRGB (linear BT.709-derived)
            color = Csp::Mat::Bt2020NormalisedTo::ScRgb(color);
        #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
            // Encode linear BT.2020 normalised to PQ
            color = Csp::Trc::LinearTo::Pq(color);
        #endif
        
        return color;
    }

    float GetLuminance(const float3 linearColour)
    {
        // Lum coefficients match the final linear space produced by DecodeToLinear
        #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
            // Linear BT.709-derived coefficients
            return dot(linearColour, Csp::Mat::Bt709ToXYZ[1]);
        #else
            // Linear BT.2020-derived coefficients (used for SCRGB/HDR10 output space)
            return dot(linearColour, Csp::Mat::Bt2020ToXYZ[1]);
        #endif
    }

    float LinearToLog2Ratio(const float linear_luma)
    {
        return log2(max(linear_luma, Constants::LUMA_EPSILON));
    }

    float Log2RatioToLinear(const float log2_ratio)
    {
        return exp2(log2_ratio);
    }
}

// ==============================================================================
// Bilateral Filtering Core
// ==============================================================================

namespace BilateralFilter
{
    // OPTIMIZATION: Calculate adaptive radius based on local gradient
    int GetAdaptiveRadius(float log2_luma, int base_radius)
    {
        // Use hardware derivatives for a virtually zero-cost gradient approximation
        float gx = ddx(log2_luma);
        float gy = ddy(log2_luma);
        float gradient_sq = gx * gx + gy * gy;

        // Avoid sqrt by working in squared space
        // Reduce radius in flat areas, maintain in detailed areas
        const float GRADIENT_SQ_THRESHOLD = 0.25; // = 0.5^2
        const float radius_scale = smoothstep(0.0, GRADIENT_SQ_THRESHOLD, gradient_sq * fGradientSensitivity);
        const float adaptive_factor = lerp(0.5, 1.0, radius_scale);
        return max(1, (int)(base_radius * adaptive_factor + 0.5));
    }
    
    // Calculate adaptive strength using weighted average for smooth transitions
    float CalculateAdaptiveStrength(
        const float local_dynamic_range, const float local_variance,
        const float base_strength, const float adaptive_amount,
        const float adaptive_curve, const int mode)
    {
        float metric;
        if (mode == 0) {
            // Dynamic Range only
            metric = saturate((local_dynamic_range - Constants::MIN_DYNAMIC_RANGE) / 
                            (Constants::MAX_DYNAMIC_RANGE - Constants::MIN_DYNAMIC_RANGE));
        } else if (mode == 1) {
            // Variance only
            metric = saturate((local_variance - Constants::MIN_VARIANCE_SQ) / 
                            (Constants::MAX_VARIANCE_SQ - Constants::MIN_VARIANCE_SQ));
        } else {
            // Hybrid: Weighted average for smooth, stable combination
            const float range_metric = saturate((local_dynamic_range - Constants::MIN_DYNAMIC_RANGE) / 
                                              (Constants::MAX_DYNAMIC_RANGE - Constants::MIN_DYNAMIC_RANGE));
            const float variance_metric = saturate((local_variance - Constants::MIN_VARIANCE_SQ) / 
                                                  (Constants::MAX_VARIANCE_SQ - Constants::MIN_VARIANCE_SQ));
            
            // Use weighted average instead of max() for smooth transitions
            const float range_weight = 1.0 - fVarianceWeight;
            metric = variance_metric * fVarianceWeight + range_metric * range_weight;
        }
        
        const float modulation = pow(metric, adaptive_curve);
        // Lerp from base strength to a modulated strength (0x to 2x)
        const float adaptive_multiplier = lerp(1.0, modulation * 2.0, adaptive_amount);
        return base_strength * adaptive_multiplier;
    }
    
    // Protection function for shadows and highlights
    float CalculateToneProtection(float log2_luma, float dark_threshold, float bright_threshold)
    {
        float protection = 1.0;
        
        // Shadow protection
        if (dark_threshold > 1e-6)
        {
            const float dark_log2 = ColorScience::LinearToLog2Ratio(dark_threshold);
            // Smoothly fade to zero as log2_luma drops from dark_log2 + 0.5 to dark_log2 - 0.5
            protection *= smoothstep(dark_log2 - 0.5, dark_log2 + 0.5, log2_luma);
        }
        
        // Highlight protection
        if (bright_threshold > 1e-6)
        {
            const float bright_log2 = ColorScience::LinearToLog2Ratio(bright_threshold);
            // Smoothly fade to zero as log2_luma rises from bright_log2 - 0.5 to bright_log2 + 0.5
            protection *= 1.0 - smoothstep(bright_log2 - 0.5, bright_log2 + 0.5, log2_luma);
        }
        
        return protection;
    }
}

// ==============================================================================
// Main Pixel Shader
// ==============================================================================

void PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    const int2 center_pos = int2(vpos.xy);
    const float3 color_encoded = tex2Dfetch(SamplerBackBuffer, center_pos).rgb;
    const float3 color_linear = ColorScience::DecodeToLinear(color_encoded);
    const float luma_linear = max(ColorScience::GetLuminance(color_linear), Constants::LUMA_EPSILON);
    const float log2_ratio_center = ColorScience::LinearToLog2Ratio(luma_linear);
    
    int base_radius = QualitySettings::GetRadius();
    int effective_radius = base_radius;
    if (bAdaptiveRadius && base_radius > 2)
    {
        effective_radius = BilateralFilter::GetAdaptiveRadius(log2_ratio_center, base_radius);

        // Early return for adaptive radius visualization (before expensive filtering)
        if (iDebugMode == 5)
        {
            float radius_norm = (float)effective_radius / (float)QualitySettings::GetRadius();
            fragColor.rgb = lerp(float3(0, 0, 1), float3(1, 0, 0), radius_norm);
            fragColor.a = 1.0;
            return;
        }
    }
    
    const float sigma_spatial = QualitySettings::GetSpatialSigma();
    const float inv_2_sigma_spatial_sq = 0.5 / (sigma_spatial * sigma_spatial);
    const float inv_2_sigma_range_sq = 0.5 / (max(fSigmaRange, 0.01) * max(fSigmaRange, 0.01));
    
    // If enabled, cutoff_radius is 3 sigma, otherwise it's just the effective_radius
    const float cutoff_radius = bEnableSpatialCutoff ? min((float)effective_radius, Constants::SPATIAL_CUTOFF_SIGMA * sigma_spatial) : (float)effective_radius;
    const float cutoff_radius_sq = cutoff_radius * cutoff_radius;
    // OPTIMIZATION: Use integer truncation (floor) for the max loop bound.
    const int max_radius = (int)cutoff_radius;
    
    // Initialize sums with the center pixel's data (weight 1.0).
    float sum_log2 = log2_ratio_center, compensation_log2 = 0.0;
    float sum_log2_sq = log2_ratio_center * log2_ratio_center, compensation_log2_sq = 0.0;
    float sum_weight = 1.0, compensation_weight = 0.0;
    float min_log2 = log2_ratio_center, max_log2 = log2_ratio_center;
    int pixels_processed = 1; // Start with 1 for the center pixel

    // OPTIMIZATION: True circular loop.
    [loop]
    for (int y = -max_radius; y <= max_radius; ++y)
    {
        const float y_sq = (float)(y * y);
        // Calculate the horizontal extent for this row to form a circle
        const int x_max = (int)sqrt(max(0.0, cutoff_radius_sq - y_sq));
        
        [loop]
        for (int x = -x_max; x <= x_max; ++x)
        {
            // FIX: Skip the center pixel as it was initialized above.
            if (x == 0 && y == 0) continue; 

            pixels_processed++;
            const float r2 = (float)(x * x) + y_sq;
            
            const int2 sample_pos = center_pos + int2(x, y);
            const float3 neighbor_linear = ColorScience::DecodeToLinear(tex2Dfetch(SamplerBackBuffer, sample_pos).rgb);
            const float log2_ratio_neighbor = ColorScience::LinearToLog2Ratio(max(ColorScience::GetLuminance(neighbor_linear), Constants::LUMA_EPSILON));
            
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
    
    float3 enhanced_linear = color_linear;
    if (sum_weight > Constants::WEIGHT_THRESHOLD)
    {
        const float blurred_log2_ratio = sum_log2 / sum_weight;
        float log2_diff = log2_ratio_center - blurred_log2_ratio;

        float effective_strength = fStrength;
        if (bAdaptiveStrength)
        {
            const float local_dynamic_range = max_log2 - min_log2;
            const float mean_log2 = sum_log2 / sum_weight;
            const float mean_sq_log2 = sum_log2_sq / sum_weight;
            // Guard against floating point imprecision causing negative variance
            const float local_variance = max(Constants::MIN_VARIANCE_SQ, mean_sq_log2 - mean_log2 * mean_log2);
            
            effective_strength = BilateralFilter::CalculateAdaptiveStrength(
                local_dynamic_range, local_variance, fStrength, 
                fAdaptiveAmount, fAdaptiveCurve, iAdaptiveMode
            );
        }
        
        // Apply tone protection (shadows and highlights)
        const float tone_protection = BilateralFilter::CalculateToneProtection(
            log2_ratio_center, fDarkProtection, fHighlightProtection
        );
        log2_diff *= tone_protection;
        
        // Apply enhancement
        const float enhanced_log2_ratio = log2_ratio_center + effective_strength * log2_diff;
        const float enhanced_luma_linear = ColorScience::Log2RatioToLinear(enhanced_log2_ratio);
        
        // Preserve color ratios
        {
            const float inv_luma_linear = 1.0 / luma_linear;  // Already guaranteed > LUMA_EPSILON earlier
            const float ratio = enhanced_luma_linear * inv_luma_linear;
            enhanced_linear = color_linear * clamp(ratio, Constants::RATIO_MIN, Constants::RATIO_MAX);
        }
        
        // Debug visualizations
        if (iDebugMode > 0)
        {
            switch(iDebugMode)
            {
                case 1: // Show weights
                    enhanced_linear = float3(sum_weight, sum_weight, sum_weight);
                    break;
                    
                case 2: // Show local variance
                {
                    const float mean_log2 = sum_log2 / sum_weight;
                    const float mean_sq_log2 = sum_log2_sq / sum_weight;
                    const float variance = max(0.0, mean_sq_log2 - mean_log2 * mean_log2);
                    // Scale variance for visibility
                    enhanced_linear = float3(variance * 10.0, variance * 5.0, 0.0); 
                    break;
                }
                    
                case 3: // Show dynamic range
                {
                    const float range = (max_log2 - min_log2) * 0.2;
                    enhanced_linear = float3(range, range * 0.5, 0.0);
                    break;
                }
                    
                case 4: // Show enhancement map
                {
                    const float enhancement = abs(log2_diff) * effective_strength * 2.0;
                    enhanced_linear = lerp(float3(0, 0, 1), float3(1, 0, 0), enhancement);
                    break;
                }
                    
                case 6: // Performance heatmap
                {
                    const float efficiency = (float)pixels_processed / 
                                           (float)((effective_radius * 2 + 1) * (effective_radius * 2 + 1));
                    enhanced_linear = lerp(float3(1, 0, 0), float3(0, 1, 0), efficiency);
                    break;
                }
            }
        }
    }
    
    // Final validation and encoding
    enhanced_linear = max(enhanced_linear, 0.0);
    
    // NaN/Inf check - component-wise fallback for better recovery
    if (any(isnan(enhanced_linear)) || any(isinf(enhanced_linear)))
    {
        enhanced_linear.r = (isnan(enhanced_linear.r) || isinf(enhanced_linear.r)) ? color_linear.r : enhanced_linear.r;
        enhanced_linear.g = (isnan(enhanced_linear.g) || isinf(enhanced_linear.g)) ? color_linear.g : enhanced_linear.g;
        enhanced_linear.b = (isnan(enhanced_linear.b) || isinf(enhanced_linear.b)) ? color_linear.b : enhanced_linear.b;
    }
    
    // Encode back to output color space
    fragColor.rgb = ColorScience::EncodeFromLinear(enhanced_linear);
    fragColor.a = 1.0;
    
    // Show performance statistics overlay
    if (bShowStatistics && center_pos.x < 200 && center_pos.y < 20)
    {
        // Darken the background of the 200x20 region
        fragColor.rgb *= 0.3; 
        
        // Draw the performance bar between y=5 and y=15
        if (center_pos.y >= 5 && center_pos.y < 15) 
        {
            const float total_pixels = (effective_radius * 2 + 1) * (effective_radius * 2 + 1);
            const float efficiency = (float)pixels_processed / max(total_pixels, 1.0);
            
            float bar_position = (float)center_pos.x / 199.0;
            if (bar_position < efficiency)
            {
                // Green (good) to Red (bad)
                fragColor.rgb = lerp(float3(1, 0, 0), float3(0, 1, 0), efficiency);
            }
        }
    }
}

// ==============================================================================
// Technique
// ==============================================================================

technique lilium__bilateral_contrast <
    ui_label = "Lilium: Physically Correct Bilateral Contrast v5.0.6";
    ui_tooltip = "Academically rigorous local contrast enhancement with content-aware tuning.\n\n"
                 "New in v5.0.6:\n"
                 "• FIXED: More intuitive and powerful Adaptive Strength logic.\n"
                 "• OPTIMIZED: Bilateral filter loop bounds are now tighter, preventing wasted iterations.\n\n"
                 "Features:\n"
                 "• Correct log2 luminance ratio processing\n"
                 "• Pure Gaussian bilateral filtering\n"
                 "• Variance-based adaptive strength\n"
                 "• Content-aware optimization controls\n"
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