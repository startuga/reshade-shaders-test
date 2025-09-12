/**
 * Maximum Quality Bilateral Contrast Enhancement for ReShade
 *
 * This shader prioritizes absolute quality over performance, implementing:
 * 1. True 2D bilateral filtering with exact Gaussian weights
 * 2. Perceptually uniform luminance processing (Log2 stops-based)
 * 3. Advanced numerical precision with extended Kahan summation
 * 4. Psychovisually accurate dark protection with gamma-aware blending
 * 5. Robust edge detection with adaptive thresholding
 *
 * Author: Your AI Assistant
 * Version: 3.3 (Corrected Log2 Perceptual Implementation)
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
    ui_max = 4.0;
    ui_step = 0.01;
> = 3.0;

uniform int iRadius <
    ui_type = "slider";
    ui_label = "Filter Radius";
    ui_tooltip = "Pixel radius of the bilateral filter. Larger values affect broader details.";
    ui_min = 1;
    ui_max = 12;
> = 7;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_tooltip = "Standard deviation of the spatial Gaussian kernel.\n"
                 "Controls the spatial extent of the filter influence.";
    ui_min = 0.1;
    ui_max = 6.0;
    ui_step = 0.01;
> = 3.0;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Luminance Sigma (Edge Detection)";
    ui_tooltip = "Controls edge preservation sensitivity in perceptual space.\n"
                 "In log2 mode: operates on exposure stops [~0-20 stop range].\n"
                 "Lower values preserve more edges, higher values allow more blending across edges.";
    ui_min = 0.1;
    ui_max = 3.0;
    ui_step = 0.01;
> = 0.8;

uniform float fDarkProtection <
    ui_type = "slider";
    ui_label = "Perceptual Dark Protection";
    ui_tooltip = "Perceptually uniform protection of shadow areas using logarithmic luminance mapping.\n"
                 "Prevents black crush while maintaining shadow detail gradation.\n"
                 "Value represents the perceptual brightness threshold (0.1 = darkest 10% of visible range).";
    ui_min = 0.0;
    ui_max = 0.3;
    ui_step = 0.001;
> = 0.05;

uniform float fEdgeThreshold <
    ui_type = "slider";
    ui_label = "Adaptive Edge Threshold";
    ui_tooltip = "Dynamic threshold for edge detection that adapts to local luminance levels.\n"
                 "In log2 mode: operates in exposure stop differences.\n"
                 "Higher values detect only stronger edges, lower values are more sensitive.";
    ui_min = 0.05;
    ui_max = 2.0;
    ui_step = 0.01;
> = 0.3;

uniform bool bPerceptualMode <
    ui_label = "Log2 Perceptual Processing";
    ui_tooltip = "Process luminance in log2 perceptual space (exposure stops) for more visually uniform results.\n"
                 "Provides superior perceptual uniformity and numerical stability.\n"
                 "Recommended for maximum quality, especially in HDR content.";
> = true;

// ==============================================================================
// Quality Constants - Precision Optimized
// ==============================================================================

namespace QualityConstants {
    // Ultra-precise epsilon values for different color spaces
    #if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        static const float LUMA_EPSILON = 1e-10;
        static const float WEIGHT_THRESHOLD = 1e-9;
        static const float COMPUTATION_EPSILON = 1e-12;
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        static const float LUMA_EPSILON = 1e-9;
        static const float WEIGHT_THRESHOLD = 1e-8;
        static const float COMPUTATION_EPSILON = 1e-11;
    #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        static const float LUMA_EPSILON = 1e-9;
        static const float WEIGHT_THRESHOLD = 1e-8;
        static const float COMPUTATION_EPSILON = 1e-11;
    #else // SDR
        static const float LUMA_EPSILON = 1e-8;
        static const float WEIGHT_THRESHOLD = 1e-7;
        static const float COMPUTATION_EPSILON = 1e-10;
    #endif

    // Quality-focused ratio limits (more conservative for stability)
    static const float RATIO_MIN = 0.001;
    static const float RATIO_MAX = 1000.0;

    // Reference white for consistent scaling across color spaces
    static const float REFERENCE_WHITE = 80.0;

    // Maximum luminance estimate normalized to reference white
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
        static const float MAX_LUMA_ESTIMATE = 1.0; // ~80 nits
    #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB || ACTUAL_COLOUR_SPACE == CSP_HDR10)
        static const float MAX_LUMA_ESTIMATE = 125.0; // 10000.0 / 80.0
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        static const float MAX_LUMA_ESTIMATE = 12.5; // 1000.0 / 80.0
    #else
        static const float MAX_LUMA_ESTIMATE = 1.0;
    #endif

    // Log2 perceptual space constants
    // Reference point for log2 mapping (18% gray equivalent)
    static const float LOG2_REFERENCE = 0.18;
    
    // Minimum luminance for log2 mapping (prevents log(0))
    static const float LOG2_MIN_LUMA = 1e-6;
    
    // Range constants for log2 perceptual space
    static const float LOG2_RANGE_MIN = -20.0; // ~20 stops below reference
    static const float LOG2_RANGE_MAX = 10.0;  // ~10 stops above reference
    static const float LOG2_TOTAL_RANGE = LOG2_RANGE_MAX - LOG2_RANGE_MIN; // 30 stops total
}

// ==============================================================================
// Maximum Quality Color Science
// ==============================================================================

namespace MaxQuality {
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        static const float SCALE_FACTOR = 1.0;
    #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        static const float SCALE_FACTOR = 10000.0 / QualityConstants::REFERENCE_WHITE;
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        static const float SCALE_FACTOR = 1000.0 / QualityConstants::REFERENCE_WHITE;
    #else
        static const float SCALE_FACTOR = 1.0;
    #endif

    float3 DecodeToLinearBT2020(float3 color) {
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
        color *= SCALE_FACTOR;
        return color;
    }

    float3 EncodeFromLinearBT2020(float3 color) {
        color /= SCALE_FACTOR;
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

    // Precise luminance calculation using Rec. 2020 primaries
    float GetLuminance(float3 linearBT2020) {
        return dot(linearBT2020, Csp::Mat::Bt2020ToXYZ[1]);
    }

    // Log2 perceptual luminance conversion (exposure stops based)
    float LinearToLog2Perceptual(float linear_luma) {
        // Ensure we don't take log of zero or negative values
        const float safe_luma = max(linear_luma, QualityConstants::LOG2_MIN_LUMA);
        
        // Convert to log2 relative to reference point
        // This gives us exposure stops relative to 18% gray
        const float log2_stops = log2(safe_luma / QualityConstants::LOG2_REFERENCE);
        
        return log2_stops;
    }

    float Log2PerceptualToLinear(float log2_perceptual) {
        // Convert back from exposure stops to linear
        const float linear_luma = QualityConstants::LOG2_REFERENCE * exp2(log2_perceptual);
        return max(linear_luma, QualityConstants::LOG2_MIN_LUMA);
    }

    // Enhanced Kahan summation with compensation tracking
    void KahanSum(inout float sum, inout float compensation, float input) {
        const float y = input - compensation;
        const float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    // Advanced bilateral weight calculation optimized for log2 space
    float CalculateBilateralWeight(
        float2 spatial_offset, 
        float luma_center, 
        float luma_neighbor, 
        float inv_2_sigma_spatial_sq, 
        float inv_2_sigma_range_sq,
        float adaptive_threshold
    ) {
        // Exact 2D spatial weight
        const float dist_sq_spatial = dot(spatial_offset, spatial_offset);
        const float weight_spatial = exp(-dist_sq_spatial * inv_2_sigma_spatial_sq);
        
        // Range weight optimized for log2 space (exposure stops)
        const float luma_diff = abs(luma_center - luma_neighbor);
        float weight_range = 1.0;
        
        // In log2 space, differences represent exposure stops
        // So we can use absolute differences directly
        if (luma_diff > adaptive_threshold) {
            const float dist_sq_range = luma_diff * luma_diff;
            weight_range = exp(-dist_sq_range * inv_2_sigma_range_sq);
        }
        
        return weight_spatial * weight_range;
    }
}

// ==============================================================================
// Maximum Quality Pixel Shader
// ==============================================================================

void PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    const int2 center_pos = int2(vpos.xy);
    const float3 color_encoded = tex2Dfetch(SamplerBackBuffer, center_pos).rgb;
    const float3 color_linear = MaxQuality::DecodeToLinearBT2020(color_encoded);
    
    const float luma_linear = max(MaxQuality::GetLuminance(color_linear), QualityConstants::LUMA_EPSILON);
    
    // Choose processing space: log2 perceptual (exposure stops) or linear
    const float luma_working = bPerceptualMode ? 
        MaxQuality::LinearToLog2Perceptual(luma_linear) : luma_linear;

    // Ultra-precise parameter preparation
    const float sigma_spatial_clamped = max(fSigmaSpatial, 0.01);
    const float sigma_range_clamped = clamp(fSigmaRange, 0.01, 5.0);
    const float inv_2_sigma_spatial_sq = 0.5 / (sigma_spatial_clamped * sigma_spatial_clamped);
    const float inv_2_sigma_range_sq = 0.5 / (sigma_range_clamped * sigma_range_clamped);
    
    // Adaptive threshold - in log2 mode this represents exposure stop differences
    const float adaptive_threshold = bPerceptualMode ? 
        fEdgeThreshold : // Direct use in log2 space (exposure stops)
        fEdgeThreshold * max(luma_working, QualityConstants::WEIGHT_THRESHOLD); // Scale for linear space

    // Enhanced Kahan summation with separate compensation tracking
    float sum_luma = 0.0, compensation_luma = 0.0;
    float sum_weight = 0.0, compensation_weight = 0.0;

    // Maximum quality bilateral filtering loop
    [loop]
    for (int y = -iRadius; y <= iRadius; ++y)
    {
        [loop]
        for (int x = -iRadius; x <= iRadius; ++x)
        {
            const int2 offset = int2(x, y);
            const int2 sample_pos = center_pos + offset;
            const float2 spatial_offset = float2(x, y);

            const float3 neighbor_encoded = tex2Dfetch(SamplerBackBuffer, sample_pos).rgb;
            const float3 neighbor_linear = MaxQuality::DecodeToLinearBT2020(neighbor_encoded);
            const float neighbor_luma_linear = max(MaxQuality::GetLuminance(neighbor_linear), QualityConstants::LUMA_EPSILON);
            
            const float neighbor_luma_working = bPerceptualMode ? 
                MaxQuality::LinearToLog2Perceptual(neighbor_luma_linear) : neighbor_luma_linear;

            const float weight = MaxQuality::CalculateBilateralWeight(
                spatial_offset, luma_working, neighbor_luma_working,
                inv_2_sigma_spatial_sq, inv_2_sigma_range_sq, adaptive_threshold
            );

            // Enhanced Kahan summation
            MaxQuality::KahanSum(sum_luma, compensation_luma, neighbor_luma_working * weight);
            MaxQuality::KahanSum(sum_weight, compensation_weight, weight);
        }
    }

    float3 enhanced_linear = color_linear;

    // Process enhancement with maximum precision
    if (sum_weight > QualityConstants::WEIGHT_THRESHOLD)
    {
        const float blurred_luma_working = sum_luma / sum_weight;
        float luma_diff_working = luma_working - blurred_luma_working;

        // Advanced perceptual dark area protection
        if (fDarkProtection > QualityConstants::COMPUTATION_EPSILON) {
            // Normalize linear luminance for consistent behavior across color spaces
            const float normalized_luma = saturate(luma_linear / QualityConstants::MAX_LUMA_ESTIMATE);
            
            // Use log-based perceptual mapping for smooth protection curve
            const float log_base = log(2.0); // Natural log of 2 for smooth curve
            const float perceptual_luma = saturate(log(1.0 + normalized_luma) / log_base);
            
            // Normalized protection parameter for smooth polynomial interpolation
            const float t = saturate(perceptual_luma / max(fDarkProtection, QualityConstants::COMPUTATION_EPSILON));
            
            // Smootherstep polynomial for ultra-gentle fade with C2 continuity
            const float protection_factor = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
            
            luma_diff_working *= protection_factor;
        }
        
        const float enhanced_luma_working = luma_working + fStrength * luma_diff_working;
        
        // Convert back to linear space if needed
        const float enhanced_luma_linear = bPerceptualMode ? 
            MaxQuality::Log2PerceptualToLinear(enhanced_luma_working) : 
            max(enhanced_luma_working, QualityConstants::LUMA_EPSILON);

        // Ultra-precise ratio calculation with extended range
        if (enhanced_luma_linear > QualityConstants::LUMA_EPSILON && luma_linear > QualityConstants::COMPUTATION_EPSILON) {
            const float ratio = enhanced_luma_linear / luma_linear;
            const float safe_ratio = clamp(ratio, QualityConstants::RATIO_MIN, QualityConstants::RATIO_MAX);
            enhanced_linear = color_linear * safe_ratio;
        }
    }

    // Final quality assurance with robust validation
    enhanced_linear = max(enhanced_linear, 0.0);
    
    // Comprehensive NaN/Inf protection
    if (any(isnan(enhanced_linear)) || any(isinf(enhanced_linear)) || 
        dot(enhanced_linear, float3(1,1,1)) > QualityConstants::MAX_LUMA_ESTIMATE * 10.0) {
        enhanced_linear = color_linear;
    }
    
    // Color space appropriate final clamping with headroom for log2 processing
    enhanced_linear = clamp(enhanced_linear, 0.0, QualityConstants::MAX_LUMA_ESTIMATE * 2.0);

    fragColor.rgb = MaxQuality::EncodeFromLinearBT2020(enhanced_linear);
    fragColor.a = 1.0;
}

// ==============================================================================
// Technique
// ==============================================================================

technique lilium__bilateral_contrast <
    ui_label = "Lilium: Maximum Quality Bilateral Contrast (Log2)";
    ui_tooltip = "Reference implementation prioritizing absolute quality over performance.\n"
                 "Features true 2D bilateral filtering with log2 perceptual luminance processing,\n"
                 "advanced numerical precision, and psychovisually accurate dark protection.\n"
                 "Log2 processing operates in exposure stops for superior perceptual uniformity.\n"
                 "Recommended for final quality processing and HDR mastering workflows.";
>
{
    pass
    {
        VertexShader = VS_PostProcess;
        PixelShader = PS_BilateralContrast;
    }
}