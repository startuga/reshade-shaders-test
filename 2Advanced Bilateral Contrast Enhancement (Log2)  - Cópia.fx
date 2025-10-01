/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement for ReShade
 *
 * This shader implements academically rigorous bilateral filtering with:
 * 1. Correct log2 luminance ratio processing (not absolute values)
 * 2. Proper Gaussian range kernel without threshold artifacts
 * 3. Smoothed, perception-aligned shadow protection
 * 4. SDR/HDR processing
 * 5. Physically accurate color space conversions
 * 6. Color-preserving luminance
 * 7. Adaptive exposure-based strength
 *
 * Version: 4.3.2
 */

#include "ReShade.fxh"
#include "lilium__include/colour_space.fxh"  // https://github.com/EndlesslyFlowering/ReShade_HDR_shaders/blob/master/Shaders/lilium__include/colour_space.fxh

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

uniform bool bAdaptiveStrength <
    ui_label = "Enable Adaptive Strength";
    ui_tooltip = "Automatically adjusts strength based on local dynamic range.\n"
                 "Reduces enhancement in low-contrast areas and increases it where detail is present.";
> = true;

uniform float fAdaptiveAmount <
    ui_type = "slider";
    ui_label = "Adaptive Strength Amount";
    ui_tooltip = "How much the local dynamic range affects the strength.\n"
                 "0.0 = No adaptation (always uses base strength).\n"
                 "1.0 = Full adaptation. Strength is reduced in low-contrast areas and can be boosted up to 2x in high-contrast areas.";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_step = 0.01;
> = 0.5;

uniform float fAdaptiveCurve <
    ui_type = "slider";
    ui_label = "Adaptive Response Curve";
    ui_tooltip = "Controls how dynamic range maps to strength modulation.\n"
                 "Lower values = More aggressive adaptation\n"
                 "Higher values = More conservative adaptation";
    ui_min = 0.5;
    ui_max = 3.0;
    ui_step = 0.01;
> = 1.5;

uniform int iRadius <
    ui_type = "slider";
    ui_label = "Filter Radius";
    ui_tooltip = "Pixel radius of the bilateral filter.\nLarger values affect broader spatial details.";
    ui_min = 1;
    ui_max = 12;
> = 7;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_tooltip = "Standard deviation of the spatial Gaussian kernel.";
    ui_min = 0.1;
    ui_max = 6.0;
    ui_step = 0.01;
> = 3.0;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Range Sigma (in Exposure Stops)";
    ui_tooltip = "Controls edge preservation in exposure stops:\n"
                 "0.3 = Sharp edges (preserves 1.2x luminance ratios)\n"
                 "0.5 = Moderate (preserves 1.4x luminance ratios)\n"
                 "1.0 = Soft edges (preserves 2.0x luminance ratios)\n"
                 "2.0 = Very soft (preserves 4.0x luminance ratios)";
    ui_min = 0.1;
    ui_max = 3.0;
    ui_step = 0.01;
> = 0.30;

uniform float fDarkProtection <
    ui_type = "slider";
    ui_label = "Shadow Protection (Linear Threshold)";
    ui_tooltip = "Linear luminance threshold below which contrast is smoothly reduced, preventing black crush.\n"
                 "Higher values protect more shadow detail.";
    ui_min = 0.0;
    ui_max = 0.2;
    ui_step = 0.001;
> = 0.0;

// ==============================================================================
// Constants & Utilities
// ==============================================================================

namespace Constants
{
    // Precision and stability constants
    static const float LUMA_EPSILON = 1e-8f;
    static const float WEIGHT_THRESHOLD = 1e-7f;
    static const float RATIO_MAX = 8.0f; // exp2(3.0)
    static const float RATIO_MIN = 0.25f; // exp2(-2.0)
    
    // Adaptive strength constants
    static const float MIN_DYNAMIC_RANGE = 0.1f;  // Minimum dynamic range in stops
    static const float MAX_DYNAMIC_RANGE = 10.0f; // Maximum expected dynamic range in stops
}

namespace FxUtils
{
    // Extended Kahan summation for improved numerical stability
    void KahanSum(inout float sum, inout float compensation, const float input)
    {
        const float y = input - compensation;
        const float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
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

    float GetLuminance(const float3 linearBt)
    {
        #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
            return dot(linearBt, Csp::Mat::Bt709ToXYZ[1]);
        #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
            return dot(linearBt, Csp::Mat::ScRgbToXYZ[1]);
        #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
            return dot(linearBt, Csp::Mat::Bt2020ToXYZ[1]);
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
    // OPTIMIZATION: CalculateWeight is now inlined in the main loop for early exit and combined exponentiation.
    
    // Calculate adaptive strength based on local dynamic range
    float CalculateAdaptiveStrength(const float local_dynamic_range, 
                                   const float base_strength,
                                   const float adaptive_amount,
                                   const float adaptive_curve)
    {
        // Normalize dynamic range to [0,1]
        const float normalized_range = saturate((local_dynamic_range - Constants::MIN_DYNAMIC_RANGE) / 
                                               (Constants::MAX_DYNAMIC_RANGE - Constants::MIN_DYNAMIC_RANGE));
        
        // Apply response curve - pow function shapes how dynamic range affects strength
        // Higher curve values make the response more conservative
        const float modulation = pow(normalized_range, adaptive_curve);
        
        // Blend between base strength and modulated strength
        // In low contrast areas (low dynamic range), strength is reduced
        // In high contrast areas (high dynamic range), strength is maintained or boosted
        const float adaptive_multiplier = lerp(1.0, modulation * 2.0, adaptive_amount);
        
        return base_strength * adaptive_multiplier;
    }
}

// ==============================================================================
// Pixel Shader
// ==============================================================================

void PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    // 1. Fetch and decode to a standard linear color space
    const int2 center_pos = int2(vpos.xy);
    const float3 color_encoded = tex2Dfetch(SamplerBackBuffer, center_pos).rgb;
    const float3 color_linear = ColorScience::DecodeToLinear(color_encoded);
    
    // 2. Convert to log2 space
    const float luma_linear = max(ColorScience::GetLuminance(color_linear), Constants::LUMA_EPSILON);
    const float log2_ratio_center = ColorScience::LinearToLog2Ratio(luma_linear);
        
    // 3. Prepare filter parameters
    const float sigma_spatial_clamped = max(fSigmaSpatial, 0.01);
    const float sigma_range_clamped = max(fSigmaRange, 0.01);
    const float inv_2_sigma_spatial_sq = 0.5 / (sigma_spatial_clamped * sigma_spatial_clamped);
    const float inv_2_sigma_range_sq = 0.5 / (sigma_range_clamped * sigma_range_clamped);
    
    float sum_log2 = 0.0, compensation_log2 = 0.0;
    float sum_weight = 0.0, compensation_weight = 0.0;
    
    float min_log2 = log2_ratio_center;
    float max_log2 = log2_ratio_center;
    
    // 4. Main Bilateral Filtering Loop
    [loop]
    for (int y = -iRadius; y <= iRadius; ++y)
    {
        const float y_sq = y * y;
        [loop]
        for (int x = -iRadius; x <= iRadius; ++x)
        {
            const float r2 = x * x + y_sq;
            const int2 sample_pos = center_pos + int2(x, y);
            
            const float3 neighbor_linear = ColorScience::DecodeToLinear(tex2Dfetch(SamplerBackBuffer, sample_pos).rgb);
            const float neighbor_luma_linear = max(ColorScience::GetLuminance(neighbor_linear), Constants::LUMA_EPSILON);
            const float log2_ratio_neighbor = ColorScience::LinearToLog2Ratio(neighbor_luma_linear);

            // OPTIMIZATION: Combined weight calculation (branchless).
            // This is kept from the previous version as it is a net positive.
            const float d = log2_ratio_center - log2_ratio_neighbor;
            const float spatial_exponent = -r2 * inv_2_sigma_spatial_sq;
            const float range_exponent = -(d * d) * inv_2_sigma_range_sq;
            const float weight = exp(spatial_exponent + range_exponent);
            
            // The original branch inside the loop is still necessary to avoid
            // floating point errors from tiny weights, but it's very cheap.
            if (weight > Constants::WEIGHT_THRESHOLD)
            {
                FxUtils::KahanSum(sum_log2, compensation_log2, log2_ratio_neighbor * weight);
                FxUtils::KahanSum(sum_weight, compensation_weight, weight);
                
                if (bAdaptiveStrength && weight > 0.1)
                // For adaptive strength, only consider significant neighbors for the dynamic range
                // calculation. This prevents distant pixels with low spatial weight from
                // disproportionately affecting the perceived local contrast.
                {
                    min_log2 = min(min_log2, log2_ratio_neighbor);
                    max_log2 = max(max_log2, log2_ratio_neighbor);
                }
            }
        }
    }
    
    // 5 & 6. Apply Enhancement and Final Encoding
    float3 enhanced_linear = color_linear;
    
    if (sum_weight > Constants::WEIGHT_THRESHOLD)
    {
        const float blurred_log2_ratio = sum_log2 / sum_weight;
        float log2_diff = log2_ratio_center - blurred_log2_ratio;
        
        float effective_strength = fStrength;
        if (bAdaptiveStrength)
        {
            const float local_dynamic_range = max_log2 - min_log2;
            effective_strength = BilateralFilter::CalculateAdaptiveStrength(
                local_dynamic_range, fStrength, fAdaptiveAmount, fAdaptiveCurve
            );
        }
        
        if (fDarkProtection > 1e-6)
        {
            const float protection_log2 = ColorScience::LinearToLog2Ratio(fDarkProtection);
            const float protection_factor = smoothstep(protection_log2 - 0.5, protection_log2 + 0.5, log2_ratio_center);
            log2_diff *= protection_factor;
        }
        
        const float enhanced_log2_ratio = log2_ratio_center + effective_strength * log2_diff;
        const float enhanced_luma_linear = ColorScience::Log2RatioToLinear(enhanced_log2_ratio);
        
        if (luma_linear > Constants::LUMA_EPSILON)
        {
            const float ratio = enhanced_luma_linear / luma_linear;
            enhanced_linear = color_linear * clamp(ratio, Constants::RATIO_MIN, Constants::RATIO_MAX);
        }
    }
    
    enhanced_linear = max(enhanced_linear, 0.0);
    
    if (any(isnan(enhanced_linear)) || any(isinf(enhanced_linear)))
    {
        enhanced_linear = color_linear;
    }
    
    fragColor.rgb = ColorScience::EncodeFromLinear(enhanced_linear);
    fragColor.a = 1.0;
}

// ==============================================================================
// Technique
// ==============================================================================

technique lilium__bilateral_contrast <
    ui_label = "Lilium: Physically Correct Bilateral Contrast";
    ui_tooltip = "Academically rigorous local contrast enhancement.\n"
                 "Features:\n"
                 "• Correct log2 luminance RATIO processing\n"
                 "• Pure Gaussian bilateral filtering (no artifacts)\n"
                 "• Adaptive exposure-based strength\n"
                 "• Perception-aligned shadow protection\n"
                 "• SDR/HDR processing with color preservation";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_BilateralContrast;
    }
}