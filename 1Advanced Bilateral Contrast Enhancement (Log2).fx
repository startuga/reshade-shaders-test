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
 *
 * Version: 4.2.2
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
    ui_tooltip = "Controls edge preservation tolerance in log2 space (exposure stops).\n"
                 "0.5 = Preserves 1.4x luminance ratios\n"
                 "1.0 = Preserves 2.0x luminance ratios";
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
            color = FetchFromHdr10ToLinearLUT(color);
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
            return dot(linearBt, FetchFromHdr10ToLinearLUT);
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
    float CalculateWeight(const float2 spatial_offset,
        const float log2_center,
        const float log2_neighbor,
        const float inv_2_sigma_spatial_sq,
        const float inv_2_sigma_range_sq)
    {
        const float r2 = dot(spatial_offset, spatial_offset);
        const float w_spatial = exp(-r2 * inv_2_sigma_spatial_sq);
        const float d = log2_center - log2_neighbor;
        const float w_range = exp(-(d * d) * inv_2_sigma_range_sq);
        return w_spatial * w_range;
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
    
    // 2. Convert to log2 perceptual space for processing
    const float luma_linear = max(ColorScience::GetLuminance(color_linear), Constants::LUMA_EPSILON);
    const float log2_ratio_center = ColorScience::LinearToLog2Ratio(luma_linear);
        
    // 3. Prepare filter parameters
    const float sigma_spatial_clamped = max(fSigmaSpatial, 0.01);
    const float sigma_range_clamped = max(fSigmaRange, 0.01);
    const float inv_2_sigma_spatial_sq = 0.5 / (sigma_spatial_clamped * sigma_spatial_clamped);
    const float inv_2_sigma_range_sq = 0.5 / (sigma_range_clamped * sigma_range_clamped);
    
    float sum_log2 = 0.0, compensation_log2 = 0.0;
    float sum_weight = 0.0, compensation_weight = 0.0;
    
    // 4. Main Bilateral Filtering Loop
    [loop]
    for (int y = -iRadius; y <= iRadius; ++y)
    {
        [loop]
        for (int x = -iRadius; x <= iRadius; ++x)
        {
            const float2 spatial_offset = float2(x, y);
            const int2 sample_pos = center_pos + int2(x, y);
            
            const float3 neighbor_linear = ColorScience::DecodeToLinear(tex2Dfetch(SamplerBackBuffer, sample_pos).rgb);
            const float neighbor_luma_linear = max(ColorScience::GetLuminance(neighbor_linear), Constants::LUMA_EPSILON);
            const float log2_ratio_neighbor = ColorScience::LinearToLog2Ratio(neighbor_luma_linear);
            
            const float weight = BilateralFilter::CalculateWeight(
                spatial_offset, log2_ratio_center, log2_ratio_neighbor,
                inv_2_sigma_spatial_sq, inv_2_sigma_range_sq
            );
            
            FxUtils::KahanSum(sum_log2, compensation_log2, log2_ratio_neighbor * weight);
            FxUtils::KahanSum(sum_weight, compensation_weight, weight);
        }
    }
    
    float3 enhanced_linear = color_linear;
    
    // 5. Apply Contrast Enhancement
    if (sum_weight > Constants::WEIGHT_THRESHOLD)
    {
        const float blurred_log2_ratio = sum_log2 / sum_weight;
        float log2_diff = log2_ratio_center - blurred_log2_ratio;
        
        // Instead of a hard cutoff, use smoothstep for a perception-aligned gentle fade of the effect in deep shadows.
        if (fDarkProtection > 1e-6)
        {
            const float protection_log2 = ColorScience::LinearToLog2Ratio(fDarkProtection);
            const float protection_factor = smoothstep(protection_log2 - 0.5, protection_log2 + 0.5, log2_ratio_center);
            log2_diff *= protection_factor;
        }
        
        const float enhanced_log2_ratio = log2_ratio_center + fStrength * log2_diff;
        const float enhanced_luma_linear = ColorScience::Log2RatioToLinear(enhanced_log2_ratio);
        
        if (luma_linear > Constants::LUMA_EPSILON)
        {
            const float ratio = enhanced_luma_linear / luma_linear;
			enhanced_linear = color_linear * clamp(ratio, Constants::RATIO_MIN, Constants::RATIO_MAX);
        }
    }
    
    // 6. Final Validation and Encoding
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
                 "Correct log2 luminance RATIO processing\n"
                 "Pure Gaussian bilateral filtering (no artifacts)\n"
                 "Corrected perception-aligned shadow protection\n"
                 "SDR/HDR processing with color preservation";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_BilateralContrast;
    }
}