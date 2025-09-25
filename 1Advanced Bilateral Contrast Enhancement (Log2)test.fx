/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement for ReShade
 *
 * This shader implements academically rigorous bilateral filtering with:
 * 1. Correct log2 luminance ratio processing (not absolute values)
 * 2. Proper Gaussian range kernel without threshold artifacts
 * 3. Smoothed, perception-aligned shadow protection (Corrected)
 * 4. Unified SDR/HDR processing with correct reference whites
 * 5. Physically accurate color space conversions
 * 6. Color-preserving luminance clamping for HDR (Corrected)
 *
 * Version: 4.2.3 (Fixed Compile-Time Constant Issue)
 */
#include "ReShade.fxh"
#include "lilium__include/colour_space.fxh" // https://github.com/EndlesslyFlowering/ReShade_HDR_shaders/blob/master/Shaders/lilium__include/colour_space.fxh
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
uniform float fPeakNits <
ui_type = "slider";
ui_min = 80;
ui_max = 10000;
ui_step = 100;
> = 100;
uniform float fDarkProtection <
    ui_type = "slider";
    ui_label = "Shadow Protection (% of Reference)";
    ui_tooltip = "Threshold below which contrast is smoothly reduced, preventing black crush.\n"
                 "Value is a percentage of the reference white level.";
    ui_min = 0.0;
    ui_max = 0.2;
    ui_step = 0.001;
> = 0.0;
uniform bool bDebugMode <
    ui_label = "Debug: Show Log2 Luminance";
    ui_tooltip = "Visualize the log2 luminance ratios being processed.\n"
                 "Gray=0 stops, Black=-10 stops, White=+3 stops.";
> = false;
// ==============================================================================
// Constants & Utilities
// ==============================================================================
namespace Constants
{
    static const float REFERENCE_WHITE_NITS = 100.0; // Diffuse white
    // REFERENCE_WHITE_LINEAR moved to runtime computation to fix compile error (uniform dependency)
   
    // Precision and stability constants
    static const float LUMA_EPSILON = 1e-8f;
    static const float WEIGHT_THRESHOLD = 1e-7f;
    static const float RATIO_MIN = 0.000001f;
    static const float RATIO_MAX = 100.0000f;
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
    float3 DecodeToLinear(float3 color, float reference_white_linear)
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
       
        return color * reference_white_linear;
    }
    float3 EncodeFromLinear(float3 color, float reference_white_linear)
    {
        color /= reference_white_linear;
       
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
    float GetLuminance(const float3 linearBT2020)
    {
        return dot(linearBT2020, Csp::Mat::Bt2020ToXYZ[1]);
    }
    float LinearToLog2Ratio(const float linear_luma, float reference_white_linear)
    {
        float ratio = linear_luma / reference_white_linear;
        return log2(ratio);
    }
    float Log2RatioToLinear(const float log2_ratio, float reference_white_linear)
    {
        return reference_white_linear * exp2(log2_ratio);
    }
}
// ==============================================================================
// Bilateral Filtering Core
// ==============================================================================
namespace BilateralFilter
{
    float CalculateWeight(
        const float2 spatial_offset,
        const float log2_center,
        const float log2_neighbor,
        const float inv_2_sigma_spatial_sq,
        const float inv_2_sigma_range_sq
    ) {
        float weight_spatial = exp(-dot(spatial_offset, spatial_offset) * inv_2_sigma_spatial_sq);
        float log_ratio_diff_sq = pow(log2_center - log2_neighbor, 2.0);
        float weight_range = exp(-log_ratio_diff_sq * inv_2_sigma_range_sq);
        return weight_spatial * weight_range;
    }
}
// ==============================================================================
// Pixel Shader
// ==============================================================================
void PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    // Compute runtime reference white (fixes compile-time constant error)
    const float reference_white_linear = Constants::REFERENCE_WHITE_NITS / fPeakNits;
   
    // 1. Fetch and decode to a standard linear color space
    const int2 center_pos = int2(vpos.xy);
    const float3 color_encoded = tex2Dfetch(SamplerBackBuffer, center_pos).rgb;
    const float3 color_linear = ColorScience::DecodeToLinear(color_encoded, reference_white_linear);
   
    // 2. Convert to log2 perceptual space for processing
    const float luma_linear = max(ColorScience::GetLuminance(color_linear), Constants::LUMA_EPSILON);
    const float log2_ratio_center = ColorScience::LinearToLog2Ratio(luma_linear, reference_white_linear);
   
    // 3. Handle Debug Visualization
    if (bDebugMode)
    {
        const float debug_value = saturate((log2_ratio_center + 10.0) / 13.0);
        fragColor = float4(debug_value.xxx, 1.0);
        return;
    }
   
    // 4. Prepare filter parameters
    const float sigma_spatial_clamped = max(fSigmaSpatial, 0.01);
    const float sigma_range_clamped = max(fSigmaRange, 0.01);
    const float inv_2_sigma_spatial_sq = 0.5 / (sigma_spatial_clamped * sigma_spatial_clamped);
    const float inv_2_sigma_range_sq = 0.5 / (sigma_range_clamped * sigma_range_clamped);
   
    float sum_log2 = 0.0, compensation_log2 = 0.0;
    float sum_weight = 0.0, compensation_weight = 0.0;
   
    // 5. Main Bilateral Filtering Loop
    [loop]
    for (int y = -iRadius; y <= iRadius; ++y)
    {
        [loop]
        for (int x = -iRadius; x <= iRadius; ++x)
        {
            const float2 spatial_offset = float2(x, y);
            const int2 sample_pos = center_pos + int2(x, y);
           
            const float3 neighbor_linear = ColorScience::DecodeToLinear(tex2Dfetch(SamplerBackBuffer, sample_pos).rgb, reference_white_linear);
            const float neighbor_luma_linear = max(ColorScience::GetLuminance(neighbor_linear), Constants::LUMA_EPSILON);
            const float log2_ratio_neighbor = ColorScience::LinearToLog2Ratio(neighbor_luma_linear, reference_white_linear);
           
            const float weight = BilateralFilter::CalculateWeight(
                spatial_offset, log2_ratio_center, log2_ratio_neighbor,
                inv_2_sigma_spatial_sq, inv_2_sigma_range_sq
            );
           
            FxUtils::KahanSum(sum_log2, compensation_log2, log2_ratio_neighbor * weight);
            FxUtils::KahanSum(sum_weight, compensation_weight, weight);
        }
    }
   
    float3 enhanced_linear = color_linear;
   
    // 6. Apply Contrast Enhancement
    if (sum_weight > Constants::WEIGHT_THRESHOLD)
    {
        const float blurred_log2_ratio = sum_log2 / sum_weight;
        float log2_diff = log2_ratio_center - blurred_log2_ratio;
       
        // Instead of a hard cutoff, use smoothstep for a perception-aligned gentle fade of the effect in deep shadows.
        if (fDarkProtection > 1e-6) {
    float protection_threshold = fDarkProtection * reference_white_linear; // Linear threshold
    float protection_log2 = ColorScience::LinearToLog2Ratio(protection_threshold, reference_white_linear);
    float fade_width = 2.0 * fSigmaRange; // Tie to range sigma for consistency
    float protection_factor = smoothstep(protection_log2 - fade_width,
                                        protection_log2,
                                        log2_ratio_center);
    log2_diff *= protection_factor;
}
       
        const float enhanced_log2_ratio = log2_ratio_center + fStrength * log2_diff;
        const float enhanced_luma_linear = ColorScience::Log2RatioToLinear(enhanced_log2_ratio, reference_white_linear);
       
        if (luma_linear > Constants::LUMA_EPSILON)
        {
            const float ratio = enhanced_luma_linear / luma_linear;
            enhanced_linear = color_linear * clamp(ratio, Constants::RATIO_MIN, Constants::RATIO_MAX);
        }
    }
   
    // 7. Final Validation and Encoding
    enhanced_linear = max(enhanced_linear, 0.0);
   
    if (any(isnan(enhanced_linear)) || any(isinf(enhanced_linear)))
    {
        enhanced_linear = color_linear;
    }
   
    fragColor.rgb = ColorScience::EncodeFromLinear(enhanced_linear, reference_white_linear);
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
                 "Unified SDR/HDR processing with color preservation";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_BilateralContrast;
    }
}