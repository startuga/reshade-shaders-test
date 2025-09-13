/**
 * PHYSICALLY CORRECT Bilateral Contrast Enhancement for ReShade
 *
 * This shader implements academically rigorous bilateral filtering with:
 * 1. Correct log2 luminance ratio processing (not absolute values)
 * 2. Proper Gaussian range kernel without threshold artifacts
 * 3. Weber's Law compliant shadow protection
 * 4. Unified SDR/HDR processing with correct reference whites
 * 5. Physically accurate color space conversions
 *
 * Version: 4.0 (Complete Physical Accuracy Rewrite)
 */

#include "ReShade.fxh"
#include "lilium__include/colour_space.fxh"

// FIX: The line defining SamplerBackBuffer was removed from here,
// as it is already defined in ReShade.fxh.

// ==============================================================================
// UI Configuration
// ==============================================================================

uniform float fStrength <
    ui_type = "slider";
    ui_label = "Contrast Strength";
    ui_tooltip = "Controls the intensity of the micro-contrast enhancement.\n"
                 "Acts as a multiplier on the difference between local and filtered luminance.";
    ui_min = 0.0;
    ui_max = 4.0;
    ui_step = 0.01;
> = 2.0;

uniform int iRadius <
    ui_type = "slider";
    ui_label = "Filter Radius";
    ui_tooltip = "Pixel radius of the bilateral filter.\n"
                 "Larger values affect broader spatial details.";
    ui_min = 1;
    ui_max = 12;
> = 7;

uniform float fSigmaSpatial <
    ui_type = "slider";
    ui_label = "Spatial Sigma";
    ui_tooltip = "Standard deviation of the spatial Gaussian kernel.\n"
                 "Controls the spatial falloff of the filter influence.";
    ui_min = 0.1;
    ui_max = 6.0;
    ui_step = 0.01;
> = 3.0;

uniform float fSigmaRange <
    ui_type = "slider";
    ui_label = "Range Sigma (Luminance Ratio)";
    ui_tooltip = "Controls edge preservation in log2 space.\n"
                 "Value represents exposure stops difference tolerance:\n"
                 "0.5 = preserves 2^0.5 = 1.41x luminance ratios\n"
                 "1.0 = preserves 2^1.0 = 2.0x luminance ratios\n"
                 "2.0 = preserves 2^2.0 = 4.0x luminance ratios";
    ui_min = 0.1;
    ui_max = 3.0;
    ui_step = 0.01;
> = 0.8;

uniform float fDarkProtection <
    ui_type = "slider";
    ui_label = "Shadow Protection (% of Reference White)";
    ui_tooltip = "Weber's Law compliant shadow protection.\n"
                 "Value is percentage of reference white below which contrast is reduced:\n"
                 "0.05 = protect below 5% of reference (4 nits in SDR, 500 nits in HDR10)\n"
                 "0.10 = protect below 10% of reference (8 nits in SDR, 1000 nits in HDR10)";
    ui_min = 0.0;
    ui_max = 0.2;
    ui_step = 0.001;
> = 0.05;

uniform bool bDebugMode <
    ui_label = "Debug: Show Log2 Values";
    ui_tooltip = "Visualize the log2 luminance ratios being processed.\n"
                 "Gray = 0 stops (reference white), Black = -10 stops, White = +3 stops";
> = false;

// ==============================================================================
// Physically Accurate Constants
// ==============================================================================

// Reference white levels for each color space (in nits)
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    #define REFERENCE_WHITE_NITS 80.0
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
    #define REFERENCE_WHITE_NITS 10000.0
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
    #define REFERENCE_WHITE_NITS 1000.0
#else
    #define REFERENCE_WHITE_NITS 80.0
#endif

// Normalized reference white for linear RGB space
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
    #define REFERENCE_WHITE_LINEAR 1.0  // SDR is normalized to 1.0
#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    #define REFERENCE_WHITE_LINEAR 1.0  // scRGB uses SDR reference
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
    #define REFERENCE_WHITE_LINEAR 125.0  // 10000/80
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
    #define REFERENCE_WHITE_LINEAR 12.5   // 1000/80
#else
    #define REFERENCE_WHITE_LINEAR 1.0
#endif

// Precision constants
#define LUMA_EPSILON 1e-8
#define LOG2_MIN_RATIO 1e-6
#define WEIGHT_THRESHOLD 1e-7

// Ratio clamping for stability
#define RATIO_MIN 0.001
#define RATIO_MAX 1000.0

// ==============================================================================
// CORRECTED Color Space Conversions
// ==============================================================================

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
    
    // Scale to absolute luminance scale
    color *= REFERENCE_WHITE_LINEAR;
    return color;
}

float3 EncodeFromLinearBT2020(float3 color) {
    // Scale back from absolute luminance
    color /= REFERENCE_WHITE_LINEAR;
    
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

float GetLuminanceBT2020(float3 linearBT2020) {
    return dot(linearBT2020, Csp::Mat::Bt2020ToXYZ[1]);
}

// ==============================================================================
// CORRECTED Log2 Perceptual Processing
// ==============================================================================

// Convert linear luminance to log2 RATIO relative to reference white
float LinearToLog2Ratio(float linear_luma) {
    // Calculate ratio relative to reference white
    float ratio = linear_luma / REFERENCE_WHITE_LINEAR;
    
    // Clamp to prevent log(0)
    ratio = max(ratio, LOG2_MIN_RATIO);
    
    // Return log2 of the ratio (exposure stops from reference)
    return log2(ratio);
}

// Convert log2 ratio back to linear luminance
float Log2RatioToLinear(float log2_ratio) {
    // exp2 gives us the ratio, multiply by reference to get absolute
    return REFERENCE_WHITE_LINEAR * exp2(log2_ratio);
}

// ==============================================================================
// CORRECTED Bilateral Weight Calculation
// ==============================================================================

float CalculateBilateralWeight(
    float2 spatial_offset,
    float log2_center,
    float log2_neighbor,
    float inv_2_sigma_spatial_sq,
    float inv_2_sigma_range_sq
) {
    // Spatial Gaussian weight (unchanged)
    float dist_sq_spatial = dot(spatial_offset, spatial_offset);
    float weight_spatial = exp(-dist_sq_spatial * inv_2_sigma_spatial_sq);
    
    // CORRECTED: Always apply Gaussian range weight
    // In log2 space, difference is the ratio in stops
    float log_ratio_diff = abs(log2_center - log2_neighbor);
    float dist_sq_range = log_ratio_diff * log_ratio_diff;
    float weight_range = exp(-dist_sq_range * inv_2_sigma_range_sq);
    
    return weight_spatial * weight_range;
}

// Extended Kahan summation for numerical stability
void KahanSum(inout float sum, inout float compensation, float input) {
    float y = input - compensation;
    float t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
}

// ==============================================================================
// PHYSICALLY CORRECT Pixel Shader
// ==============================================================================

void PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD0, out float4 fragColor : SV_Target)
{
    int2 center_pos = int2(vpos.xy);
    float3 color_encoded = tex2Dfetch(SamplerBackBuffer, center_pos).rgb;
    float3 color_linear = DecodeToLinearBT2020(color_encoded);
    
    float luma_linear = max(GetLuminanceBT2020(color_linear), LUMA_EPSILON);
    
    // CORRECTED: Convert to log2 ratio space
    float log2_ratio_center = LinearToLog2Ratio(luma_linear);
    
    // Debug mode: visualize log2 ratios
    if (bDebugMode) {
        // Map -10 to +3 stops to 0-1 for visualization
        float debug_value = saturate((log2_ratio_center + 10.0) / 13.0);
        fragColor = float4(debug_value.xxx, 1.0);
        return;
    }
    
    // Prepare filter parameters
    float sigma_spatial_clamped = max(fSigmaSpatial, 0.01);
    float sigma_range_clamped = max(fSigmaRange, 0.01);
    float inv_2_sigma_spatial_sq = 0.5 / (sigma_spatial_clamped * sigma_spatial_clamped);
    float inv_2_sigma_range_sq = 0.5 / (sigma_range_clamped * sigma_range_clamped);
    
    // Kahan summation for numerical precision
    float sum_log2 = 0.0, compensation_log2 = 0.0;
    float sum_weight = 0.0, compensation_weight = 0.0;
    
    // Bilateral filtering loop
    [loop]
    for (int y = -iRadius; y <= iRadius; ++y)
    {
        [loop]
        for (int x = -iRadius; x <= iRadius; ++x)
        {
            int2 sample_pos = center_pos + int2(x, y);
            float2 spatial_offset = float2(x, y);
            
            float3 neighbor_encoded = tex2Dfetch(SamplerBackBuffer, sample_pos).rgb;
            float3 neighbor_linear = DecodeToLinearBT2020(neighbor_encoded);
            
            float neighbor_luma_linear = max(GetLuminanceBT2020(neighbor_linear), LUMA_EPSILON);
            
            // CORRECTED: Work in log2 ratio space
            float log2_ratio_neighbor = LinearToLog2Ratio(neighbor_luma_linear);
            
            // CORRECTED: Calculate weight with proper Gaussian range kernel
            float weight = CalculateBilateralWeight(
                spatial_offset,
                log2_ratio_center,
                log2_ratio_neighbor,
                inv_2_sigma_spatial_sq,
                inv_2_sigma_range_sq
            );
            
            // Accumulate with Kahan summation
            KahanSum(sum_log2, compensation_log2, log2_ratio_neighbor * weight);
            KahanSum(sum_weight, compensation_weight, weight);
        }
    }
    
    float3 enhanced_linear = color_linear;
    
    // Process enhancement
    if (sum_weight > WEIGHT_THRESHOLD)
    {
        float blurred_log2_ratio = sum_log2 / sum_weight;
        float log2_diff = log2_ratio_center - blurred_log2_ratio;
        
        // CORRECTED: Weber's Law compliant shadow protection
        if (fDarkProtection > 0.001) {
            // Protection threshold as absolute luminance
            float protection_threshold = fDarkProtection * REFERENCE_WHITE_LINEAR;
            
            // Weber-Fechner Law: sensitivity proportional to luminance
            // Gradually reduce enhancement below threshold
            float protection_factor = saturate(
                (luma_linear - protection_threshold) / 
                max(luma_linear, LUMA_EPSILON)
            );
            
            log2_diff *= protection_factor;
        }
        
        // Apply enhancement in log2 space
        float enhanced_log2_ratio = log2_ratio_center + fStrength * log2_diff;
        
        // Convert back to linear
        float enhanced_luma_linear = Log2RatioToLinear(enhanced_log2_ratio);
        
        // Calculate and apply ratio to preserve color
        if (enhanced_luma_linear > LUMA_EPSILON && luma_linear > LUMA_EPSILON) {
            float ratio = enhanced_luma_linear / luma_linear;
            float safe_ratio = clamp(ratio, RATIO_MIN, RATIO_MAX);
            enhanced_linear = color_linear * safe_ratio;
        }
    }
    
    // Final validation and clamping
    enhanced_linear = max(enhanced_linear, 0.0);
    
    // NaN/Inf protection
    if (any(isnan(enhanced_linear)) || any(isinf(enhanced_linear))) {
        enhanced_linear = color_linear;
    }
    
    // Clamp to maximum luminance for color space
    float max_luminance = REFERENCE_WHITE_LINEAR * 2.0; // Allow 1 stop headroom
    #if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        max_luminance = 125.0 * 2.0; // 20,000 nits
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        max_luminance = 12.5 * 2.0;  // 2,000 nits
    #endif
    
    enhanced_linear = min(enhanced_linear, max_luminance);
    
    fragColor.rgb = EncodeFromLinearBT2020(enhanced_linear);
    fragColor.a = 1.0;
}

// ==============================================================================
// Technique
// ==============================================================================

technique lilium__bilateral_contrast <
    ui_label = "Lilium: PHYSICALLY CORRECT Bilateral Contrast";
    ui_tooltip = "Academically rigorous implementation with:\n"
                 "• Correct log2 luminance RATIO processing (not absolute values)\n"
                 "• Proper Gaussian bilateral filtering (no threshold artifacts)\n"
                 "• Weber's Law compliant shadow protection\n"
                 "• Unified SDR/HDR processing with correct reference whites\n"
                 "• Debug mode to visualize log2 ratio processing\n\n"
                 "Sigma values represent exposure stop tolerances:\n"
                 "0.5 = ±0.5 stops (1.41x ratio), 1.0 = ±1 stop (2x ratio)";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_BilateralContrast;
    }
}