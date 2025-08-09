/*
 * bilateral-contrast-fast.fx
 *
 * This shader must be used with the "colour_space.fxh" header file.
 *
 * This is a performance-optimized shader that enhances texture clarity using
 * a bilateral filter operating on linear luminance.
 */

#include "colour_space.fxh" // By lilium (https://github.com/EndlesslyFlowering/ReShade_HDR_shaders)

/**
 * Bilateral Local Contrast Enhancement (Fast Edition)
 *
 * Developed by Gemini, reviewed and refined with community feedback.
 *
 * Key Features:
 * - Operates on linear luminance for a significant performance gain over
 *   perceptually uniform color spaces.
 * - This is a clean, single-pass, full-resolution implementation.
 * - Automatically handles SDR and HDR content via the lilium header.
 */

//==============================================================================
// UI Configuration
//==============================================================================

uniform float Strength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0;
    ui_label = "Contrast Strength";
    ui_tooltip = "Controls the intensity of the contrast enhancement.";
> = 1.0;

uniform int Radius <
    ui_type = "slider";
    ui_min = 1; ui_max = 10;
    ui_label = "Filter Radius (Kernel Size)";
    ui_tooltip = "Pixel radius of the bilateral filter.\nEven small increases have a large performance impact.";
> = 5;

uniform float SigmaSpatial <
    ui_type = "slider";
    ui_min = 0.5; ui_max = 8.0;
    ui_step = 0.1;
    ui_label = "Spatial Sigma";
    ui_tooltip = "Controls the blurriness/spread of the spatial filter.";
> = 2.0;

uniform float SigmaRange <
    ui_type = "slider";
    ui_min = 0.01; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Sigma Range (Luminance Preservation)";
    ui_tooltip = "Linear luminance similarity threshold.\nHigher values smooth more details; lower values preserve more edges.";
> = 0.15;


//==============================================================================
// Main Pixel Shader
//==============================================================================

float4 PS_BilateralContrast(float4 pos : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    // --- Get Center Luminance ---
    float3 linear_rgb;
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
    linear_rgb = Csp::Trc::SrgbTo::Linear(tex2D(SamplerBackBuffer, texcoord).rgb);
#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    linear_rgb = tex2D(SamplerBackBuffer, texcoord).rgb;
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
    linear_rgb = Csp::Trc::PqTo::Nits(tex2D(SamplerBackBuffer, texcoord).rgb);
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
    linear_rgb = Csp::Trc::HlgTo::Linear(tex2D(SamplerBackBuffer, texcoord).rgb) * 1000.0;
#endif

    float luma_center;
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    luma_center = dot(linear_rgb, Csp::Mat::Bt709ToXYZ[1].rgb);
#else
    luma_center = dot(linear_rgb, Csp::Mat::Bt2020ToXYZ[1].rgb);
#endif

    // --- Apply Bilateral Filter ---
    float sigma_spatial_sq = pow(SigmaSpatial, 2.0);
    float sigma_range_sq = pow(SigmaRange, 2.0);

    float total_w = 0.0;
    float total_l = 0.0;

    for (int y = -Radius; y <= Radius; ++y)
    {
        for (int x = -Radius; x <= Radius; ++x)
        {
            float2 offset = float2(x, y) * PIXEL_SIZE;
            float3 linear_sample;
        #if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
            linear_sample = Csp::Trc::SrgbTo::Linear(tex2Dlod(SamplerBackBuffer, float4(texcoord + offset, 0, 0)).rgb);
        #elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
            linear_sample = tex2Dlod(SamplerBackBuffer, float4(texcoord + offset, 0, 0)).rgb;
        #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
            linear_sample = Csp::Trc::PqTo::Nits(tex2Dlod(SamplerBackBuffer, float4(texcoord + offset, 0, 0)).rgb);
        #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
            linear_sample = Csp::Trc::HlgTo::Linear(tex2Dlod(SamplerBackBuffer, float4(texcoord + offset, 0, 0)).rgb) * 1000.0;
        #endif

            float luma_sample;
        #if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)
            luma_sample = dot(linear_sample, Csp::Mat::Bt709ToXYZ[1].rgb);
        #else
            luma_sample = dot(linear_sample, Csp::Mat::Bt2020ToXYZ[1].rgb);
        #endif

            float weight_spatial = exp(-(x*x + y*y) / (2.0 * sigma_spatial_sq));
            float l_diff = luma_sample - luma_center;
            float weight_range = exp(-(l_diff*l_diff) / (2.0 * sigma_range_sq));
            float current_weight = weight_spatial * weight_range;

            total_w += current_weight;
            total_l += luma_sample * current_weight;
        }
    }
    
    float filtered_L = total_l / max(total_w, 1e-6);

    // --- Apply Enhancement ---
    float detail = luma_center - filtered_L;
    float luma_ratio = (luma_center + detail * Strength) / max(luma_center, 1e-6);
    float3 new_linear_rgb = linear_rgb * luma_ratio;

    // --- Convert Back to Output Space ---
    float3 final_rgb;
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
    final_rgb = Csp::Trc::LinearTo::Srgb(new_linear_rgb);
    return float4(saturate(final_rgb), 1.0);
#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    return float4(new_linear_rgb, 1.0);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
    final_rgb = Csp::Trc::NitsTo::Pq(new_linear_rgb);
    return float4(final_rgb, 1.0);
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
    final_rgb = Csp::Trc::NitsTo::Hlg(new_linear_rgb);
    return float4(final_rgb, 1.0);
#endif
}

//==============================================================================
// Technique Definition
//==============================================================================
technique BilateralContrast_Fast
<
    ui_label = "Bilateral Contrast (Fast)";
    ui_tooltip = "A fast, single-pass version of the Bilateral Contrast shader that operates on linear luminance.";
>
{
    pass
    {
        VertexShader = VS_PostProcess;
        PixelShader  = PS_BilateralContrast;
    }
}