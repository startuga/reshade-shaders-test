/*
 * bilateral-contrast.fx
 *
 * This shader must be used with the "colour_space.fxh" header file.
 *
 * It enhances texture clarity using a bilateral filter operating in the
 * OkLab color space for true perceptual uniformity. This is a reference-quality
 * implementation adhering to modern color science principles for maximum
 * image quality.
 */

#include "colour_space.fxh" // By lilium (https://github.com/EndlesslyFlowering/ReShade_HDR_shaders)

/**
 * Bilateral Local Contrast Enhancement (Reference, OkLab Edition)
 *
 * Developed by Gemini, reviewed and refined with community feedback.
 *
 * Key Features:
 * - Operates on the L (lightness) component of the OkLab color space for
 *   superior perceptual uniformity.
 * - Utilizes high-precision color matrix and specification-correct constants
 *   provided by the included header.
 * - Employs float-based Kahan summation in the filter loop to minimize
 *   numerical error, ensuring maximum accuracy on all hardware.
 * - Automatically handles SDR, scRGB, HDR10, and HLG content via the
 *   lilium header's robust color management system.
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
    ui_min = 1; ui_max = 16;
    ui_label = "Filter Radius (Kernel Size)";
    ui_tooltip = "Pixel radius of the bilateral filter (e.g., Radius 6 = 13x13 kernel).\nLarger values affect broader details but are significantly more demanding on the GPU.";
> = 6;

uniform float SigmaSpatial <
    ui_type = "slider";
    ui_min = 0.5; ui_max = 8.0;
    ui_step = 0.1;
    ui_label = "Spatial Sigma";
    ui_tooltip = "Controls the blurriness/spread of the spatial filter.\nHigher values give more weight to distant pixels, resulting in a smoother effect.";
> = 2.0;

uniform float SigmaRange <
    ui_type = "slider";
    ui_min = 0.01; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Sigma Range (L Preservation)";
    ui_tooltip = "Perceptual lightness (OkLab L) similarity threshold.\nHigher values smooth more details; lower values preserve more edges.";
> = 0.15;


//==============================================================================
// Main Pixel Shader
//==============================================================================

float4 PS_BilateralContrast(float4 pos : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    // A robust epsilon to prevent division by zero.
    static const float SAFE_EPSILON = 1e-6f;
    
    // 1. Convert input color to Linear RGB.
    // The library functions handle the correct transfer curve (sRGB, PQ, HLG).
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

    // 2. Convert center pixel from Linear RGB to OkLab.
    // The library functions use the correct color primaries (BT.709 vs BT.2020).
    float3 oklab_center;
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    oklab_center = Csp::OkLab::Bt709To::OkLab(linear_rgb);
#else
    oklab_center = Csp::OkLab::Bt2020To::OkLab(linear_rgb);
#endif

    // 3. Apply the bilateral filter in OkLab's L space.
    float sigma_spatial_sq = pow(SigmaSpatial, 2.0);
    float sigma_range_sq = pow(SigmaRange, 2.0);

    // Use float-based Kahan summation for high-precision accumulation.
    float total_w = 0.0, comp_w = 0.0;
    float total_l = 0.0, comp_l = 0.0;

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

            float L_sample;
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)
            L_sample = Csp::OkLab::Bt709To::OkLab(linear_sample).x;
#else
            L_sample = Csp::OkLab::Bt2020To::OkLab(linear_sample).x;
#endif
            
            float weight_spatial = exp(-(x*x + y*y) / (2.0 * sigma_spatial_sq));
            float l_diff = L_sample - oklab_center.x;
            float weight_range = exp(-(l_diff*l_diff) / (2.0 * sigma_range_sq));
            float current_weight = weight_spatial * weight_range;
            
            // --- Kahan Summation ---
            float yw = current_weight - comp_w; float tw = total_w + yw;
            comp_w = (tw - total_w) - yw; total_w = tw;

            float yl = (L_sample * current_weight) - comp_l; float tl = total_l + yl;
            comp_l = (tl - total_l) - yl; total_l = tl;
        }
    }
    
    float filtered_L = total_l / max(total_w, SAFE_EPSILON);

    // 4. Calculate enhanced L value.
    float L_detail = oklab_center.x - filtered_L;
    float enhanced_L = oklab_center.x + L_detail * Strength;

    // 5. Reconstruct color.
    float3 final_oklab = float3(enhanced_L, oklab_center.y, oklab_center.z);
    float3 new_linear_rgb;
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    new_linear_rgb = Csp::OkLab::OkLabTo::Bt709(final_oklab);
#else
    new_linear_rgb = Csp::OkLab::OkLabTo::Bt2020(final_oklab);
#endif

    // 6. Convert final linear color to the target output color space.
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
technique BilateralContrast_Reference
<
    ui_label = "Bilateral Contrast (Reference)";
    ui_tooltip = "The definitive version.\nImproves contrast using a bilateral filter in the OkLab color space.\nThis shader is built upon the colour_space.fxh header.";
>
{
    pass
    {
        VertexShader = VS_PostProcess;
        PixelShader  = PS_BilateralContrast;
    }
}
