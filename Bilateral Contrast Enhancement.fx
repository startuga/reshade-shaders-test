/*
 * Bilateral Contrast Enhancement for ReShade 6+
 * 
 * This shader applies a bilateral filter to enhance micro-contrast and local contrast on the luminance channel,
 * preserving edges and original colors (hue and saturation). Computations are performed in linear RGB space for accuracy.
 * Compatible with SDR (sRGB), HDR10 (PQ Rec.2020), and scRGB (linear Rec.709 extended).
 * 
 * Highest image quality prioritized, with no strict performance constraints.
 * 
 * Parameters:
 * - Strength: Amount of contrast enhancement.
 * - Radius: Kernel radius for bilateral filter (larger = smoother blur, affects larger scales).
 * - SigmaSpatial: Spatial sigma for Gaussian weight.
 * - SigmaRange: Range sigma for intensity difference (relative, for edge preservation).
 * 
 * Author: Grok 4 (xAI)
 * Version: 1.0
 * Date: August 15, 2025
 */

#include "ReShade.fxh"

// Define color space detection
#define IS_SDR (BUFFER_COLOR_BIT_DEPTH == 8)
#define IS_HDR10 (BUFFER_COLOR_BIT_DEPTH == 10)
#define IS_SCRGB (BUFFER_COLOR_BIT_DEPTH == 16)

// Luminance coefficients
#if IS_HDR10
    static const float3 LUMA_COEFFS = float3(0.2627, 0.6780, 0.0593); // Rec.2020 for HDR10
#else
    static const float3 LUMA_COEFFS = float3(0.2126, 0.7152, 0.0722); // Rec.709 for SDR and scRGB
#endif

// PQ constants for HDR10
static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;
static const float PQ_MAX_NITS = 10000.0;

// Color space conversion functions
float3 to_linear(float3 color) {
#if IS_SDR
    // Exact sRGB to linear
    return float3(
        color.r <= 0.04045 ? color.r / 12.92 : pow((color.r + 0.055) / 1.055, 2.4),
        color.g <= 0.04045 ? color.g / 12.92 : pow((color.g + 0.055) / 1.055, 2.4),
        color.b <= 0.04045 ? color.b / 12.92 : pow((color.b + 0.055) / 1.055, 2.4)
    );
#elif IS_HDR10
    // PQ to linear (scene-referred, in cd/m^2)
    float3 p = pow(color, 1.0 / PQ_M2);
    float3 num = max(p - PQ_C1, 0.0);
    float3 den = PQ_C2 - (PQ_C3 * p);
    return pow(num / den, 1.0 / PQ_M1) * PQ_MAX_NITS;
#elif IS_SCRGB
    // scRGB is already linear
    return color;
#else
    // Fallback to identity
    return color;
#endif
}

float3 to_display(float3 linear_color) {
#if IS_SDR
    // Linear to sRGB
    return float3(
        linear_color.r <= 0.0031308 ? linear_color.r * 12.92 : 1.055 * pow(linear_color.r, 1.0 / 2.4) - 0.055,
        linear_color.g <= 0.0031308 ? linear_color.g * 12.92 : 1.055 * pow(linear_color.g, 1.0 / 2.4) - 0.055,
        linear_color.b <= 0.0031308 ? linear_color.b * 12.92 : 1.055 * pow(linear_color.b, 1.0 / 2.4) - 0.055
    );
#elif IS_HDR10
    // Linear to PQ
    float3 y = linear_color / PQ_MAX_NITS;
    float3 num = PQ_C1 + PQ_C2 * pow(y, PQ_M1);
    float3 den = 1.0 + PQ_C3 * pow(y, PQ_M1);
    return pow(num / den, PQ_M2);
#elif IS_SCRGB
    // scRGB is linear
    return linear_color;
#else
    // Fallback to identity
    return linear_color;
#endif
}

float get_luma(float3 linear_color) {
    return dot(linear_color, LUMA_COEFFS);
}

// Uniforms
uniform float Strength <
    ui_type = "slider";
    ui_min = 0.0;
    ui_max = 2.0;
    ui_step = 0.01;
    ui_tooltip = "Contrast enhancement strength.";
> = 0.5;

uniform int Radius <
    ui_type = "slider";
    ui_min = 1;
    ui_max = 5;
    ui_step = 1;
    ui_tooltip = "Bilateral filter radius (higher affects larger scales).";
> = 3;

uniform float SigmaSpatial <
    ui_type = "slider";
    ui_min = 0.1;
    ui_max = 5.0;
    ui_step = 0.1;
    ui_tooltip = "Spatial sigma for Gaussian weighting.";
> = 1.5;

uniform float SigmaRange <
    ui_type = "slider";
    ui_min = 0.01;
    ui_max = 0.5;
    ui_step = 0.01;
    ui_tooltip = "Range sigma for relative intensity difference (edge preservation).";
> = 0.1;

float3 PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target {
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float3 linear_color = to_linear(color);
    float luma = get_luma(linear_color);

    float sum_luma = 0.0;
    float sum_weight = 0.0;
    const float epsilon = 0.001;
    const float2 pixel_size = BUFFER_PIXEL_SIZE;

    for (int y = -Radius; y <= Radius; y++) {
        for (int x = -Radius; x <= Radius; x++) {
            float2 offset = float2(x, y) * pixel_size;
            float3 neigh_color = tex2D(ReShade::BackBuffer, texcoord + offset).rgb;
            float3 neigh_linear = to_linear(neigh_color);
            float neigh_luma = get_luma(neigh_linear);

            // Spatial weight (Gaussian)
            float dist_spatial = length(float2(x, y));
            float w_spatial = exp(- (dist_spatial * dist_spatial) / (2.0 * SigmaSpatial * SigmaSpatial));

            // Range weight (relative for HDR compatibility)
            float avg_luma = (luma + neigh_luma) * 0.5 + epsilon;
            float dist_range = abs(luma - neigh_luma) / avg_luma;
            float w_range = exp(- (dist_range * dist_range) / (2.0 * SigmaRange * SigmaRange));

            float weight = w_spatial * w_range;
            sum_luma += neigh_luma * weight;
            sum_weight += weight;
        }
    }

    float blurred_luma = sum_luma / sum_weight;
    float enhanced_luma = luma + Strength * (luma - blurred_luma);

    // Apply ratio to preserve hue and saturation
    float ratio = (luma > epsilon) ? enhanced_luma / luma : 1.0;
    float3 enhanced_linear = linear_color * ratio;

    float3 output_color = to_display(enhanced_linear);
    return output_color;
}

technique BilateralContrast {
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_BilateralContrast;
    }
}