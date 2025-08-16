/*
 * Entropy-based Bilateral Contrast Enhancement for ReShade 6+
 * 
 * Enhanced version with Kahan summation for improved numerical accuracy in summations.
 * Added Entropy-based Adaptive Bilateral Filtering (EABF) for maximum image quality.
 * The range sigma is adapted based on local entropy to better preserve details in textured areas.
 * Based on the method from "Entropy-based bilateral filtering with a new range kernel" by Tao Dai et al.
 * Local entropy is computed using a histogram in the neighborhood.
 * Adaptive sigma_range = K * SigmaRange, where K = k / (1 + exp(-alpha * (e - T)))
 * Higher entropy leads to smaller sigma_range for better detail preservation.
 * 
 * Parameters updated accordingly.
 * 
 * Author: Grok 4 (xAI)
 * Version: 1.1
 * Date: August 15, 2025
 */

#include "ReShade.fxh"

// Define color space detection
#define IS_SDR (BUFFER_COLOR_SPACE == 1)
#define IS_HDR10 (BUFFER_COLOR_SPACE == 3)
#define IS_SCRGB (BUFFER_COLOR_SPACE == 2)

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
    return float3(
        color.r <= 0.04045 ? color.r / 12.92 : pow((color.r + 0.055) / 1.055, 2.4),
        color.g <= 0.04045 ? color.g / 12.92 : pow((color.g + 0.055) / 1.055, 2.4),
        color.b <= 0.04045 ? color.b / 12.92 : pow((color.b + 0.055) / 1.055, 2.4)
    );
#elif IS_HDR10
    float3 p = pow(color, 1.0 / PQ_M2);
    float3 num = max(p - PQ_C1, 0.0);
    float3 den = PQ_C2 - (PQ_C3 * p);
    return pow(num / den, 1.0 / PQ_M1) * PQ_MAX_NITS;
#elif IS_SCRGB
    return color;
#else
    return color;
#endif
}

float3 to_display(float3 linear_color) {
#if IS_SDR
    return float3(
        linear_color.r <= 0.0031308 ? linear_color.r * 12.92 : 1.055 * pow(linear_color.r, 1.0 / 2.4) - 0.055,
        linear_color.g <= 0.0031308 ? linear_color.g * 12.92 : 1.055 * pow(linear_color.g, 1.0 / 2.4) - 0.055,
        linear_color.b <= 0.0031308 ? linear_color.b * 12.92 : 1.055 * pow(linear_color.b, 1.0 / 2.4) - 0.055
    );
#elif IS_HDR10
    float3 y = linear_color / PQ_MAX_NITS;
    float3 num = PQ_C1 + PQ_C2 * pow(y, PQ_M1);
    float3 den = 1.0 + PQ_C3 * pow(y, PQ_M1);
    return pow(num / den, PQ_M2);
#elif IS_SCRGB
    return linear_color;
#else
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
    ui_tooltip = "Base range sigma for relative intensity difference (edge preservation).";
> = 0.1;

uniform float Alpha <
    ui_type = "slider";
    ui_min = -5.0;
    ui_max = 0.0;
    ui_step = 0.1;
    ui_tooltip = "Alpha for EABF sigmoid (negative for decreasing with entropy).";
> = -1.0;

uniform float KParam <
    ui_type = "slider";
    ui_min = 1.0;
    ui_max = 5.0;
    ui_step = 0.1;
    ui_tooltip = "k for EABF amplitude.";
> = 2.5;

uniform float EntropyThreshold <
    ui_type = "slider";
    ui_min = 0.0;
    ui_max = 5.0;
    ui_step = 0.1;
    ui_tooltip = "T for EABF threshold (approx 0.7 * max local entropy).";
> = 2.0;

// Define number of bins for entropy histogram
#define ENTROPY_BINS 16

float3 PS_BilateralContrast(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target {
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;
    float3 linear_color = to_linear(color);
    float luma = get_luma(linear_color);

    const float epsilon = 0.001;
    const float2 pixel_size = BUFFER_PIXEL_SIZE;
    const int window_size = 2 * Radius + 1;
    const float total_pixels = float(window_size * window_size);

    // First pass: Find min and max luma in neighborhood for normalization
    float min_luma = luma;
    float max_luma = luma;
    for (int y = -Radius; y <= Radius; y++) {
        for (int x = -Radius; x <= Radius; x++) {
            float2 offset = float2(x, y) * pixel_size;
            float3 neigh_color = tex2D(ReShade::BackBuffer, texcoord + offset).rgb;
            float3 neigh_linear = to_linear(neigh_color);
            float neigh_luma = get_luma(neigh_linear);
            min_luma = min(min_luma, neigh_luma);
            max_luma = max(max_luma, neigh_luma);
        }
    }

    // Second pass: Compute local entropy using histogram
    float hist[ENTROPY_BINS];
    for (int i = 0; i < ENTROPY_BINS; i++) hist[i] = 0.0;
    for (int y = -Radius; y <= Radius; y++) {
        for (int x = -Radius; x <= Radius; x++) {
            float2 offset = float2(x, y) * pixel_size;
            float3 neigh_color = tex2D(ReShade::BackBuffer, texcoord + offset).rgb;
            float3 neigh_linear = to_linear(neigh_color);
            float neigh_luma = get_luma(neigh_linear);
            float norm_luma = (neigh_luma - min_luma) / (max_luma - min_luma + epsilon);
            int bin = int(floor(norm_luma * (ENTROPY_BINS - 1)));
            hist[bin] += 1.0;
        }
    }
    float entropy = 0.0;
    for (int i = 0; i < ENTROPY_BINS; i++) {
        float p = hist[i] / total_pixels;
        if (p > 0.0) entropy -= p * log(p);
    }

    // Compute adaptive SigmaRange
    float exp_term = exp(-Alpha * (entropy - EntropyThreshold));
    float K = KParam / (1.0 + exp_term);
    float local_sigma_range = K * SigmaRange;

    // Third pass: Bilateral filter with Kahan summation
    float sum_luma = 0.0;
    float c_luma = 0.0;
    float sum_weight = 0.0;
    float c_weight = 0.0;

    for (int y = -Radius; y <= Radius; y++) {
        for (int x = -Radius; x <= Radius; x++) {
            float2 offset = float2(x, y) * pixel_size;
            float3 neigh_color = tex2D(ReShade::BackBuffer, texcoord + offset).rgb;
            float3 neigh_linear = to_linear(neigh_color);
            float neigh_luma = get_luma(neigh_linear);

            // Spatial weight (Gaussian)
            float dist_spatial = length(float2(x, y));
            float w_spatial = exp(- (dist_spatial * dist_spatial) / (2.0 * SigmaSpatial * SigmaSpatial));

            // Range weight (relative)
            float avg_luma = (luma + neigh_luma) * 0.5 + epsilon;
            float dist_range = abs(luma - neigh_luma) / avg_luma;
            float w_range = exp(- (dist_range * dist_range) / (2.0 * local_sigma_range * local_sigma_range));

            float weight = w_spatial * w_range;

            // Kahan summation for sum_luma += neigh_luma * weight
            float input_luma = neigh_luma * weight;
            float y_luma = input_luma - c_luma;
            float t_luma = sum_luma + y_luma;
            c_luma = (t_luma - sum_luma) - y_luma;
            sum_luma = t_luma;

            // Kahan summation for sum_weight += weight
            float y_weight = weight - c_weight;
            float t_weight = sum_weight + y_weight;
            c_weight = (t_weight - sum_weight) - y_weight;
            sum_weight = t_weight;
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
