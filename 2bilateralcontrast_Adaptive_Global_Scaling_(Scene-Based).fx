/**
 * Bilateral Local Contrast Enhancement (v4 - Fully Configurable)
 *
 * Developed by Gemini for ReShade 6.0+
 *
 * This shader enhances texture clarity and local contrast using a bilateral
 * filter. This version introduces adaptive scaling for the Sigma Range,
 * which analyzes the average scene brightness to provide a more accurate and
 * consistent effect across different lighting conditions.
 *
 * Key Features:
 * - Adaptive Sigma Range: Dynamically scales edge preservation based on scene
 *   luminance for superior results in both dark and bright areas.
 * - Operates in a linear RGB color space for high-fidelity color calculations.
 * - Preserves original hue and saturation by primarily adjusting luminance.
 * - Automatically handles SDR (sRGB), scRGB, HDR10 (PQ), and HLG color spaces.
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
    ui_label = "Sigma Range (Edge Preservation)";
    ui_tooltip = "Luminance similarity threshold.\nThis is a normalized value that is automatically scaled for HDR content to ensure consistent behavior.";
> = 0.15;

uniform float AdaptiveSensitivity <
    ui_type = "slider";
    ui_min = 0.1; ui_max = 2.0;
    ui_step = 0.05;
    ui_label = "Adaptive Sensitivity";
    ui_tooltip = "Controls how aggressively the Sigma Range adapts to scene brightness.\nHigher values cause stronger adaptation in bright scenes.";
> = 0.5;

//==============================================================================
// Textures and Samplers
//==============================================================================

texture2D texColorBuffer : COLOR;
sampler2D samplerColor { Texture = texColorBuffer; };

// Texture to store luminance data for adaptive scaling
texture2D texLuma { Width = 64; Height = 64; Format = R16F; MipLevels = 0; };
sampler2D samplerLuma { Texture = texLuma; };

//==============================================================================
// Helper Functions
//==============================================================================

// --- Constants ---
static const float3 LUMINANCE_VECTOR = float3(0.2126, 0.7152, 0.0722);
static const float EPSILON = 1e-6;

// --- Color Space Conversions (unchanged) ---
float3 SRGBToLinear(float3 srgb) { return srgb <= 0.04045 ? srgb / 12.92 : pow((srgb + 0.055) / 1.055, 2.4); }
float3 LinearToSRGB(float3 linear_in) { return linear_in <= 0.0031308 ? linear_in * 12.92 : 1.055 * pow(abs(linear_in), 1.0 / 2.4) - 0.055; }
static const float PQ_M1 = 0.1593017578125, PQ_M2 = 78.84375, PQ_C1 = 0.8359375, PQ_C2 = 18.8515625, PQ_C3 = 18.6875, PEAK_LUMINANCE = 10000.0;
float3 PQToLinear(float3 pq) { float3 l = pow(max(pq, 0.0), 1.0 / PQ_M2); float3 n = max(l - PQ_C1, 0.0); float3 d = PQ_C2 - PQ_C3 * l; return PEAK_LUMINANCE * pow(n / d, 1.0 / PQ_M1); }
float3 LinearToPQ(float3 linear_in) { linear_in = max(linear_in, 0.0) / PEAK_LUMINANCE; float3 l = pow(linear_in, PQ_M1); float3 n = PQ_C2 * l + PQ_C1; float3 d = PQ_C3 * l + 1.0; return pow(n / d, PQ_M2); }
static const float HLG_A = 0.17883277, HLG_B = 0.28466892, HLG_C = 0.55991073;
float3 HLGToLinear(float3 hlg) { const float3 scale = 1.0 / 12.0; return hlg <= 0.5 ? hlg * hlg / 3.0 : (exp((hlg - HLG_C) / HLG_A) + HLG_B) * scale; }
float3 LinearToHLG(float3 linear_in) { linear_in *= 12.0; return linear_in <= 1.0 ? sqrt(3.0 * linear_in) : HLG_A * log(linear_in - HLG_B) + HLG_C; }

//==============================================================================
// Shader Entry Points
//==============================================================================

void FullscreenVS(uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD0)
{
    texcoord.x = (id == 2) ? 2.0 : 0.0;
    texcoord.y = (id == 1) ? 2.0 : 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// --- Pass 1: Analyze scene luminance ---
float4 LuminanceAnalysisPS(float4 pos : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    float3 color = tex2D(samplerColor, texcoord).rgb;
    float luma;

#if BUFFER_COLOR_SPACE == 1
    luma = dot(SRGBToLinear(color), LUMINANCE_VECTOR);
#elif BUFFER_COLOR_SPACE == 2
    luma = dot(color, LUMINANCE_VECTOR);
#elif BUFFER_COLOR_SPACE == 3
    luma = dot(PQToLinear(color), LUMINANCE_VECTOR);
#elif BUFFER_COLOR_SPACE == 4
    luma = dot(HLGToLinear(color), LUMINANCE_VECTOR);
#else
    luma = dot(SRGBToLinear(color), LUMINANCE_VECTOR);
#endif
    
    // Use log-average for better perceptual results
    return float4(log(max(luma, EPSILON)), 0, 0, 1);
}

// --- Pass 2: Main bilateral filter ---
float4 BilateralContrastPS(float4 pos : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    // Get average scene log-luminance from the 1x1 mipmap of our analysis texture
    float avg_log_luma = tex2Dlod(samplerLuma, float4(0.5, 0.5, 0, 100)).r;
    float avg_luma = exp(avg_log_luma);

    // --- ADAPTIVE SCALING ---
    // Scale the range based on average scene brightness using the UI sensitivity control.
    float range_scale = max(1.0, avg_luma * AdaptiveSensitivity);

    float4 color_in = tex2D(samplerColor, texcoord);
    float3 linear_rgb;
#if BUFFER_COLOR_SPACE == 1
    linear_rgb = SRGBToLinear(color_in.rgb);
#elif BUFFER_COLOR_SPACE == 2
    linear_rgb = color_in.rgb;
#elif BUFFER_COLOR_SPACE == 3
    linear_rgb = PQToLinear(color_in.rgb);
#elif BUFFER_COLOR_SPACE == 4
    linear_rgb = HLGToLinear(color_in.rgb);
#else
    linear_rgb = SRGBToLinear(color_in.rgb);
#endif

    float luma_center = dot(linear_rgb, LUMINANCE_VECTOR);
    float sigma_spatial_sq = pow(SigmaSpatial, 2.0);
    float effective_sigma_range = SigmaRange * range_scale;
    float sigma_range_sq = pow(effective_sigma_range, 2.0);

    float total_weight = 0.0;
    float filtered_luma = 0.0;

    for (int y = -Radius; y <= Radius; ++y)
    {
        for (int x = -Radius; x <= Radius; ++x)
        {
            float2 offset = float2(x, y) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
            float4 color_sample = tex2Dlod(samplerColor, float4(texcoord + offset, 0, 0));

            float3 linear_sample;
#if BUFFER_COLOR_SPACE == 1
            linear_sample = SRGBToLinear(color_sample.rgb);
#elif BUFFER_COLOR_SPACE == 2
            linear_sample = color_sample.rgb;
#elif BUFFER_COLOR_SPACE == 3
            linear_sample = PQToLinear(color_sample.rgb);
#elif BUFFER_COLOR_SPACE == 4
            linear_sample = HLGToLinear(color_sample.rgb);
#else
            linear_sample = SRGBToLinear(color_sample.rgb);
#endif
            float luma_sample = dot(linear_sample, LUMINANCE_VECTOR);

            float weight_spatial = exp(-(x * x + y * y) / (2.0 * sigma_spatial_sq));
            float luma_diff = luma_sample - luma_center;
            float weight_range = exp(-(luma_diff * luma_diff) / (2.0 * sigma_range_sq));
            float weight = weight_spatial * weight_range;

            filtered_luma += luma_sample * weight;
            total_weight += weight;
        }
    }

    filtered_luma /= total_weight;

    float detail = luma_center - filtered_luma;
    float enhanced_luma = luma_center + detail * Strength;
    enhanced_luma = max(EPSILON, enhanced_luma);

    float luma_ratio = enhanced_luma / max(luma_center, EPSILON);
    float3 new_linear_rgb = linear_rgb * luma_ratio;

    float3 final_rgb;
#if BUFFER_COLOR_SPACE == 1
    final_rgb = LinearToSRGB(new_linear_rgb);
    return float4(saturate(final_rgb), color_in.a);
#elif BUFFER_COLOR_SPACE == 2
    final_rgb = new_linear_rgb;
    return float4(final_rgb, color_in.a);
#elif BUFFER_COLOR_SPACE == 3
    final_rgb = LinearToPQ(new_linear_rgb);
    return float4(final_rgb, color_in.a);
#elif BUFFER_COLOR_SPACE == 4
    final_rgb = LinearToHLG(new_linear_rgb);
    return float4(final_rgb, color_in.a);
#else
    final_rgb = LinearToSRGB(new_linear_rgb);
    return float4(saturate(final_rgb), color_in.a);
#endif
}

//==============================================================================
// Technique Definition
//==============================================================================

technique BilateralContrastEnhance <
    ui_label = "Bilateral Local Contrast Enhancement";
    ui_tooltip = "Improves texture clarity and local contrast using a bilateral filter.\nThis effect operates in linear light and is compatible with SDR, scRGB, and HDR formats.\nIt enhances details by primarily adjusting luminance, preserving color fidelity.";
>
{
    // Pass 1: Analyze the scene and write log-luminance to texLuma
    pass LuminancePass
    {
        VertexShader = FullscreenVS;
        PixelShader = LuminanceAnalysisPS;
        RenderTarget = texLuma;
    }

    // Pass 2: The main effect, which reads the result of the analysis pass
    pass FilterPass
    {
        VertexShader = FullscreenVS;
        PixelShader = BilateralContrastPS;
    }
}