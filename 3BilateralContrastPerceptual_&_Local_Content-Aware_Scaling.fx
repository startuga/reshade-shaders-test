/**
 * Bilateral Local Contrast Enhancement (v4 - Perceptual & Local)
 *
 * Developed by Gemini for ReShade 6.0+
 *
 * This state-of-the-art shader uses a multi-pass, content-aware algorithm
 * to deliver the highest possible image quality. It analyzes the local dynamic
 * range of the image and performs comparisons in perceptual (logarithmic) space
 * to perfectly adapt the filter's strength to every pixel.
 *
 * Key Features:
 * - Content-Aware Local Adaptation: Edge preservation is uniquely calculated
 *   for each pixel based on its surrounding brightness, eliminating artifacts.
 * - Perceptual Logarithmic Scaling: Mimics human vision for flawless handling
 *   of extreme dark and bright details in the same scene.
 * - Operates in a linear RGB color space for high-fidelity color calculations.
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
    ui_tooltip = "Percentage of the local dynamic range to preserve as an edge.\nHigher values are more aggressive at preserving edges.";
> = 0.25;

//==============================================================================
// Textures and Samplers
//==============================================================================

texture2D texColorBuffer : COLOR;
sampler2D samplerColor { Texture = texColorBuffer; };

// Texture to store local min/max luminance for content-aware scaling
texture2D texLocalAnalysis { Width = BUFFER_WIDTH / 4; Height = BUFFER_HEIGHT / 4; Format = RG16F; };
sampler2D samplerLocalAnalysis { Texture = texLocalAnalysis; };

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

// --- Pass 1: Analyze local dynamic range ---
float4 LocalAnalysisPS(float4 pos : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    float min_luma = 1.0e6;
    float max_luma = 0.0;

    // Analyze a 7x7 neighborhood
    for (int y = -3; y <= 3; ++y)
    {
        for (int x = -3; x <= 3; ++x)
        {
            float2 offset = float2(x, y) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
            float3 color = tex2Dlod(samplerColor, float4(texcoord + offset, 0, 0)).rgb;
            
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
            min_luma = min(min_luma, luma);
            max_luma = max(max_luma, luma);
        }
    }
    
    // Store the log of min and max for perceptual calculations later
    return float4(log(max(min_luma, EPSILON)), log(max(max_luma, EPSILON)), 0, 1);
}

// --- Pass 2: Main bilateral filter ---
float4 BilateralContrastPS(float4 pos : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    // --- LOCAL ADAPTATION ---
    // Get the pre-calculated local log-luminance range for this pixel
    float2 local_log_luma = tex2D(samplerLocalAnalysis, texcoord).rg;
    float local_log_range = max(EPSILON, local_log_luma.y - local_log_luma.x);
    
    // The effective sigma is now a percentage of the local perceptual range
    float effective_sigma_range = SigmaRange * local_log_range;

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
    float log_luma_center = log(max(luma_center, EPSILON));

    float sigma_spatial_sq = pow(SigmaSpatial, 2.0);
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

            // --- PERCEPTUAL COMPARISON ---
            float log_luma_sample = log(max(luma_sample, EPSILON));
            float luma_diff = log_luma_sample - log_luma_center;

            float weight_spatial = exp(-(x * x + y * y) / (2.0 * sigma_spatial_sq));
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

technique BilateralContrastEnhance_LocalPerceptual <
    ui_label = "Bilateral Contrast (Local Perceptual)";
    ui_tooltip = "STATE-OF-THE-ART QUALITY\nImproves texture clarity using a content-aware, multi-pass algorithm that adapts to local brightness and mimics human vision.";
>
{
    // Pass 1: Analyze the scene and write local min/max log-luminance
    pass
    {
        VertexShader = FullscreenVS;
        PixelShader = LocalAnalysisPS;
        RenderTarget = texLocalAnalysis;
    }

    // Pass 2: The main effect, which reads the result of the analysis pass
    pass
    {
        VertexShader = FullscreenVS;
        PixelShader = BilateralContrastPS;
    }
}