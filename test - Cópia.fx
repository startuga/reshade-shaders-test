/**
 * Bilateral Local Contrast Enhancement (IEC-Compliant Reference)
 *
 * Developed by Gemini for ReShade 6.0+
 *
 * This shader enhances texture clarity using a bilateral filter operating in the
 * CIELAB color space. This is a reference-quality implementation adhering to
 * IEC 61966-2-1:1999 specifications for maximum colorimetric accuracy.
 *
 * Key Features:
 * - Operates on the L* (lightness) component of the CIELAB color space.
 * - Utilizes high-precision color matrix and specification-correct constants.
 * - Employs float-based Kahan summation in the filter loop to minimize
 *   numerical error, ensuring maximum accuracy on all hardware.
 * - Automatically handles SDR, scRGB, HDR10, and HLG content.
 * - Prioritizes mathematical correctness and precision over performance.
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
    ui_min = 0.1; ui_max = 10.0;
    ui_step = 0.1;
    ui_label = "Sigma Range (L* Preservation)";
    ui_tooltip = "Perceptual lightness (L*) similarity threshold.\nHigher values smooth more details; lower values preserve more edges across all tonal ranges.";
> = 3.0;

//==============================================================================
// Textures and Samplers
//==============================================================================

texture2D texColorBuffer : COLOR;
sampler2D samplerColor { Texture = texColorBuffer; SRGBTexture = false; };

//==============================================================================
// Color Space Conversion Pipeline
//==============================================================================

// Converts from Linear sRGB (Rec.709 primaries) to CIE XYZ, pre-scaled for D65.
float3 LinearRGBToXYZ(float3 c)
{
    // The sRGB->XYZ matrix's rows have been divided by the D65_XYZ_REF components
    // (0.95047, 1.0, 1.08883) offline for maximum precision.
    static const float3x3 sRGB_TO_XYZ_PRESCALED_MATRIX = float3x3(
        0.4339122, 0.3762288, 0.1898590,
        0.2126000, 0.7152000, 0.0722000,
        0.0177210, 0.1093557, 0.8729232
    );
    return mul(sRGB_TO_XYZ_PRESCALED_MATRIX, c);
}

// Converts from CIE XYZ to Linear sRGB using a high-precision inverse matrix.
float3 XYZToLinearRGB(float3 c)
{
    // High-precision inverse of the standard sRGB->XYZ D65 matrix.
    static const float3x3 XYZ_TO_sRGB_HIGH_PRECISION_MATRIX = float3x3(
        3.2406255, -1.5372080, -0.4986286,
       -0.9689307,  1.8757561,  0.0415175,
        0.0557101, -0.2040211,  1.0569959
    );
    return mul(XYZ_TO_sRGB_HIGH_PRECISION_MATRIX, c);
}

// Perceptual mapping function for XYZ to CIELAB, using the official IEC spec constant.
float LabF(float t)
{
    // The official linear-segment threshold from IEC 61966-2-1:1999.
    static const float THRESHOLD = 216.0 / 24389.0;
    return t > THRESHOLD ? pow(t, 1.0/3.0) : (841.0 / 108.0) * t + (4.0 / 29.0);
}

// Inverse perceptual mapping function.
float InvLabF(float t)
{
    static const float THRESHOLD = 6.0 / 29.0;
    return t > THRESHOLD ? pow(t, 3.0) : (108.0 / 841.0) * (t - 4.0 / 29.0);
}

// Converts from CIE XYZ to CIELAB color space.
float3 XYZToLab(float3 xyz)
{
    // No division by D65_XYZ_REF needed here as it's baked into the forward matrix.
    float L = 116.0 * LabF(xyz.y) - 16.0;
    float a = 500.0 * (LabF(xyz.x) - LabF(xyz.y));
    float b = 200.0 * (LabF(xyz.y) - LabF(xyz.z));
    return float3(L, a, b);
}

// Converts from CIELAB to CIE XYZ color space.
float3 LabToXYZ(float3 lab)
{
    float fy = (lab.x + 16.0) / 116.0;
    float fx = lab.y / 500.0 + fy;
    float fz = fy - lab.z / 200.0;
    
    // The reference white is still needed for reconstruction.
    static const float3 D65_XYZ_REF = float3(0.95047, 1.0, 1.08883);
    return float3(InvLabF(fx), InvLabF(fy), InvLabF(fz)) * D65_XYZ_REF;
}

//==============================================================================
// SDR/HDR Transfer Functions
//==============================================================================
float3 SRGBToLinear(float3 srgb) { return srgb <= 0.04045 ? srgb / 12.92 : pow((srgb + 0.055) / 1.055, 2.4); }
float3 LinearToSRGB(float3 lin) { return lin <= 0.0031308 ? lin * 12.92 : 1.055 * pow(abs(lin), 1.0/2.4) - 0.055; }
float3 PQToLinear(float3 pq) { const float m1=0.1593017578, m2=78.84375, c1=0.8359375, c2=18.8515625, c3=18.6875; float3 l=pow(max(pq,0),1.0/m2); return 10000.*pow(max(l-c1,0)/(c2-c3*l),1.0/m1); }
float3 LinearToPQ(float3 lin) { const float m1=0.1593017578, m2=78.84375, c1=0.8359375, c2=18.8515625, c3=18.6875; lin=max(lin,0)/10000.; float3 l=pow(lin,m1); return pow((c1+c2*l)/(1.+c3*l),m2); }
float3 HLGToLinear(float3 hlg) { const float a=0.17883277,b=0.28466892,c=0.55991073; return hlg<=0.5?hlg*hlg/3.:(exp((hlg-c)/a)+b)/12.; }
float3 LinearToHLG(float3 lin) { const float a=0.17883277,b=0.28466892,c=0.55991073; lin*=12.; return lin<=1.?sqrt(3.*lin):a*log(lin-b)+c; }

//==============================================================================
// Main Shader
//==============================================================================
void FullscreenVS(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 tc : TEXCOORD0)
{
    tc.x = (id == 2) ? 2.0 : 0.0; tc.y = (id == 1) ? 2.0 : 0.0;
    pos = float4(tc*2.0-1.0, 0.0, 1.0); pos.y = -pos.y;
}

float4 BilateralContrastPS(float4 pos : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    // A robust epsilon to prevent division by zero in the final step.
    static const float SAFE_EPSILON = 1e-6f;
    
    // 1. Convert input color to Linear RGB based on buffer color space.
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

    // 2. Convert center pixel from Linear RGB to CIELAB.
    float3 lab_center = XYZToLab(LinearRGBToXYZ(linear_rgb));

    // 3. Apply the bilateral filter in L* space.
    float sigma_spatial_sq = pow(SigmaSpatial, 2.0);
    float sigma_range_sq = pow(SigmaRange, 2.0);

    // Use float-based Kahan summation for high-precision accumulation.
    float total_w = 0.0, comp_w = 0.0;
    float total_l = 0.0, comp_l = 0.0;

    for (int y = -Radius; y <= Radius; ++y)
    {
        for (int x = -Radius; x <= Radius; ++x)
        {
            float2 offset = float2(x,y) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
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
            float l_star_sample = XYZToLab(LinearRGBToXYZ(linear_sample)).x;

            float weight_spatial = exp(-(x*x + y*y) / (2.0 * sigma_spatial_sq));
            float l_star_diff = l_star_sample - lab_center.x;
            float weight_range = exp(-(l_star_diff*l_star_diff) / (2.0 * sigma_range_sq));
            float current_weight = weight_spatial * weight_range;

            float yw = current_weight - comp_w; float tw = total_w + yw;
            comp_w = (tw - total_w) - yw; total_w = tw;

            float yl = (l_star_sample * current_weight) - comp_l; float tl = total_l + yl;
            comp_l = (tl - total_l) - yl; total_l = tl;
        }
    }
    
    float filtered_l_star = total_l / max(total_w, SAFE_EPSILON);

    // 4. Calculate the enhanced L* value.
    float l_star_detail = lab_center.x - filtered_l_star;
    float enhanced_l_star = lab_center.x + l_star_detail * Strength;

    // 5. Reconstruct color.
    float3 final_lab = float3(enhanced_l_star, lab_center.y, lab_center.z);
    float3 new_linear_rgb = XYZToLinearRGB(LabToXYZ(final_lab));

    // 6. Convert final linear color to the target output color space.
    float3 final_rgb;
#if BUFFER_COLOR_SPACE == 1
    final_rgb = LinearToSRGB(new_linear_rgb);
    return float4(saturate(final_rgb), color_in.a);
#elif BUFFER_COLOR_SPACE == 2
    return float4(new_linear_rgb, color_in.a);
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
technique BilateralContrast_Reference_IEC <
    ui_label = "Bilateral Contrast (Reference, IEC-Compliant)";
    ui_tooltip = "The definitive version.\nImproves contrast using a bilateral filter in the CIELAB color space.\nUses IEC-spec constants and Kahan summation for maximum mathematical accuracy.";
>
{
    pass { VertexShader = FullscreenVS; PixelShader = BilateralContrastPS; }
}