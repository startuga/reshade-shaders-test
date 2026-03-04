// =========================================================================
// Photoreal HDR Color Grader
// Designed specifically to remove "game-y" dusty filters and yellow tints
// while 100% preserving High Dynamic Range (HDR) peak brightness.
// =========================================================================

#include "ReShade.fxh"

uniform float fBlackPoint <
    ui_type = "slider";
    ui_min = 0.000; ui_max = 0.050; ui_step = 0.001;
    ui_label = "Dehaze / Black Point";
    ui_tooltip = "Cuts through the 'dusty' lifted black levels of the game to restore real-world shadows.";
> = 0.005;

uniform float fTemperature <
    ui_type = "slider";
    ui_min = -0.50; ui_max = 0.50; ui_step = 0.01;
    ui_label = "Color Temperature";
    ui_tooltip = "Negative = Cooler (Removes yellow/sand tint)\nPositive = Warmer";
> = -0.12; // Default negative to cool down AC Mirage's heavy yellow tint

uniform float fTint <
    ui_type = "slider";
    ui_min = -0.50; ui_max = 0.50; ui_step = 0.01;
    ui_label = "Color Tint";
    ui_tooltip = "Negative = Greener\nPositive = More Magenta";
> = 0.02;

uniform float fExposure <
    ui_type = "slider";
    ui_min = -2.00; ui_max = 2.00; ui_step = 0.01;
    ui_label = "Exposure (EV)";
> = 0.00;

uniform float fContrast <
    ui_type = "slider";
    ui_min = 0.80; ui_max = 1.50; ui_step = 0.01;
    ui_label = "HDR Log Contrast";
    ui_tooltip = "Adds depth without destroying the extremely bright HDR highlights (sun/sky).";
> = 1.05;

uniform float fSaturation <
    ui_type = "slider";
    ui_min = 0.00; ui_max = 2.00; ui_step = 0.01;
    ui_label = "HDR Saturation";
> = 1.15;

// Rec.2020 Luma coefficients (Standard color space for HDR scRGB)
static const float3 LumaCoeff = float3(0.2627, 0.6780, 0.0593);

float3 ApplyTemperature(float3 color, float temp, float tint) 
{
    // Adjusts balance linearly (safe for HDR)
    float3 balance = float3(1.0 + temp, 1.0, 1.0 - temp);
    balance *= float3(1.0 + tint, 1.0 - tint, 1.0 + tint);
    return color * balance;
}

float4 PS_PhotorealHDR(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target 
{
    // Sample the backbuffer. In HDR mode, values can be well over 1.0.
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;

    // 1. Exposure
    color *= exp2(fExposure);

    // 2. Black Point (Dehaze)
    // Removes the gray/brown low-end haze without touching the midtones.
    color = max(0.0, color - fBlackPoint);

    // 3. White Balance
    color = ApplyTemperature(color, fTemperature, fTint);

    // 4. Log Contrast
    // Standard linear contrast ruins HDR. Log contrast curves the midtones 
    // around an 18% gray pivot while allowing peaks to remain unbounded.
    const float pivot = 0.18; 
    float3 logColor = log2(color + 1e-6);
    logColor = log2(pivot) + (logColor - log2(pivot)) * fContrast;
    color = max(0.0, exp2(logColor) - 1e-6);

    // 5. Saturation
    float luma = dot(color, LumaCoeff);
    color = lerp((float3)luma, color, fSaturation);

    // DO NOT USE saturate()! It clips HDR signals. Let the values go > 1.0.
    return float4(color, 1.0);
}

technique PhotorealHDR <
    ui_tooltip = "Photorealistic grading designed safely for HDR displays.\nRemoves the dusty filter and enhances natural colors.";
>
{
    pass 
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_PhotorealHDR;
    }
}