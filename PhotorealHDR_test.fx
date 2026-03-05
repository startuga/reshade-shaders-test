// =========================================================================
// Photoreal HDR Color Grader (V3 - WCG Gamut Preserving)
// Designed specifically to remove "game-y" dusty filters and yellow tints
// while 100% preserving HDR peak brightness AND Wide Color Gamut.
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
> = -0.12; 

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

// Dynamically assign accurate Luma coefficients based on the game's HDR output format
#if BUFFER_COLOR_SPACE == 3 // HDR10 (BT.2020 primaries)
    static const float3 LumaCoeff = float3(0.2627, 0.6780, 0.0593);
#else                       // scRGB (BT.709 primaries) or SDR
    static const float3 LumaCoeff = float3(0.2126, 0.7152, 0.0722);
#endif

float3 ApplyTemperature(float3 color, float temp, float tint) 
{
    // Adjusts balance linearly. Multipliers preserve negative WCG values.
    float3 balance = float3(1.0 + temp, 1.0, 1.0 - temp);
    balance *= float3(1.0 + tint, 1.0 - tint, 1.0 + tint);
    return color * balance;
}

float4 PS_PhotorealHDR(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target 
{
    // Sample the backbuffer. 
    // In scRGB HDR, colors > 1.0 are highlights, and colors < 0.0 are WCG (Wide Color Gamut).
    float3 color = tex2D(ReShade::BackBuffer, texcoord).rgb;

    // 1. Exposure (Multiplication safely scales negatives)
    color *= exp2(fExposure);

    // 2. Black Point (Dehaze) - Hue Preserving
    // By calculating a scalar multiplier based on luma, we preserve RGB ratios and negative WCG signs.
    float luma = dot(color, LumaCoeff);
    float newLuma = max(0.0, luma - fBlackPoint);
    color *= (luma > 1e-6) ? (newLuma / luma) : 0.0; 

    // 3. White Balance
    color = ApplyTemperature(color, fTemperature, fTint);

    // 4. Log Contrast (WCG SAFE)
    // We cannot use log2() on negative WCG values, and max(0.0) strips them entirely.
    // Solution: Extract the sign, apply the contrast curve to the absolute values, and restore the sign.
    const float pivot = 0.18; 
    
    float3 colorSign = sign(color);
    float3 absColor = abs(color);
    
    float3 logColor = log2(absColor + 1e-6);
    logColor = log2(pivot) + (logColor - log2(pivot)) * fContrast;
    
    // Apply exp2 to the magnitude, then multiply back the original sign to restore WCG
    color = colorSign * max(0.0, exp2(logColor) - 1e-6);

    // 5. Saturation
    // Recalculate luma. lerp() naturally expands WCG colors safely.
    luma = dot(color, LumaCoeff);
    color = lerp((float3)luma, color, fSaturation);

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
