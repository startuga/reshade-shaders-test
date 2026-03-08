// ============================================================================
// Calibration Monitor for Photoreal HDR V5.5 (Mastering Edition)
//
// Professional diagnostic overlay and test-pattern generator.
// Place this technique AFTER Bilateral Contrast and Photoreal HDR
// in the ReShade technique list.
//
// Features:
//   - 11 diagnostic visualization modes
//   - Cinema-standard False Color zone palette
//   - Oklab chroma / skin-tone analysis
//   - Test-pattern generation (Zone Wedge, Color Bars)
//   - Spot Meter with zone-scale readout
//   - Color-space-aware (sRGB / scRGB / HDR10 PQ)
//
// Version: 1.0.2 (Mastering Edition)
// - Fix: Neutral Grey Detector now uses tint-direction diagnostic output
// - Fix: Chroma Heat Map and Neutral Detector use dark-chroma reliability fade
// - Fix: Spot Meter keeps the center decode hoisted out of the overlay function
// - Fix: Spot Meter marker now tracks true zone widths smoothly
// - Fix: Gradient + Color Bars avoids per-pixel array initialization
// ============================================================================

#include "ReShade.fxh"

// ==============================================================================
// 1. Constants (matching companion shaders)
// ==============================================================================

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

static const float FLT_MIN     = 1.175494351e-38;
static const float SCRGB_WHITE = 80.0;
static const float PI          = 3.14159265359;

// sRGB thresholds
static const float SRGB_THRESHOLD_EOTF = 0.04045;
static const float SRGB_THRESHOLD_OETF = (0.04045 / 12.92);

// ST 2084 PQ
static const float PQ_M1   = 0.1593017578125;
static const float PQ_M2   = 78.84375;
static const float PQ_C1   = 0.8359375;
static const float PQ_C2   = 18.8515625;
static const float PQ_C3   = 18.6875;
static const float PQ_PEAK = 10000.0;

// Chroma reliability alignment with companion shaders
static const float CHROMA_STABILITY_THRESH  = 1e-4;
static const float CHROMA_RELIABILITY_START = 5e-5;
static const float INV_CHROMA_RELIABILITY_SPAN =
    1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);

// ITU-R luma
static const float3 Luma709  = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020 = float3(0.2627, 0.6780, 0.0593);

// Oklab M1 (Rec.709)
static const float3x3 RGB709_to_LMS = float3x3(
    0.4122214708, 0.5363325363, 0.0514459929,
    0.2119034982, 0.6806995451, 0.1073969566,
    0.0883024619, 0.2817188376, 0.6299787005
);

// Oklab M1 (Rec.2020 row-sum-normalized)
static const float3x3 RGB2020_to_LMS = float3x3(
    0.6167596970, 0.3601880240, 0.0230522790,
    0.2651316740, 0.6358515800, 0.0990167460,
    0.1001279150, 0.2038783840, 0.6959937010
);

static const float3x3 LMS_to_Oklab = float3x3(
    0.2104542553, 0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050, 0.4505937099,
    0.0259040371, 0.7827717662, -0.8086757660
);

// Zone boundaries (powers of 2, Zone V = 18% grey = 2^-2.5)
static const float ZONE_I    = 0.04419417382;
static const float ZONE_II   = 0.06250000000;
static const float ZONE_III  = 0.08838834764;
static const float ZONE_IV   = 0.12500000000;
static const float ZONE_V    = 0.17677669529;
static const float ZONE_VI   = 0.25000000000;
static const float ZONE_VII  = 0.35355339059;
static const float ZONE_VIII = 0.50000000000;
static const float ZONE_IX   = 0.70710678118;
static const float ZONE_X    = 1.00000000000;
static const float ZONE_XI   = 2.00000000000;

// Smooth spot-meter marker calibration
static const float ZoneSegLower[12] = {
    0.0,
    ZONE_I, ZONE_II, ZONE_III, ZONE_IV, ZONE_V,
    ZONE_VI, ZONE_VII, ZONE_VIII, ZONE_IX, ZONE_X, ZONE_XI
};

static const float ZoneSegUpper[12] = {
    ZONE_I, ZONE_II, ZONE_III, ZONE_IV, ZONE_V, ZONE_VI,
    ZONE_VII, ZONE_VIII, ZONE_IX, ZONE_X, ZONE_XI, 4.0
};

// ==============================================================================
// 2. Texture & Sampler
// ==============================================================================

texture2D TextureBackBuffer : COLOR;
sampler2D SamplerBackBuffer
{
    Texture   = TextureBackBuffer;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = NONE;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

// ==============================================================================
// 3. UI
// ==============================================================================

uniform int iMode <
    ui_type  = "combo";
    ui_label = "Display Mode";
    ui_items = "Off (Passthrough)\0"
               "False Color (Zones)\0"
               "Clipping Warning\0"
               "Luminance Only\0"
               "Chroma Heat Map\0"
               "Neutral Grey Detector\0"
               "Skin Tone Indicator\0"
               "Black Point Preview\0"
               "Gamut Warning\0"
               "Zebra Stripes\0"
               "Test: Zone Step Wedge\0"
               "Test: Gradient + Color Bars\0";
    ui_tooltip = "Diagnostic visualisation mode.\n"
                 "Overlay modes blend with the image.\n"
                 "Test-pattern modes replace the image entirely.";
    ui_category = "Monitor";
> = 0;

uniform float fOverlayOpacity <
    ui_type  = "slider";
    ui_label = "Overlay Opacity";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.01;
    ui_tooltip = "Blend strength for overlay modes (Clipping, Neutral, Skin, BP, Gamut, Zebra).";
    ui_category = "Monitor";
> = 0.75;

uniform float fZebraThreshold <
    ui_type  = "slider";
    ui_label = "Zebra IRE Threshold";
    ui_min = 0.5; ui_max = 1.05; ui_step = 0.01;
    ui_tooltip = "Normalised luminance above which zebra stripes appear.\n"
                 "1.0 = reference white. 0.9 = 90%% IRE.";
    ui_category = "Monitor";
> = 0.95;

uniform float fBPPreviewLevel <
    ui_type  = "slider";
    ui_label = "Black-Point Preview (%% of White)";
    ui_min = 0.000; ui_max = 0.050; ui_step = 0.001;
    ui_tooltip = "Highlights pixels at or below this percentage of white.\n"
                 "Match this to Photoreal HDR's Dehaze slider to preview the cut.";
    ui_category = "Monitor";
> = 0.005;

uniform bool bSpotMeter <
    ui_label = "Show Spot Meter";
    ui_tooltip = "Draws a centre-screen zone indicator and scale bar.";
    ui_category = "Spot Meter";
> = true;

uniform int iSpotSize <
    ui_type = "slider";
    ui_label = "Spot Meter Radius (px)";
    ui_min = 8; ui_max = 40;
    ui_category = "Spot Meter";
> = 16;

uniform int iColorSpaceOverride <
    ui_type  = "combo";
    ui_label = "Color Space Override";
    ui_items = "Auto (Default)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0";
    ui_tooltip = "Must match Bilateral Contrast / Photoreal HDR.";
    ui_category = "System";
> = 0;

uniform float fWhitePoint <
    ui_type  = "slider";
    ui_label = "Reference White (Nits)";
    ui_min = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_tooltip = "Must match the companion shaders.";
    ui_category = "System";
> = 203.0;

// ==============================================================================
// 4. Math
// ==============================================================================

float PowNonNeg(float x, float e)
{
    if (x <= 0.0) return 0.0;
    return pow(x, e);
}

float3 PowNonNeg3(float3 x, float e)
{
    return float3(PowNonNeg(x.r, e), PowNonNeg(x.g, e), PowNonNeg(x.b, e));
}

float GetChromaReliability(float luma)
{
    float t = saturate((luma - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    return t * t * (3.0 - 2.0 * t);
}

// ==============================================================================
// 5. EOTF / OETF (matching companions exactly)
// ==============================================================================

float3 sRGB_EOTF(float3 V)
{
    float3 a  = abs(V);
    float3 lo = a / 12.92;
    float3 hi = PowNonNeg3((a + 0.055) / 1.055, 2.4);

    float3 o;
    o.r = (a.r <= SRGB_THRESHOLD_EOTF) ? lo.r : hi.r;
    o.g = (a.g <= SRGB_THRESHOLD_EOTF) ? lo.g : hi.g;
    o.b = (a.b <= SRGB_THRESHOLD_EOTF) ? lo.b : hi.b;
    return sign(V) * o;
}

float3 sRGB_OETF(float3 L)
{
    float3 a  = abs(L);
    float3 lo = a * 12.92;
    float3 hi = 1.055 * PowNonNeg3(a, 1.0 / 2.4) - 0.055;

    float3 o;
    o.r = (a.r <= SRGB_THRESHOLD_OETF) ? lo.r : hi.r;
    o.g = (a.g <= SRGB_THRESHOLD_OETF) ? lo.g : hi.g;
    o.b = (a.b <= SRGB_THRESHOLD_OETF) ? lo.b : hi.b;
    return sign(L) * o;
}

float3 PQ_EOTF(float3 N)
{
    float3 Np  = PowNonNeg3(max(N, 0.0), 1.0 / PQ_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    return PowNonNeg3(num / den, 1.0 / PQ_M1) * PQ_PEAK;
}

float3 PQ_InvEOTF(float3 L)
{
    float3 Lp  = PowNonNeg3(max(L, 0.0) / PQ_PEAK, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    return PowNonNeg3(num / den, PQ_M2);
}

float3 Decode(float3 enc, int sp)
{
    [branch] if (sp == 3) return PQ_EOTF(enc);
    [branch] if (sp == 2) return enc * SCRGB_WHITE;
    return sRGB_EOTF(enc) * SCRGB_WHITE;
}

float3 Encode(float3 lin, int sp)
{
    [branch] if (sp == 3) return PQ_InvEOTF(lin);
    [branch] if (sp == 2) return lin / SCRGB_WHITE;
    return sRGB_OETF(lin / SCRGB_WHITE);
}

// Convert an sRGB-authored diagnostic colour to the active display encoding.
float3 DiagToDisplay(float3 srgb, int sp)
{
    if (sp <= 1) return srgb;
    float3 lin = sRGB_EOTF(srgb) * SCRGB_WHITE;
    [branch] if (sp == 3) return PQ_InvEOTF(lin);
    return lin / SCRGB_WHITE;
}

// ==============================================================================
// 6. Oklab
// ==============================================================================

float3 LinearToOklab(float3 rgb, int sp)
{
    float3x3 m;
    if (sp >= 3)
        m = RGB2020_to_LMS;
    else
        m = RGB709_to_LMS;

    float3 lms   = mul(m, rgb);
    float3 lms_p = sign(lms) * pow(max(abs(lms), FLT_MIN), 1.0 / 3.0);
    return mul(LMS_to_Oklab, lms_p);
}

// ==============================================================================
// 7. Zone Logic
// ==============================================================================

int GetZone(float nl)
{
    if (nl <  0.0)       return 0;
    if (nl <  ZONE_I)    return 1;
    if (nl <  ZONE_II)   return 2;
    if (nl <  ZONE_III)  return 3;
    if (nl <  ZONE_IV)   return 4;
    if (nl <  ZONE_V)    return 5;
    if (nl <  ZONE_VI)   return 6;
    if (nl <  ZONE_VII)  return 7;
    if (nl <  ZONE_VIII) return 8;
    if (nl <  ZONE_IX)   return 9;
    if (nl <  ZONE_X)    return 10;
    if (nl <  ZONE_XI)   return 11;
    return 12;
}

float3 FalseColorPalette(int zone)
{
    [flatten]
    switch (clamp(zone, 0, 12)) {
        case 0:  return float3(0.80, 0.00, 0.80);
        case 1:  return float3(0.06, 0.03, 0.12);
        case 2:  return float3(0.12, 0.06, 0.35);
        case 3:  return float3(0.12, 0.18, 0.65);
        case 4:  return float3(0.00, 0.42, 0.72);
        case 5:  return float3(0.00, 0.52, 0.38);
        case 6:  return float3(0.00, 0.78, 0.00);
        case 7:  return float3(0.48, 0.78, 0.00);
        case 8:  return float3(0.88, 0.88, 0.00);
        case 9:  return float3(1.00, 0.58, 0.00);
        case 10: return float3(0.96, 0.12, 0.06);
        case 11: return float3(1.00, 0.00, 0.00);
        case 12: return float3(1.00, 0.42, 0.72);
        default: return float3(0.50, 0.50, 0.50);
    }
}

// ==============================================================================
// 8. Diagnostic Renderers
// ==============================================================================

float3 RenderFalseColor(float3 lin, float3 lumaC, float wp, int sp)
{
    float nl = dot(lin, lumaC) / wp;
    return DiagToDisplay(FalseColorPalette(GetZone(nl)), sp);
}

float3 RenderClipping(float3 lin, float3 enc, float3 lumaC, float wp, float opacity, int sp)
{
    float nl = dot(lin, lumaC) / wp;

    float3 overlay = enc;
    float  mask    = 0.0;

    if (nl < ZONE_I && nl >= 0.0) {
        overlay = DiagToDisplay(float3(0.0, 0.0, 1.0), sp);
        mask = 1.0;
    }

    if (nl >= ZONE_X) {
        overlay = DiagToDisplay(float3(1.0, 0.0, 0.0), sp);
        mask = 1.0;
    }

    if (nl < 0.0) {
        overlay = DiagToDisplay(float3(1.0, 0.0, 1.0), sp);
        mask = 1.0;
    }

    return lerp(enc, overlay, mask * opacity);
}

float3 RenderLuminance(float3 lin, float3 lumaC, int sp)
{
    float luma = max(dot(lin, lumaC), 0.0);
    return Encode(float3(luma, luma, luma), sp);
}

float3 RenderChromaMap(float3 lin, float3 enc, float opacity, int sp, float3 lumaC)
{
    float3 lab = LinearToOklab(lin, sp);
    float  L   = max(abs(lab.x), FLT_MIN);
    float  c   = length(lab.yz) / L;

    float rel = GetChromaReliability(dot(lin, lumaC));
    c *= rel;

    float3 cold = float3(0.1, 0.2, 0.8);
    float3 mid  = float3(0.1, 0.8, 0.2);
    float3 hot  = float3(0.9, 0.1, 0.1);

    float t = saturate(c / 0.30);
    float3 heat = (t < 0.5) ? lerp(cold, mid, t * 2.0)
                            : lerp(mid, hot, (t - 0.5) * 2.0);

    float3 diag = DiagToDisplay(heat, sp);
    return lerp(enc, diag, opacity * rel);
}

// ── 5. Neutral Grey Detector (FIXED: Normalized Tint Warnings) ───────────────
float3 RenderNeutral(float3 lin, float3 enc, float opacity, int sp, float3 lumaC)
{
    float3 lab = LinearToOklab(lin, sp);
    float  L   = max(abs(lab.x), FLT_MIN);
    float  c   = length(lab.yz) / L;

    // Fade out noise in deep blacks
    float rel = GetChromaReliability(dot(lin, lumaC));
    if (rel <= 0.0) return enc;

    // Only analyze pixels close to neutral (Chroma < 0.03)
    float mask = 1.0 - saturate(c / 0.03);
    mask *= mask; // Smooth fade out
    mask *= rel;

    // 1. Normalize the direction of the tint so it ALWAYS shows a bright, visible color
    // lab.y = Green(-)/Red(+) axis | lab.z = Blue(-)/Yellow(+) axis
    float2 dir = lab.yz / (length(lab.yz) + 1e-6);

    // 2. Map Oklab direction to RGB visual warnings
    float3 tintIndicator = saturate(float3(
        0.5 + dir.y * 1.0 + dir.x * 1.0,  // Adds Red/Yellow
        0.5 + dir.y * 1.0 - dir.x * 1.0,  // Adds Green/Yellow
        0.5 - dir.y * 2.0                 // Adds Blue
    ));

    // 3. Distance to pure neutral (Target Lock)
    // If chroma is under 1.0% (c < 0.01), it crossfades into the Green Target color.
    float tintAmt = saturate(c / 0.01);
    
    // Target Color: Teal/Green (Means mathematically perfect white balance)
    float3 targetColor = float3(0.0, 1.0, 0.5);

    // Blend between the Bright Green Target and the Bright Warning Color
    float3 diagColor = lerp(targetColor, tintIndicator, tintAmt);

    float3 diag = DiagToDisplay(diagColor, sp);
    return lerp(enc, diag, mask * opacity);
}

float3 RenderSkinTone(float3 lin, float3 enc, float opacity, int sp)
{
    float3 lab = LinearToOklab(lin, sp);
    float  L   = max(abs(lab.x), FLT_MIN);
    float  cR  = length(lab.yz) / L;

    float hue = atan2(lab.z, lab.y);
    float hueDeg = hue * (180.0 / PI);
    if (hueDeg < 0.0) hueDeg += 360.0;

    bool hueOK    = (hueDeg > 15.0 && hueDeg < 75.0);
    bool chromaOK = (cR > 0.025 && cR < 0.18);
    bool lightOK  = (lab.x > 0.15 && lab.x < 0.95);

    float mask = (hueOK && chromaOK && lightOK) ? 1.0 : 0.0;

    float3 highlight = DiagToDisplay(float3(0.2, 0.9, 0.6), sp);
    return lerp(enc, highlight, mask * opacity * 0.6);
}

float3 RenderBlackPoint(float3 lin, float3 enc, float3 lumaC, float wp, float bpLevel, float opacity, int sp)
{
    float nl = dot(lin, lumaC) / wp;
    float bp = bpLevel;

    float below = (nl < bp && nl >= 0.0) ? 1.0 : 0.0;
    float inToe = (nl < 2.0 * bp && nl >= bp) ? 0.6 : 0.0;

    float mask = max(below, inToe);
    float3 colour = (below > 0.0) ? float3(0.9, 0.1, 0.1) : float3(1.0, 0.6, 0.1);

    return lerp(enc, DiagToDisplay(colour, sp), mask * opacity);
}

float3 RenderGamut(float3 lin, float3 enc, float opacity, int sp)
{
    float minC = min(min(lin.r, lin.g), lin.b);
    float maxC = max(max(lin.r, lin.g), lin.b);

    float mask = 0.0;
    float3 colour = float3(0.0, 0.0, 0.0);

    if (minC < 0.0) {
        colour = float3(1.0, 0.0, 1.0);
        mask = 1.0;
    }
    else if (minC < maxC * 0.01 && maxC > FLT_MIN) {
        colour = float3(0.8, 0.4, 0.0);
        mask = 0.5;
    }

    return lerp(enc, DiagToDisplay(colour, sp), mask * opacity);
}

float3 RenderZebra(float3 lin, float3 enc, float3 lumaC, float wp, float threshold, float opacity, float2 pos, int sp)
{
    float nl = dot(lin, lumaC) / wp;

    if (nl < threshold) return enc;

    float stripe = step(0.5, frac((pos.x + pos.y) * 0.125));
    float3 zebra = DiagToDisplay(float3(stripe, stripe, stripe), sp);

    return lerp(enc, zebra, opacity);
}

// ==============================================================================
// 9. Test Pattern Generators
// ==============================================================================

float3 GenerateZoneWedge(float2 uv, float wp, int sp)
{
    int strip = clamp((int)(uv.y * 12.0), 0, 11);

    static const float ZoneLuma[12] = {
        0.04419417382, 0.06250000000, 0.08838834764,
        0.12500000000, 0.17677669529, 0.25000000000,
        0.35355339059, 0.50000000000, 0.70710678118,
        1.00000000000, 2.00000000000, 4.00000000000
    };

    if (uv.x < 0.12)
    {
        float3 fc = FalseColorPalette(strip + 1);
        return DiagToDisplay(fc, sp);
    }

    float lumaVal = ZoneLuma[strip] * wp;
    return Encode(float3(lumaVal, lumaVal, lumaVal), sp);
}

float3 GenerateGradientAndBars(float2 uv, float wp, int sp)
{
    if (uv.y < 0.55)
    {
        float luma = uv.x * wp;
        return Encode(float3(luma, luma, luma), sp);
    }

    if (uv.y < 0.90)
    {
        float barWidth = 1.0 / 8.0;
        int   barIdx   = clamp((int)(uv.x / barWidth), 0, 7);

        float lev = 0.75 * wp;
        float3 bar;

        [flatten]
        switch (barIdx)
        {
            case 0:  bar = float3(lev, lev, lev); break;
            case 1:  bar = float3(lev, lev, 0.0); break;
            case 2:  bar = float3(0.0, lev, lev); break;
            case 3:  bar = float3(0.0, lev, 0.0); break;
            case 4:  bar = float3(lev, 0.0, lev); break;
            case 5:  bar = float3(lev, 0.0, 0.0); break;
            case 6:  bar = float3(0.0, 0.0, lev); break;
            default: bar = float3(0.0, 0.0, 0.0); break;
        }

        return Encode(bar, sp);
    }

    float grey18 = 0.18 * wp;
    return Encode(float3(grey18, grey18, grey18), sp);
}

// ==============================================================================
// 10. Spot Meter Overlay
// ==============================================================================

float3 OverlaySpotMeter(float3 current, float2 pos, float3 centreLin, float3 lumaC, float wp, int sp)
{
    float2 centre = float2(BUFFER_WIDTH * 0.5, BUFFER_HEIGHT * 0.5);
    float  r      = float(iSpotSize);
    float  nl     = dot(centreLin, lumaC) / wp;

    float dist = length(pos - centre);
    if (dist < r)
    {
        int z = GetZone(nl);
        return DiagToDisplay(FalseColorPalette(z), sp);
    }

    if (dist < r + 2.0)
        return DiagToDisplay(float3(0.9, 0.9, 0.9), sp);

    float chLen = r * 2.5;
    bool onCH = false;
    if (abs(pos.x - centre.x) < 0.8 && abs(pos.y - centre.y) < chLen && abs(pos.y - centre.y) > r + 2.0)
        onCH = true;
    if (abs(pos.y - centre.y) < 0.8 && abs(pos.x - centre.x) < chLen && abs(pos.x - centre.x) > r + 2.0)
        onCH = true;
    if (onCH)
        return DiagToDisplay(float3(0.8, 0.8, 0.8), sp);

    float barY    = centre.y + r + 12.0;
    float barH    = 10.0;
    float barW    = 180.0;
    float barLeft = centre.x - barW * 0.5;

    if (pos.y >= barY && pos.y < barY + barH &&
        pos.x >= barLeft && pos.x < barLeft + barW)
    {
        float t = (pos.x - barLeft) / barW;
        int seg = clamp((int)(t * 12.0), 0, 11);
        return DiagToDisplay(FalseColorPalette(seg + 1), sp);
    }

    // Zone-aware smooth marker position
    int seg = clamp(GetZone(nl), 1, 12) - 1;
    float lower = ZoneSegLower[seg];
    float upper = ZoneSegUpper[seg];
    float frac_in_seg = saturate((nl - lower) / max(upper - lower, FLT_MIN));
    float pos_on_bar = (float(seg) + frac_in_seg) / 12.0;
    float markerX = barLeft + pos_on_bar * barW;

    float markerY = barY - 3.0;
    if (pos.y >= markerY && pos.y < barY &&
        abs(pos.x - markerX) < (barY - pos.y) * 1.5)
        return DiagToDisplay(float3(1.0, 1.0, 1.0), sp);

    return current;
}

// ==============================================================================
// 11. Main Shader
// ==============================================================================

void PS_CalibrationMonitor(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 fragColor : SV_Target)
{
    int2 pos = int2(vpos.xy);

    [branch]
    if (iMode == 0) {
        fragColor = tex2Dfetch(SamplerBackBuffer, pos);
        return;
    }

    int    sp    = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float3 lumaC = (sp >= 3) ? Luma2020 : Luma709;
    float  wp    = (sp <= 1) ? SCRGB_WHITE : fWhitePoint;

    float3 enc = tex2Dfetch(SamplerBackBuffer, pos).rgb;
    float3 lin = Decode(enc, sp);

    float3 result = enc;

    switch (iMode)
    {
        case 1:  result = RenderFalseColor(lin, lumaC, wp, sp); break;
        case 2:  result = RenderClipping(lin, enc, lumaC, wp, fOverlayOpacity, sp); break;
        case 3:  result = RenderLuminance(lin, lumaC, sp); break;
        case 4:  result = RenderChromaMap(lin, enc, fOverlayOpacity, sp, lumaC); break;
        case 5:  result = RenderNeutral(lin, enc, fOverlayOpacity, sp, lumaC); break;
        case 6:  result = RenderSkinTone(lin, enc, fOverlayOpacity, sp); break;
        case 7:  result = RenderBlackPoint(lin, enc, lumaC, wp, fBPPreviewLevel, fOverlayOpacity, sp); break;
        case 8:  result = RenderGamut(lin, enc, fOverlayOpacity, sp); break;
        case 9:  result = RenderZebra(lin, enc, lumaC, wp, fZebraThreshold, fOverlayOpacity, float2(pos), sp); break;
        case 10: result = GenerateZoneWedge(texcoord, wp, sp); break;
        case 11: result = GenerateGradientAndBars(texcoord, wp, sp); break;
    }

    [branch]
    if (bSpotMeter && iMode != 10 && iMode != 11)
    {
        int2 centrePos = int2(BUFFER_WIDTH / 2, BUFFER_HEIGHT / 2);
        float3 centreEnc = tex2Dfetch(SamplerBackBuffer, centrePos).rgb;
        float3 centreLin = Decode(centreEnc, sp);

        result = OverlaySpotMeter(result, float2(pos), centreLin, lumaC, wp, sp);
    }

    [flatten]
    if (sp <= 1)
        result = saturate(result);

    fragColor = float4(result, 1.0);
}

// ==============================================================================
// 12. Technique
// ==============================================================================

technique CalibrationMonitor <
    ui_label = "Calibration Monitor v1.0.2 (Mastering Edition)";
    ui_tooltip = "Professional diagnostic overlay for Photoreal HDR V5.5.\n"
                 "Place AFTER Bilateral Contrast and Photoreal HDR.\n\n"
                 "Calibration Workflow:\n"
                 "  Exposure:    Use False Color - aim for GREEN on key subject\n"
                 "  Black Point: Use BP Preview - match to Photoreal Dehaze slider\n"
                 "  White Bal:   Use Neutral Detector - greys should show no tint\n"
                 "  Contrast:    Use Zone Wedge - verify all zones remain distinct\n"
                 "  Saturation:  Use Chroma Map - check for oversaturation (red)\n"
                 "  Skin Tones:  Use Skin Indicator - verify faces are highlighted\n\n"
                 "Modes:\n"
                 "  1  False Color       Exposure zones (cinema palette)\n"
                 "  2  Clipping Warning  Over/under-exposure overlay\n"
                 "  3  Luminance Only    Greyscale (removes colour)\n"
                 "  4  Chroma Map        Saturation heat map (Oklab)\n"
                 "  5  Neutral Detector  White-balance verification\n"
                 "  6  Skin Tone         Highlights skin-tone Oklab range\n"
                 "  7  Black Point       Preview dehaze threshold\n"
                 "  8  Gamut Warning     Out-of-gamut pixel detection\n"
                 "  9  Zebra Stripes     Broadcast-style highlight zebra\n"
                 " 10  Zone Wedge        Test pattern: zone luminance steps\n"
                 " 11  Gradient+Bars     Test pattern: ramp + SMPTE bars\n\n"
                 "v1.0.2 Fixes:\n"
                 "  - Neutral Detector now shows tint direction correctly\n"
                 "  - Chroma diagnostics fade out in unstable near-black regions\n"
                 "  - Spot Meter center sample is fetched once per pixel\n"
                 "  - Spot Meter marker now follows true zone widths\n"
                 "  - Gradient+Bars code path simplified";
>
{
    pass Main
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_CalibrationMonitor;
    }
}