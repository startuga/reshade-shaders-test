// =================================================================================================
// Photoreal HDR + RenoDX PsychoV-22 Master Pipeline (V1.2 - Seam Integrity Edition)
// =================================================================================================
//
// Architecture: TWO INDEPENDENT COLOR SCIENCE DOMAINS, COMPOSED AT THE RGB BOUNDARY
//
//   STAGE 1 - Photoreal Grader (Ottoson Hunt-Pointer-Estevez D65 basis, white @ MB (0.5, 0.5))
//     - Stop-domain EV exposure, isoluminant LMS white balance (Delta-Y locked)
//     - Rational dehaze/black point with C1 shadow floor
//     - Filmic log-domain contrast pivoted at 18% grey with stop recovery
//     - Troland highlight bleaching (REINSTATED V1.2)
//     - MacLeod-Boynton purity/vibrance, 3D Melanin-Hemoglobin skin locus, Abney compensation,
//       radial-only gamut guard (all constants valid ONLY in this basis - do not mix bases)
//
//   HANDOFF - Container-primary conversion AND scale normalization (nits -> white-relative),
//     performed only for the PsychoV path. Negatives preserved across primaries, then WCG
//     excursions pulled achromatic just enough to satisfy the 709-representability contract
//     (V1.2: prevents hue mirroring through the observer model's abs() fold).
//
//   STAGE 2 - RenoDX PsychoV-22 (Stockman-Sharpe 2deg CIE 170-2 weighted LMS basis)
//     Adapted from renodx::tonemap::psychov (C) Carlos Lopez, MIT license, deviations inline.
//     Operates exclusively WHITE-RELATIVE in BT.709 linear. Contract at every call site:
//       IN: nits -> normalize -> white-relative 709 -> TM -> denormalize -> OUT: nits
//
// V1.2 Changes (audit-driven):
//   - FIX (Hue Mirroring, HDR10/HLG ship-blocker): wide-gamut negative 709 components were being
//     folded through abs() inside the observer model, reflecting chromaticity about the neutral
//     axis (2020 green entered as magenta). Closed-form luminance-preserving desaturation now
//     enforces the 709-representability contract at the seam before the TM.
//   - FIX (Degenerate Shoulder): Anchor Out >= peak collapsed the NR curve (everything -> peak).
//     Anchor is now clamped to 98% of the effective peak at the call site.
//   - POLICY (Hull Handling): PsychoV Gamut Strength default raised 0.00 -> 0.50. Enabling a tone
//     mapper implies consent to device-hull handling; 0 remains available but documented as
//     "not recommended" with the TM active (out-of-hull content falls to hard encoder clipping).
//   - RESTORE (Stage 1 Bleaching): fBleaching reinstated and wired to the Troland stage and to
//     debug visualization 3. Removes the V1.1 zombie call site / permanently-blue debug view.
//   - CLEANUP: Auto-h headroom floor is now a compile-time constant (10^0.1); no runtime
//     transcendental round-trip. Unused texcoord varying removed from VS_Output. Khronos path
//     now applies the same negative-luma bypass policy as PsychoV. Unit-contract markers added
//     at both seam points.
//
// Carried from V1.1:
//   - Factory defaults remain a bit-transparent bypass (TM off).
//   - Zero shared constants across the two color-science domains.
//   - PsychoV hue restore / M-cone crosstalk intrinsic (faithful to upstream).
//   - SDR tone-map ceiling == diffuse white; extended-range scRGB negatives bypass all stages.
//
// References:
// - https://github.com/crosire/reshade-shaders/blob/slim/Shaders/ReShade.fxh
// - RenoDX psychov test22: Kunkel & Reinhard 2010 doi:10.1145/1836248.1836251;
//   Jiang & Fairchild 2021 doi:10.2352/J.ImagingSci.Technol.2021.65.5.050401;
//   MacLeod & Boynton 1979 doi:10.1364/JOSA.69.001183; CIE 170-2; CVRL.org
// =================================================================================================

#include "ReShade.fxh"

// =================================================================================================
// 1. Constants & Matrices
// =================================================================================================

#if defined(__RESHADE__) && __RESHADE__ < 40800
    #error "Photoreal HDR requires ReShade 4.8.0 or newer."
#endif

#ifndef BUFFER_COLOR_SPACE
    #define BUFFER_COLOR_SPACE 1
#endif

static const float FLT_MIN              = 1.175494351e-38;
static const float SCRGB_WHITE_NITS     = 80.0;
static const float NEUTRAL_EPS          = 1e-6;
static const float PSYCHO_EPS           = 1e-6;
static const float PI                   = 3.14159265358979323846;

// --- sRGB (IEC 61966-2-1:1999) -------------------------------------------------------------------
static const float SRGB_THRESHOLD_EOTF  = 0.040448236277123205;
static const float SRGB_THRESHOLD_OETF  = 0.003130668442501796;
static const float SRGB_GAMMA           = 2.4;
static const float SRGB_INV_GAMMA       = 0.41666666666666667;

// --- ST.2084 (PQ) --------------------------------------------------------------------------------
static const float PQ_M1                = 0.1593017578125;
static const float PQ_M2                = 78.84375;
static const float PQ_C1                = 0.8359375;
static const float PQ_C2                = 18.8515625;
static const float PQ_C3                = 18.6875;
static const float PQ_PEAK_LUMINANCE    = 10000.0;
static const float PQ_INV_M1            = 6.2773946360153257;
static const float PQ_INV_M2            = 0.012683313515655966;

// --- Chroma reliability (STAGE 1 domain: FRACTIONS OF DIFFUSE WHITE, V6.1.6 fix) ------------------
static const float CHROMA_RELIABILITY_START     = 5e-4;   // 0.05% diffuse white
static const float CHROMA_STABILITY_THRESH      = 2e-3;   // 0.20% diffuse white
static const float INV_CHROMA_RELIABILITY_SPAN  = 1.0 / (CHROMA_STABILITY_THRESH - CHROMA_RELIABILITY_START);

static const float3 Luma709             = float3(0.2126, 0.7152, 0.0722);
static const float3 Luma2020            = float3(0.2627, 0.6780, 0.0593);

// --- Trolands (STAGE 1) --------------------------------------------------------------------------
static const float TROLAND_LMS_SCALE    = 4.0;
static const float TROLAND_HALF_SAT     = 8000.0;

// =================================================================================================
// STAGE 1 BASIS: Ottoson HPE(D65). Row sums = 1 (gray-preserving). White = MB (0.5, 0.5).
// All Stage 1 constants are calibrated to THIS basis exclusively.
// =================================================================================================
static const float3x3 RGB709_to_LMS = float3x3(
    0.4122214708,  0.5363325363,  0.0514459929,
    0.2119034982,  0.6806995451,  0.1073969566,
    0.0883024619,  0.2817188376,  0.6299787005
);
static const float3x3 LMS_to_RGB709 = float3x3(
     4.0767416621, -3.3077115913,  0.2309699292,
    -1.2684380046,  2.6097574011, -0.3413193965,
    -0.0041960863, -0.7034186147,  1.7076147010
);
// DCI-P3-D65 through the same HPE(D65) basis. Row sums = 1.
// PROVENANCE NOTE: least-exercised matrix here; validate round-trip on CPU before mastering use.
static const float3x3 LMS_to_P3D65 = float3x3(
     3.12776899, -2.25713580,  0.12936681,
    -1.09100905,  2.41333176, -0.32232271,
    -0.02601081, -0.50804133,  1.53405214
);
static const float3x3 RGB2020_to_LMS = float3x3(
    0.6167596970,  0.3601880240,  0.0230522790,
    0.2651316740,  0.6358515800,  0.0990167460,
    0.1001279150,  0.2038783840,  0.6959937010
);
static const float3x3 LMS_to_RGB2020 = float3x3(
     2.1398540771, -1.2462788877,  0.1064290765,
    -0.8846737634,  2.1631158093, -0.2784377818,
    -0.0486976682, -0.4543507342,  1.5030526721
);
static const float2 MB_WHITE_D65 = float2(0.5, 0.5);

// --- Container primary conversions (HANDOFF only; luma-preserving; negatives preserved) -----------
static const float3x3 MAT_BT2020_TO_BT709 = float3x3(
     1.6605, -0.5876, -0.0729,
    -0.1246,  1.1329, -0.0083,
    -0.0181, -0.1006,  1.1187
);
static const float3x3 MAT_BT709_TO_BT2020 = float3x3(
    0.6274, 0.3293, 0.0433,
    0.0690, 0.9195, 0.0114,
    0.0163, 0.0880, 0.8956
);

// =================================================================================================
// STAGE 2 BASIS: Stockman-Sharpe (2000) 2deg / CIE 170-2 physiological weights.
// Yf = Lw + Mw. WeighLMS(D65-gray via 709) sums to exactly 1.0; the 2020 handoff lands at
// 1.0487 - harmless because every Stage 2 quantity derives from the live matrices, but any
// HARDCODED constant assuming yf == 1 must never be introduced here.
// =================================================================================================
static const float3 LMS_WEIGHTS_SS = float3(0.68990272, 0.34832189, 1.93485343);

static const float3x3 MAT_BT709_TO_LMS_SS = float3x3(
    0.267942, 0.682090, 0.061988,
    0.079865, 0.702156, 0.084434,
    0.009992, 0.061596, 0.491150
);
static const float3x3 MAT_LMS_SS_TO_BT709 = float3x3(
     5.249836, -5.118883,  0.217415,
    -0.593240,  2.024436, -0.273148,
    -0.032401, -0.149772,  2.065872
);

// --- PsychoV auto-compression references -----------------------------------------------------------
// Kunkel & Reinhard 2010: ~3.7 log10 simultaneous range (conservative default).
// Jiang & Fairchild 2021: 3.3 avg / 3.47 OBS1 @1600cd/m2; fits 3.24 @452 / 3.40 @1600.
static const float PSYCHO22_REF_RANGE_LOG10     = 3.7;
static const float PSYCHO22_MIN_AUTO_H          = 1.0;
static const float PSYCHO22_MAX_AUTO_H          = 12.0;         // keeps pow() finite at PQ extremes
static const float PSYCHO22_MIN_HEADROOM_RATIO  = 1.258925412;  // 10^0.1 (compile-time constant)

// --- Zone System ----------------------------------------------------------------------------------
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

// =================================================================================================
// 2. Texture & Sampler
// =================================================================================================

texture2D TextureBackBuffer : COLOR;
sampler2D SamplerBackBuffer
{
    Texture   = TextureBackBuffer;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};

// =================================================================================================
// 3. UI Parameters
// =================================================================================================

// --- Display & Pipeline Context ---
uniform int iColorSpaceOverride <
    ui_type     = "combo";
    ui_label    = "Color Space Override";
    ui_items    = "Auto (Default via ReShade)\0sRGB (SDR)\0scRGB (HDR Linear)\0HDR10 (PQ)\0HLG (HDR)\0";
    ui_category = "1. Display & Pipeline Context";
> = 0;

uniform float fWhitePoint <
    ui_type     = "slider";
    ui_min      = 80.0; ui_max = 10000.0; ui_step = 1.0;
    ui_label    = "Reference White (Nits)";
    ui_tooltip  = "Paper white anchor for HDR mapping (BT.2408 default 203).";
    ui_category = "1. Display & Pipeline Context";
> = 203.0;

uniform float fDisplayPeakNits <
    ui_type     = "slider";
    ui_min      = 80.0; ui_max = 4000.0; ui_step = 10.0;
    ui_label    = "Display Peak Luminance (Nits)";
    ui_tooltip  = "Hardware peak of your display. Clamped to diffuse white automatically in SDR.";
    ui_category = "1. Display & Pipeline Context";
> = 800.0;

// --- Stage 1: Scene Grade (Photoreal, native HPE basis) ---
uniform float fExposure <
    ui_type = "slider"; ui_min = -3.00; ui_max = 3.00; ui_step = 0.01;
    ui_label = "Exposure (EV)";
    ui_tooltip = "Linear EV shift: multiply by 2^EV.";
    ui_category = "2. Scene Grade";
> = 0.00;

uniform float fTemperature <
    ui_type = "slider"; ui_min = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label = "Color Temperature (LMS)";
    ui_tooltip = "Isoluminant, Delta-Y = 0. Negative = Cooler, Positive = Warmer.";
    ui_category = "2. Scene Grade";
> = 0.00;

uniform float fTint <
    ui_type = "slider"; ui_min = -0.50; ui_max = 0.50; ui_step = 0.001;
    ui_label = "Color Tint (LMS)";
    ui_tooltip = "Isoluminant green-magenta balance (Delta-Y = 0).";
    ui_category = "2. Scene Grade";
> = 0.00;

uniform float fBlackPoint <
    ui_type = "slider"; ui_min = 0.000; ui_max = 0.050; ui_step = 0.001;
    ui_label = "Dehaze / Black Point";
    ui_tooltip = "Negative-luma (extended-range scRGB WCG) pixels bypass dehaze untouched.";
    ui_category = "2. Scene Grade";
> = 0.000;

uniform float fShadowFloor <
    ui_type = "slider"; ui_min = 0.00; ui_max = 0.50; ui_step = 0.005;
    ui_label = "Dehaze Shadow Floor";
    ui_category = "2. Scene Grade";
> = 0.03;

uniform float fContrast <
    ui_type = "slider"; ui_min = 0.80; ui_max = 1.50; ui_step = 0.001;
    ui_label = "Filmic Contrast";
    ui_tooltip = "Log-domain luminance power curve pivoted at 18% grey.";
    ui_category = "2. Scene Grade";
> = 1.00;

uniform float fContrastPivot <
    ui_type = "slider"; ui_min = 0.01; ui_max = 1.00; ui_step = 0.01;
    ui_label = "Contrast Pivot (fraction of Reference White)";
    ui_category = "2. Scene Grade";
> = 0.17677669529;

uniform float fShadows <
    ui_type = "slider"; ui_min = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label = "Shadows (Log Recovery)";
    ui_category = "2. Scene Grade";
> = 0.0;

uniform float fHighlights <
    ui_type = "slider"; ui_min = -1.0; ui_max = 1.0; ui_step = 0.001;
    ui_label = "Highlights (Log Recovery)";
    ui_category = "2. Scene Grade";
> = 0.0;

uniform float fSaturation <
    ui_type = "slider"; ui_min = 0.00; ui_max = 2.00; ui_step = 0.01;
    ui_label = "Purity / Saturation (MacLeod-Boynton)";
    ui_tooltip = "Set 1.0 = neutral. Set 0.0 = exact R=G=B monochrome (overrides Vibrance).";
    ui_category = "2. Scene Grade";
> = 1.00;

uniform float fVibrance <
    ui_type = "slider"; ui_min = -1.00; ui_max = 2.00; ui_step = 0.01;
    ui_label = "Smart Saturation (Vibrance)";
    ui_tooltip = "Gamut-relative Weber-Fechner demand. Identical normalization in clamped and unclamped modes.";
    ui_category = "2. Scene Grade";
> = 0.00;

uniform float fSkinProtection <
    ui_type = "slider"; ui_min = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label = "Skin Tone Protection";
    ui_tooltip = "3D Melanin-Hemoglobin locus (Fitzpatrick I-VI), evaluated in the STAGE 1 basis.";
    ui_category = "2. Scene Grade";
> = 0.85;

uniform float fAbneyCorrection <
    ui_type = "slider"; ui_min = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label = "Abney Hue Compensation";
    ui_category = "2. Scene Grade";
> = 0.00;

// REINSTATED V1.2: was dropped with the old TM block in V1.1, leaving a zombie call site.
uniform float fBleaching <
    ui_type = "slider"; ui_min = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label = "Highlight Bleaching (Trolands)";
    ui_tooltip = "Physiological highlight desaturation toward a white-hot core. 0.0 = bypass.\n"
                 "Operates in STAGE 1 (pre-tonemap). Visualize with Debug: Bleaching Factor.";
    ui_category = "2. Scene Grade";
> = 0.00;

uniform int iGamutTarget <
    ui_type     = "combo";
    ui_label    = "Gamut Guard Target Limit";
    ui_items    = "Auto (Container Gamut)\0Rec. 709 (SDR Standard)\0DCI-P3 (Cinema)\0Rec. 2020 (UHD Display)\0Bypass / Unclamped\0";
    ui_tooltip  = "STAGE 1 chromaticity guard (MacLeod-Boynton, HPE basis).\n"
                  "scRGB note: 'Auto' guards against the Rec.2020 volume because scRGB carries wide\n"
                  "gamut via extended/negative 709 excursions; the linear container passes it losslessly.";
    ui_category = "2. Scene Grade";
> = 0;

uniform float fGamutGuardKnee <
    ui_type = "slider"; ui_min = 0.00; ui_max = 0.50; ui_step = 0.01;
    ui_label = "Gamut Guard Knee";
    ui_tooltip = "Soft-knee boundary compression. Default 0.0 keeps factory settings a true bypass.";
    ui_category = "2. Scene Grade";
> = 0.00;

uniform float fMConeCrosstalk <
    ui_type = "slider"; ui_min = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label = "Spectral M-Cone Crosstalk (Stage 1)";
    ui_category = "2. Scene Grade";
> = 0.00;

// --- Stage 2a: Khronos PBR Neutral (optional alternative TM) ---
uniform float fCompressionStart <
    ui_type = "slider"; ui_min = 0.50; ui_max = 0.95; ui_step = 0.01;
    ui_label = "Compression Start (%)";
    ui_category = "3. Tone Mapping";
> = 0.80;

uniform float fDesaturationStrength <
    ui_type = "slider"; ui_min = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label = "Desaturation Strength (Khronos)";
    ui_category = "3. Tone Mapping";
> = 0.15;

// --- Stage 2b: RenoDX PsychoV-22 ---
uniform int iToneMapperMode <
    ui_type     = "combo";
    ui_label    = "Tone Mapping Operator";
    ui_items    = "Bypass\0Khronos PBR Neutral\0RenoDX PsychoV-22\0";
    ui_tooltip  = "PsychoV-22: physiological observer cascade + device-hull mapping in\n"
                  "Stockman-Sharpe weighted LMS. Hue-direction restore and M-cone crosstalk are\n"
                  "INTRINSIC (fixed behavior, no strength knobs - faithful to upstream).\n"
                  "All grading remains in Stage 1; the TM receives neutral grade multipliers.";
    ui_category = "3. Tone Mapping";
> = 0;

uniform float fPsychoCompression <
    ui_type = "slider"; ui_min = 0.00; ui_max = 5.00; ui_step = 0.01;
    ui_label = "PsychoV Compression Shoulder (h)";
    ui_tooltip = "0.0 = AUTO from simultaneous dynamic range literature (~3.7 log10, centered).\n"
                 "Auto includes a degenerate-headroom floor so SDR (peak == white) stays finite.\n"
                 "> 0 = manual exponent.";
    ui_category = "3. Tone Mapping";
> = 0.00;

uniform float fPsychoAnchorIn <
    ui_type = "slider"; ui_min = 0.01; ui_max = 1.00; ui_step = 0.01;
    ui_label = "Adapted State (Anchor In)";
    ui_tooltip = "Source adapted-state anchor (0.18 = 18% mid-grey). Input == anchor maps EXACTLY to Anchor Out for ANY h.";
    ui_category = "3. Tone Mapping";
> = 0.18;

uniform float fPsychoAnchorOut <
    ui_type = "slider"; ui_min = 0.01; ui_max = 1.00; ui_step = 0.01;
    ui_label = "Output Background (Anchor Out)";
    ui_tooltip = "Internally clamped to 98% of the effective peak (degenerate-shoulder guard,\n"
                 "V1.2): anchor >= peak collapses the compression curve.";
    ui_category = "3. Tone Mapping";
> = 0.18;

// POLICY V1.2: default 0.50 (was 0.00). Enabling a tone mapper implies consent to device-hull
// handling; 0 disables entirely and is NOT recommended with the TM active (out-of-hull content
// then terminates in hard encoder clipping).
uniform float fPsychoGamutStrength <
    ui_type = "slider"; ui_min = 0.00; ui_max = 1.00; ui_step = 0.01;
    ui_label = "PsychoV Gamut Compression Strength";
    ui_tooltip = "Device-hull compression lerp applied after tone mapping.\n"
                 "Recommended >= 0.5 whenever a tone mapper is active. At 0, out-of-hull colors\n"
                 "are hard-clipped by the display encoder (hue distortion at the gamut edge).";
    ui_category = "3. Tone Mapping";
> = 0.50;

uniform int iPsychoGamutMode <
    ui_type     = "combo";
    ui_label    = "PsychoV Gamut Bound";
    ui_items    = "Auto (Container)\0Rec. 709\0Rec. 2020\0";
    ui_category = "3. Tone Mapping";
> = 0;

// --- Debug ---
uniform int iDebugMode <
    ui_type     = "combo";
    ui_label    = "Debug Visualization";
    ui_items    = "Off\0"
                  "Luminance (False Color Stops)\0"
                  "Zone Map\0"
                  "Bleaching Factor\0"
                  "MB Purity\0"
                  "MB Hue Wheel\0"
                  "LMS Cone Response\0"
                  "Negative / WCG\0"
                  "Compression Map\0"
                  "Skin Protection Mask\0";
    ui_category = "Debug";
> = 0;

// =================================================================================================
// 4. Math Utilities
// =================================================================================================

float PowNonNegPreserveZero(float x, float e) { return (x <= 0.0) ? 0.0 : pow(x, e); }
float3 PowNonNegPreserveZero3(float3 x, float e)
{
    return float3(PowNonNegPreserveZero(x.r, e), PowNonNegPreserveZero(x.g, e), PowNonNegPreserveZero(x.b, e));
}
float SqrtIEEE(float x) { return sqrt(max(x, 0.0)); }
bool3 IsNan3(float3 v) { return (asuint(v) & 0x7FFFFFFFu) > 0x7F800000u; }
bool3 IsInf3(float3 v) { return (asuint(v) & 0x7FFFFFFFu) == 0x7F800000u; }

float PSY22_DivideSafe(float n, float d, float fallback) { return (abs(d) > PSYCHO_EPS) ? (n / d) : fallback; }
float3 PSY22_CopySign(float3 mag, float3 sign_src) { return (sign_src >= 0.0) ? abs(mag) : -abs(mag); }

// =================================================================================================
// 5. EOTF / OETF
// =================================================================================================

float3 sRGB_EOTF(float3 V)
{
    float3 abs_V  = abs(V);
    float3 lin_lo = abs_V / 12.92;
    float3 lin_hi = PowNonNegPreserveZero3((abs_V + 0.055) / 1.055, SRGB_GAMMA);
    return sign(V) * ((abs_V <= SRGB_THRESHOLD_EOTF) ? lin_lo : lin_hi);
}
float3 sRGB_OETF(float3 L)
{
    float3 abs_L  = abs(L);
    float3 enc_lo = abs_L * 12.92;
    float3 enc_hi = 1.055 * PowNonNegPreserveZero3(abs_L, SRGB_INV_GAMMA) - 0.055;
    return sign(L) * ((abs_L <= SRGB_THRESHOLD_OETF) ? enc_lo : enc_hi);
}
float3 PQ_EOTF(float3 N)
{
    N = saturate(N);
    float3 Np  = PowNonNegPreserveZero3(N, PQ_INV_M2);
    float3 num = max(Np - PQ_C1, 0.0);
    float3 den = max(PQ_C2 - PQ_C3 * Np, FLT_MIN);
    return PowNonNegPreserveZero3(num / den, PQ_INV_M1) * PQ_PEAK_LUMINANCE;
}
float3 PQ_InverseEOTF(float3 L)
{
    L = clamp(L, 0.0, PQ_PEAK_LUMINANCE);
    float3 Lp  = PowNonNegPreserveZero3(L / PQ_PEAK_LUMINANCE, PQ_M1);
    float3 num = PQ_C1 + PQ_C2 * Lp;
    float3 den = 1.0 + PQ_C3 * Lp;
    return saturate(PowNonNegPreserveZero3(num / den, PQ_M2));
}
// NOTE: No inverse-OOTF pair for HLG. Game backbuffers are display-referred; applying the
// BT.2100 scene-referred system gamma here would double-apply it.
float3 HLG_EOTF(float3 x)
{
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    float3 r;
    r.r = (x.r <= 0.5) ? (x.r * x.r) / 3.0 : (exp((x.r - c) / a) + b) / 12.0;
    r.g = (x.g <= 0.5) ? (x.g * x.g) / 3.0 : (exp((x.g - c) / a) + b) / 12.0;
    r.b = (x.b <= 0.5) ? (x.b * x.b) / 3.0 : (exp((x.b - c) / a) + b) / 12.0;
    return r * 1000.0;
}
float3 HLG_OETF(float3 x)
{
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    float3 E = max(x / 1000.0, 0.0);
    float3 r;
    r.r = (E.r <= 1.0 / 12.0) ? sqrt(3.0 * E.r) : a * log(max(12.0 * E.r - b, FLT_MIN)) + c;
    r.g = (E.g <= 1.0 / 12.0) ? sqrt(3.0 * E.g) : a * log(max(12.0 * E.g - b, FLT_MIN)) + c;
    r.b = (E.b <= 1.0 / 12.0) ? sqrt(3.0 * E.b) : a * log(max(12.0 * E.b - b, FLT_MIN)) + c;
    return r;
}
float3 DecodeToLinear(float3 encoded, int space)
{
    [branch] if (space == 4) return HLG_EOTF(encoded);
    [branch] if (space == 3) return PQ_EOTF(encoded);
    [branch] if (space == 2) return encoded * SCRGB_WHITE_NITS;
    return sRGB_EOTF(encoded) * SCRGB_WHITE_NITS;
}
float3 EncodeFromLinear(float3 lin, int space)
{
    [branch] if (space == 4) return HLG_OETF(lin);
    [branch] if (space == 3) return PQ_InverseEOTF(lin);
    [branch] if (space == 2) return lin / SCRGB_WHITE_NITS;
    return sRGB_OETF(lin / SCRGB_WHITE_NITS);
}

// =================================================================================================
// 6. STAGE 1 Physiology (native HPE basis - DO NOT reuse these in the Stockman domain)
// =================================================================================================

float3 LMS_to_MB(float3 lms)
{
    float lum = max(lms.r + lms.g, FLT_MIN);
    return float3(lms.r / lum, lms.b / lum, lum);
}
float3 MB_to_LMS(float3 mb)
{
    return float3(mb.x * mb.z, mb.z - (mb.x * mb.z), mb.y * mb.z);
}

/**
 * Evaluate3DSkinLocusMB - 3D Melanin-Hemoglobin skin confidence.
 * Thresholds calibrated to the STAGE 1 Ottoson-HPE basis with white at (0.5, 0.5).
 */
float Evaluate3DSkinLocusMB(float luma_norm, float3 mb)
{
    float l_gate = smoothstep(0.505, 0.515, mb.x) * (1.0 - smoothstep(0.585, 0.620, mb.x));
    float expected_s_min = clamp(0.16 + 0.16 * luma_norm, 0.16, 0.32);
    float s_gate = smoothstep(expected_s_min - 0.03, expected_s_min + 0.02, mb.y) * (1.0 - smoothstep(0.44, 0.48, mb.y));
    float luma_gate = smoothstep(0.010, 0.035, luma_norm) * (1.0 - smoothstep(0.95, 1.30, luma_norm));
    return saturate(l_gate * s_gate * luma_gate);
}

float ComputeBlackPointRatio(float luma, float bpNits, float shadowFloor)
{
    float raw = max((luma - bpNits) / max(luma, FLT_MIN), shadowFloor);
    float t = saturate(luma / max(4.0 * bpNits, FLT_MIN));
    float smooth_t = t * t * (3.0 - 2.0 * t);
    return lerp(shadowFloor, raw, smooth_t);
}

/** Boundary solver valid for ANY pixel luminance (ratio-space sign test), STAGE 1 basis. */
float SolveGamutBoundaryExact(float2 chroma_direction, float3 luma_LMS_coeffs, float3x3 to_RGB_boundary, float2 mb_white)
{
    float t_low = 0.0;
    float t_high = 8.0;
    [unroll]
    for (int iter = 0; iter < 24; iter++)
    {
        float t = 0.5 * (t_low + t_high);
        float x = mb_white.x + t * chroma_direction.x;
        float y = mb_white.y + t * chroma_direction.y;
        float denom = x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
        if (denom <= FLT_MIN) { t_high = t; }
        else
        {
            float3 test_lms = float3(x, 1.0 - x, y);
            float3 rgb_point = mul(to_RGB_boundary, test_lms);
            float min_rgb = min(min(rgb_point.r, rgb_point.g), rgb_point.b);
            if (min_rgb < 0.0 || !(min_rgb >= 0.0)) t_high = t; else t_low = t;
        }
    }
    return t_low;
}

float3 ApplyTrolandBleachingLMS(float3 lms, float strength, float3 luma_LMS_coeffs)
{
    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0 || strength <= NEUTRAL_EPS) return lms;
    float3 safe_lms = max(lms, 0.0);
    float3 stimulus = safe_lms * TROLAND_LMS_SCALE;
    float stim_lm = 0.5 * (stimulus.r + stimulus.g);
    float availability = 1.0 / (1.0 + (stim_lm / max(TROLAND_HALF_SAT, FLT_MIN)));
    float k = lerp(1.0, availability, saturate(strength));
    float luma = dot(lms, luma_LMS_coeffs);
    float denom = dot(float3(1.0, 1.0, 1.0), luma_LMS_coeffs);
    float3 neutral = float3(1.0, 1.0, 1.0) * (luma / max(denom, FLT_MIN));
    return lerp(neutral, lms, k);
}

/** V6.1.6 purity/vibrance/gamut-guard. White-relative chroma reliability; grayscale precedence
 *  over vibrance; unconditional normalization solve; Delta-Y z-recovery. */
float3 ApplyMBPurityAndGamutGuardLMS(
    float3 lms, float purity_scale, float vibrance_amount, float skin_protection,
    int gamut_target_mode, float knee, float abney_correction,
    float3 luma_LMS_coeffs, float3x3 to_RGB_boundary, float2 mb_white, float whitePt)
{
    bool is_unclamped = (gamut_target_mode == 4);
    bool force_gamut_clamp = (gamut_target_mode == 1 || gamut_target_mode == 2 || gamut_target_mode == 3);

    if (!force_gamut_clamp &&
        abs(purity_scale - 1.0) < NEUTRAL_EPS && abs(vibrance_amount) < NEUTRAL_EPS &&
        knee < NEUTRAL_EPS && abney_correction < NEUTRAL_EPS)
    {
        return lms;
    }

    float lm_sum = lms.r + lms.g;
    if (lm_sum <= FLT_MIN) return lms;
    float luma = dot(lms, luma_LMS_coeffs);
    if (luma <= FLT_MIN) return lms;

    float ct = saturate((luma / max(whitePt, FLT_MIN) - CHROMA_RELIABILITY_START) * INV_CHROMA_RELIABILITY_SPAN);
    float chroma_reliability = ct * ct * (3.0 - 2.0 * ct);
    if (chroma_reliability <= 0.0 && purity_scale >= 1.0 && vibrance_amount >= 0.0 && !force_gamut_clamp) return lms;

    float3 mb = LMS_to_MB(lms);
    float2 chroma_offset = mb.xy - mb_white;
    float purity = SqrtIEEE(dot(chroma_offset, chroma_offset));

    if (purity < 1e-6 || purity_scale <= NEUTRAL_EPS)
    {
        mb.xy = mb_white;
        float denom = mb.x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + mb.y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
        mb.z = luma / max(denom, FLT_MIN);
        return MB_to_LMS(mb);
    }

    float2 chroma_dir = chroma_offset / max(purity, FLT_MIN);
    float relative_lightness = luma / max(whitePt, FLT_MIN);

    // Unconditional normalization solve: identical vibrance behavior in ALL gamut modes.
    float initial_max_purity = SolveGamutBoundaryExact(chroma_dir, luma_LMS_coeffs, to_RGB_boundary, mb_white);
    float relative_purity = saturate(purity / max(initial_max_purity, FLT_MIN));

    float effective_scale = purity_scale;
    float skin_confidence = Evaluate3DSkinLocusMB(relative_lightness, mb);
    float effective_skin_mask = saturate(skin_confidence * skin_protection);

    if (effective_scale > 1.0 && effective_skin_mask > NEUTRAL_EPS)
    {
        float master_boost = effective_scale - 1.0;
        master_boost *= (1.0 - effective_skin_mask * 0.85);
        effective_scale = 1.0 + master_boost;
    }

    if (abs(vibrance_amount) > NEUTRAL_EPS)
    {
        float saturation_demand = exp(-3.5 * pow(relative_purity, 1.25));
        float vibrance_gain = vibrance_amount * saturation_demand
                            * saturate(purity_scale)
                            * (1.0 - effective_skin_mask) * chroma_reliability;
        effective_scale *= max(1.0 + vibrance_gain, 0.0);
    }

    if (effective_scale > 1.0)
    {
        float boost = effective_scale - 1.0;
        boost = boost / (1.0 + 0.35 * relative_purity * boost);
        effective_scale = 1.0 + boost;
    }
    effective_scale = lerp(1.0, effective_scale, chroma_reliability);

    float2 scaled_chroma_offset = chroma_offset * effective_scale;

    if (abney_correction > NEUTRAL_EPS)
    {
        float angle = atan2(chroma_offset.y, chroma_offset.x);
        float abney_profile = 0.15 * sin(2.0 * angle + 0.4) * (1.0 + 0.3 * cos(angle));
        float shift = abney_profile * relative_purity * abney_correction * chroma_reliability;
        angle += shift;
        float sp = SqrtIEEE(dot(scaled_chroma_offset, scaled_chroma_offset));
        scaled_chroma_offset = float2(cos(angle), sin(angle)) * sp;
    }

    float scaled_purity = SqrtIEEE(dot(scaled_chroma_offset, scaled_chroma_offset));

    if (!is_unclamped && scaled_purity > FLT_MIN)
    {
        float2 new_chroma_dir = scaled_chroma_offset / scaled_purity;
        float max_purity = (abney_correction > NEUTRAL_EPS)
            ? SolveGamutBoundaryExact(new_chroma_dir, luma_LMS_coeffs, to_RGB_boundary, mb_white)
            : initial_max_purity;

        if (max_purity > FLT_MIN)
        {
            if (knee > FLT_MIN)
            {
                float threshold = max_purity * (1.0 - knee);
                if (scaled_purity > threshold && threshold > FLT_MIN)
                {
                    float excess = scaled_purity - threshold;
                    float headroom = max_purity - threshold;
                    float compressed = threshold + headroom * (1.0 - exp(-excess / max(headroom, FLT_MIN)));
                    scaled_chroma_offset = new_chroma_dir * compressed;
                    scaled_purity = compressed;
                }
            }
            float p_safe = max_purity * (1.0 - NEUTRAL_EPS);
            if (scaled_purity > p_safe)
            {
                scaled_chroma_offset = new_chroma_dir * p_safe;
                scaled_purity = p_safe;
            }

            mb.xy = mb_white + scaled_chroma_offset;
            float denom = mb.x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + mb.y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
            mb.z = luma / max(denom, FLT_MIN);
            float3 lms_test = MB_to_LMS(mb);
            float3 boundary_check = mul(to_RGB_boundary, lms_test);
            if (min(min(boundary_check.r, boundary_check.g), boundary_check.b) < 0.0)
            {
                float p_exact = SolveGamutBoundaryExact(new_chroma_dir, luma_LMS_coeffs, to_RGB_boundary, mb_white) * (1.0 - NEUTRAL_EPS);
                scaled_chroma_offset = new_chroma_dir * min(scaled_purity, p_exact);
            }
        }
    }

    mb.xy = mb_white + scaled_chroma_offset;
    float denom = mb.x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + mb.y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
    mb.z = luma / max(denom, FLT_MIN);
    return MB_to_LMS(mb);
}

float3 ApplyMConeCrosstalkLMS(float3 lms, float strength, float3 lms_before_tm, float3 luma_LMS_coeffs, float2 mb_white, float whitePt)
{
    if (strength <= NEUTRAL_EPS) return lms;
    float original_Y = dot(lms, luma_LMS_coeffs);
    if (original_Y <= 0.0) return lms;

    float3 global_white_lms = float3(1.0, 1.0, 1.0) * whitePt;
    float3 drive = abs(lms_before_tm) / max(global_white_lms, FLT_MIN);
    float l_over_m = max(drive.x - drive.y, 0.0);
    float s_over_m = max(drive.z - drive.y, 0.0);
    float ls_mixed_share = l_over_m + s_over_m > FLT_MIN ? min(l_over_m, s_over_m) / (l_over_m + s_over_m) : 0.0;
    float cone_complementary_gate = smoothstep(0.02, 0.15, ls_mixed_share);

    float3 mb = LMS_to_MB(lms);
    float2 offset = mb.xy - mb_white;
    float chroma = max(length(offset), FLT_MIN);
    float rg_pos = saturate(offset.x / chroma);
    float yv_pos = saturate(offset.y / chroma);
    float mixed = rg_pos * yv_pos > FLT_MIN ? min(abs(offset.x), abs(offset.y)) / max(max(abs(offset.x), abs(offset.y)), FLT_MIN) : 0.0;
    float acc_complementary_gate = saturate(2.0 * rg_pos * yv_pos * mixed);
    float spectral_confidence = 1.0 - max(cone_complementary_gate, acc_complementary_gate);

    float lm_share = drive.x + drive.y > FLT_MIN ? drive.y / (drive.x + drive.y) : 0.0;
    float l_bias = saturate(l_over_m / max(drive.x, FLT_MIN)) * saturate(lm_share / 0.25);
    float s_bias = 0.15 * saturate(s_over_m / max(drive.z, FLT_MIN));
    float m_bias_weight = spectral_confidence * (0.12 * l_bias + 0.025 * s_bias) * strength;

    float target_radius = length(offset);
    float2 m_offset = -mb_white;
    float m_radius = length(m_offset);

    if (target_radius > FLT_MIN && m_radius > FLT_MIN)
    {
        float2 bent_offset = lerp(offset, m_offset * (target_radius / m_radius), saturate(m_bias_weight));
        bent_offset *= target_radius / max(length(bent_offset), FLT_MIN);
        mb.xy = mb_white + bent_offset;
        float denom = mb.x * (luma_LMS_coeffs.r - luma_LMS_coeffs.g) + mb.y * luma_LMS_coeffs.b + luma_LMS_coeffs.g;
        mb.z = original_Y / max(denom, FLT_MIN);
        return MB_to_LMS(mb);
    }
    return lms;
}

// =================================================================================================
// 7. STAGE 2: RenoDX PsychoV-22 (faithful port, Stockman-Sharpe weighted LMS)
//    Input/output: BT.709 LINEAR, WHITE-RELATIVE. Deviations documented inline.
// =================================================================================================

float3 PSY22_WeighLMS(float3 lms)   { return lms * LMS_WEIGHTS_SS; }
float3 PSY22_UnweighLMS(float3 w)   { return w / LMS_WEIGHTS_SS; }
float  PSY22_YfFromLMS(float3 lms)  { float3 w = PSY22_WeighLMS(lms); return max(w.x + w.y, PSYCHO_EPS); }

float3 PSY22_MBFromWLMS(float3 w)  { float yf = max(w.x + w.y, PSYCHO_EPS); return float3(w.x / yf, w.z / yf, yf); }
float3 PSY22_WLMSFromMB(float3 mb) { return float3(mb.x * mb.z, mb.z - mb.x * mb.z, mb.y * mb.z); }

float3 PSY22_ToRelW(float3 lms, float3 adapt)    { return PSY22_WeighLMS(lms) / max(adapt, PSYCHO_EPS.xxx); }
float3 PSY22_FromRelW(float3 relw, float3 adapt) { return PSY22_UnweighLMS(relw * max(adapt, PSYCHO_EPS.xxx)); }

/** Adaptive-relative invariant: WeighLMS(A)/A == LMS_WEIGHTS for ANY adapted state A. */
float3 PSY22_RelNeutral() { return LMS_WEIGHTS_SS; }

/**
 * Auto-h from the centered simultaneous-range heuristic.
 * DEVIATIONS (documented): upstream clamps only h >= 1, which diverges when peak == anchor.
 * V1.2: headroom floor is the compile-time constant 10^0.1 (was a runtime exp2*log2 round-trip).
 */
float PSY22_AutoCompression(float anchor_out_yf, float peak_yf)
{
    float peak_over_anchor = PSY22_DivideSafe(max(peak_yf, PSYCHO_EPS), max(anchor_out_yf, PSYCHO_EPS), 1.0);
    peak_over_anchor = clamp(peak_over_anchor, PSYCHO22_MIN_HEADROOM_RATIO, 1e6);
    float h = (PSYCHO22_REF_RANGE_LOG10 * 0.5) / max(log10(peak_over_anchor), PSYCHO_EPS);
    return clamp(h, PSYCHO22_MIN_AUTO_H, PSYCHO22_MAX_AUTO_H);
}

float3 PSY22_ACCFromRelDelta(float3 delta_w)
{
    float3 nw = PSY22_RelNeutral();
    float mc1 = PSY22_DivideSafe(nw.x, nw.y, 0.0);
    float mc2 = PSY22_DivideSafe(nw.x + nw.y, nw.z, 0.0);
    return float3(
        delta_w.x + delta_w.y,
        delta_w.x - mc1 * delta_w.y,
        -delta_w.x - delta_w.y + mc2 * delta_w.z);
}

float PSY22_PurpleGate(float3 acc)
{
    float rg = acc.y, yv = acc.z;
    float chroma = max(length(acc.yz), 1e-6);
    float rg_pos = saturate(rg / chroma);
    float yv_pos = saturate(yv / chroma);
    float mixed  = saturate(min(abs(rg), abs(yv)) / max(max(abs(rg), abs(yv)), 1e-6));
    return saturate(2.0 * rg_pos * yv_pos * mixed);
}

/**
 * V1.1 RECONSTRUCTION NOTICE (unchanged in V1.2):
 * Upstream delegates to renodx::color::gamut::GamutCompressWeightedLMSCoreRGBBoundFromAdaptiveWeightedInput,
 * whose body is NOT part of the published test22 file. This stand-in matches the contract
 * (adaptive-relative weighted-LMS in, bound matrix semantic, strength as compression lerp):
 * monotone ray-desaturation toward the adapted achromatic axis until the signal fits the bound
 * RGB unit cube. Hue direction preserved; no clipping-to-white. Swap in the upstream helper
 * verbatim if/when building inside the RenoDX tree. Full-strength landings terminate at
 * anchor_out luminance - a policy choice the upstream body may or may not share.
 */
// V1.2.1 FIX: containment is evaluated in WHITE-RELATIVE units, so the display hull cube must
// be scaled by peak_rel (= peak_nits / paper_white), NOT left at the container-encode cube
// [0,1]^3. Testing [0,1] collapsed every super-white neutral onto (x+1)/2, capping the image
// near half of display peak. SDR (peak_rel == 1) is unaffected either way.
bool PSY22_BoundContains(float3 lms_abs_rel, int mode, float peak_rel)
{
    float3 rgb709 = mul(MAT_LMS_SS_TO_BT709, lms_abs_rel);
    float3 rgb = (mode == 1) ? mul(MAT_BT709_TO_BT2020, rgb709) : rgb709;
    return all(rgb >= -PSYCHO_EPS.xxx) && all(rgb <= (peak_rel + PSYCHO_EPS).xxx);
}

float3 PSY22_GamutCompress(float3 lms_abs, float3 achromatic_axis_lms, int mode, float strength, float peak_rel)
{
    if (strength <= NEUTRAL_EPS) return lms_abs;
    if (PSY22_BoundContains(lms_abs, mode, peak_rel)) return lms_abs;

    float3 dir = lms_abs - achromatic_axis_lms;
    if (dot(dir, dir) < 1e-12) return lms_abs;

    // Axis at s=0 is contained for any legal anchor (anchor clamped < peak upstream).
    float s_lo = 0.0, s_hi = 1.0;
    [unroll]
    for (int iter = 0; iter < 24; iter++)
    {
        float s = 0.5 * (s_lo + s_hi);
        if (PSY22_BoundContains(achromatic_axis_lms + s * dir, mode, peak_rel)) s_lo = s; else s_hi = s;
    }
    float s_safe = s_lo * (1.0 - 1e-5);
    float s_applied = lerp(1.0, s_safe, saturate(strength));
    return achromatic_axis_lms + s_applied * dir;
}

/**
 * PsychoV-22 core. Faithful to upstream psychotm_test22 with:
 *  - grade multipliers pinned neutral (Stage 1 owns grading); deprecated/reserved params pruned;
 *  - intrinsic (unconditional) hue restore at the fixed 0.5 blend, as upstream;
 *  - intrinsic M-cone crosstalk with no strength knob, as upstream;
 *  - overflow-safe algebraic rearrangement of the shoulder (identical math for shoulder > 0).
 *
 * DOMAIN CONTRACT (enforced by callers):
 *   c709_linear is BT.709-linear, DIFFUSE-WHITE-RELATIVE (white == 1.0) and 709-REPRESENTABLE
 *   (all components >= 0). Violating representability mirrors hues through abs(); violating
 *   the scale silently caps output at peak_ratio nits (the V1.1 "3.901 nits" incident).
 */
float3 PSYCHOV22_ToneMapBT709(
    float3 c709_linear,
    float  peak_value,          // white-relative (peak_nits / paper_white), >= 1
    float  compression,         // 0 => auto
    float  anchor_in,           // adapted state, 709-linear scalar (0.18)
    float  anchor_out,          // desired background, 709-linear scalar; caller clamps < peak
    float  gamut_strength,      // compression lerp
    int    gamut_mode)          // 0 = BT.709 bound, 1 = BT.2020 bound
{
    float3 lms_in         = mul(MAT_BT709_TO_LMS_SS, c709_linear);
    float3 adapt_lms      = mul(MAT_BT709_TO_LMS_SS, anchor_in.xxx);
    float3 bg_lms         = mul(MAT_BT709_TO_LMS_SS, anchor_out.xxx);
    float3 lms_peak       = mul(MAT_BT709_TO_LMS_SS, max(peak_value, 1.0).xxx);

    float3 anchor_in_s  = max(adapt_lms, PSYCHO_EPS.xxx);
    float3 anchor_out_s = max(bg_lms, PSYCHO_EPS.xxx);

    // Grades neutral: graded magnitude == |input| (Stage 1 owns all grading).
    float3 contrast_mag    = abs(lms_in);
    float3 response_source = PSY22_CopySign(contrast_mag, lms_in);

    // --- Compression exponent ---
    float h = compression;
    if (abs(compression) <= PSYCHO_EPS)
    {
        h = PSY22_AutoCompression(PSY22_YfFromLMS(anchor_out_s), PSY22_YfFromLMS(lms_peak));
    }
    h = max(h, 1e-6);
    float h_rcp = rcp(h);

    // --- Slope-normalized Naka-Rushton shoulder (anchor-exact, overflow-safe form) ---
    float3 anchor_over_peak = anchor_out_s / max(lms_peak, PSYCHO_EPS.xxx);
    float3 slope_norm = 1.0 - pow(max(anchor_over_peak, PSYCHO_EPS.xxx), h);
    float3 n = 1.0 / max(slope_norm, PSYCHO_EPS.xxx);

    float3 q_rcp    = pow(max(contrast_mag / anchor_in_s, PSYCHO_EPS.xxx), -(n * h));
    float3 shoulder = pow(max(lms_peak / anchor_out_s, PSYCHO_EPS.xxx), h) - 1.0;
    float3 fraction = 1.0 / (1.0 + shoulder * q_rcp);
    float3 saturated = lms_peak * pow(max(fraction, PSYCHO_EPS.xxx), h_rcp);

    float3 display_scaled = PSY22_CopySign(saturated, response_source);

    float3 disp_rel = PSY22_ToRelW(display_scaled, adapt_lms);
    float3 src_rel  = PSY22_ToRelW(response_source, adapt_lms);
    float3 neutral_w = PSY22_RelNeutral();
    float2 mb_neutral_xy = neutral_w.xz * rcp(max(neutral_w.x + neutral_w.y, 1e-6));

    // --- Intrinsic MB hue-direction restore (fixed 0.5 blend; radius + y carrier locked) ---
    {
        float2 src_off = PSY22_MBFromWLMS(src_rel).xy - mb_neutral_xy;
        float3 tgt_mb  = PSY22_MBFromWLMS(disp_rel);
        float2 tgt_off = tgt_mb.xy - mb_neutral_xy;
        float target_radius = length(tgt_off);
        float source_radius = length(src_off);

        if (target_radius > PSYCHO_EPS && source_radius > PSYCHO_EPS)
        {
            float2 blended = lerp(tgt_off, src_off * (target_radius / source_radius), 0.5);
            blended *= target_radius / max(length(blended), PSYCHO_EPS);
            disp_rel = PSY22_WLMSFromMB(float3(mb_neutral_xy + blended, tgt_mb.z));
        }
    }

    // --- Intrinsic M-cone crosstalk (gate-weighted; no strength parameter upstream) ---
    {
        float3 src_acc = PSY22_ACCFromRelDelta(src_rel - neutral_w) / max(neutral_w.x + neutral_w.y, 1e-6);
        float3 drive = abs(response_source) / max(adapt_lms, PSYCHO_EPS.xxx);
        float l_over_m = max(drive.x - drive.y, 0.0);
        float s_over_m = max(drive.z - drive.y, 0.0);
        float ls_mixed = PSY22_DivideSafe(min(l_over_m, s_over_m), max(l_over_m + s_over_m, 1e-6), 0.0);
        float cone_gate = smoothstep(0.02, 0.15, ls_mixed);
        float spectral_confidence = 1.0 - max(cone_gate, PSY22_PurpleGate(src_acc));

        float lm_share = PSY22_DivideSafe(drive.y, max(drive.x + drive.y, 1e-6), 0.0);
        float l_bias = saturate(l_over_m / max(drive.x, 1e-6)) * saturate(lm_share / 0.25);
        float s_bias = 0.15 * saturate(s_over_m / max(drive.z, 1e-6));
        float m_bias_weight = spectral_confidence * (0.12 * l_bias + 0.025 * s_bias);

        float3 tgt_mb = PSY22_MBFromWLMS(disp_rel);
        float2 tgt_off = tgt_mb.xy - mb_neutral_xy;
        float target_radius = length(tgt_off);
        float2 m_offset = -mb_neutral_xy;

        if (target_radius > PSYCHO_EPS && length(m_offset) > PSYCHO_EPS)
        {
            float2 bent = lerp(tgt_off, m_offset * (target_radius / length(m_offset)), saturate(m_bias_weight));
            bent *= target_radius / max(length(bent), PSYCHO_EPS);
            disp_rel = PSY22_WLMSFromMB(float3(mb_neutral_xy + bent, tgt_mb.z));
        }
    }

    // --- Device-hull gamut compression ---
    float3 lms_final = PSY22_FromRelW(disp_rel, adapt_lms);
    lms_final = PSY22_GamutCompress(lms_final, anchor_out_s, gamut_mode, gamut_strength, max(peak_value, 1.0));

    return mul(MAT_LMS_SS_TO_BT709, lms_final);
}

// =================================================================================================
// 8. Khronos PBR Neutral (single-source shoulder math)
// =================================================================================================

void ComputeKhronosParams(float3 safeColor, float targetPeak, float compressionStart,
                          out float offset, out float peak, out float startComp, out float d, out float newPeak)
{
    float x = min(safeColor.r, min(safeColor.g, safeColor.b));
    offset    = x < 0.08 ? x - 6.25 * x * x : 0.04;
    peak      = max(safeColor.r, max(safeColor.g, safeColor.b)) - offset;
    startComp = (targetPeak * compressionStart) - 0.04;
    d         = targetPeak - startComp;
    newPeak   = targetPeak - (d * d) / (peak + d - startComp);
}

float3 ApplyKhronosPBRNeutral(float3 color, float targetPeak, float compressionStart, float desatStrength)
{
    float3 safeColor = max(color, 0.0);
    float offset, peak, startComp, d, newPeak;
    ComputeKhronosParams(safeColor, targetPeak, compressionStart, offset, peak, startComp, d, newPeak);
    [branch]
    if (peak >= startComp && startComp > 0.0)
    {
        float3 working = color - offset;
        working *= newPeak / max(peak, FLT_MIN);
        float t = saturate((newPeak - startComp) / max(d, FLT_MIN));
        working = lerp(working, newPeak.xxx, desatStrength * t * t);
        return working + offset;
    }
    return color;
}

float ComputeCompressionRatio(float3 color, float targetPeak, float compressionStart)
{
    float3 safeColor = max(color, 0.0);
    float offset, peak, startComp, d, newPeak;
    ComputeKhronosParams(safeColor, targetPeak, compressionStart, offset, peak, startComp, d, newPeak);
    return (peak >= startComp && startComp > 0.0) ? newPeak / max(peak, FLT_MIN) : 1.0;
}

// =================================================================================================
// 9. Debug Visualization
// =================================================================================================

float3 EncodeDebug(float3 debug_out, int space)
{
    debug_out = max(debug_out, 0.0);
    [branch]
    if (space == 4)      return HLG_OETF(lerp(100.0, 600.0, saturate(debug_out)));
    else if (space == 3) return PQ_InverseEOTF(lerp(100.0, 600.0, saturate(debug_out)));
    else if (space == 2) return lerp(0.05, 2.5, saturate(debug_out)); // visible band in linear scRGB
    else                 return sRGB_OETF(saturate(debug_out));
}

int GetZone(float nl)
{
    if (nl < 0.0)       return 0;
    if (nl < ZONE_I)    return 1;
    if (nl < ZONE_II)   return 2;
    if (nl < ZONE_III)  return 3;
    if (nl < ZONE_IV)   return 4;
    if (nl < ZONE_V)    return 5;
    if (nl < ZONE_VI)   return 6;
    if (nl < ZONE_VII)  return 7;
    if (nl < ZONE_VIII) return 8;
    if (nl < ZONE_IX)   return 9;
    if (nl < ZONE_X)    return 10;
    if (nl < ZONE_XI)   return 11;
    return 12;
}

float3 GetZoneColor(int index)
{
    [flatten]
    switch (clamp(index, 0, 12))
    {
        case 0:  return float3(0.5, 0.0, 0.5);
        case 1:  return float3(0.02, 0.02, 0.05);
        case 2:  return float3(0.1, 0.0, 0.1);
        case 3:  return float3(0.2, 0.0, 0.3);
        case 4:  return float3(0.3, 0.0, 0.5);
        case 5:  return float3(0.2, 0.2, 0.8);
        case 6:  return float3(0.5, 0.5, 0.5);
        case 7:  return float3(0.8, 0.8, 0.2);
        case 8:  return float3(1.0, 0.8, 0.3);
        case 9:  return float3(1.0, 0.6, 0.4);
        case 10: return float3(1.0, 0.9, 0.8);
        case 11: return float3(1.0, 1.0, 1.0);
        case 12: return float3(1.0, 1.0, 0.5);
    }
    return float3(0.0, 0.0, 0.0);
}

float3 StopsToFalseColor(float stops)
{
    float t = saturate((stops + 8.0) / 16.0);
    if (t < 0.2)       return float3(0.0, 0.0, t / 0.2);
    else if (t < 0.4)  return float3(0.0, (t - 0.2) / 0.2, 1.0 - (t - 0.2) / 0.2);
    else if (t < 0.6)  return float3((t - 0.4) / 0.2, 1.0, 0.0);
    else if (t < 0.8)  return float3(1.0, 1.0 - (t - 0.6) / 0.2, 0.0);
    else               return float3(1.0, (t - 0.8) / 0.2, (t - 0.8) / 0.2);
}

float3 HueToRGB(float hue)
{
    return saturate(abs(frac(hue + float3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
}

float ComputeBleachingKLMS(float3 lms, float strength)
{
    if (strength <= NEUTRAL_EPS) return 1.0;
    float lm_sum = lms.r + lms.g;
    if (lm_sum <= 0.0) return 1.0;
    float3 stimulus = max(lms, 0.0) * TROLAND_LMS_SCALE;
    float stim_lm = 0.5 * (stimulus.r + stimulus.g);
    float availability = 1.0 / (1.0 + (stim_lm / max(TROLAND_HALF_SAT, FLT_MIN)));
    return lerp(1.0, availability, saturate(strength));
}

// =================================================================================================
// 10. Vertex Shader
// =================================================================================================

// V1.2: unused texcoord varying removed (fetch is SV_Position-based; saves an interpolator).
struct VS_Output
{
    float4 vpos : SV_Position;
    nointerpolation float3 wbScale : TEXCOORD1;
    nointerpolation float3 luma_LMS_coeffs : TEXCOORD2;
};

VS_Output VS_Master(uint id : SV_VertexID)
{
    VS_Output output;

    float2 quad = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    output.vpos = float4(quad * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

    int space = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;

    float3 lumaCoeffs;
    float3x3 to_RGB;
    if (space >= 3)
    {
        lumaCoeffs = Luma2020;
        to_RGB     = LMS_to_RGB2020;
    }
    else
    {
        lumaCoeffs = Luma709;
        to_RGB     = LMS_to_RGB709;
    }

    output.luma_LMS_coeffs = mul(lumaCoeffs, to_RGB);

    float3 wbStopsLMS = 0.35 * float3(fTemperature + fTint, -fTint, -fTemperature + fTint);
    float3 wbScaleLMS = exp2(wbStopsLMS);
    float lumaScale = dot(wbScaleLMS, output.luma_LMS_coeffs);
    float refLuma   = dot(float3(1.0, 1.0, 1.0), output.luma_LMS_coeffs);
    output.wbScale  = wbScaleLMS * (refLuma / max(lumaScale, FLT_MIN));

    return output;
}

// =================================================================================================
// 11. Pixel Shader - Master Pipeline
// =================================================================================================

void PS_Master(VS_Output input, out float4 fragColor : SV_Target)
{
    int2 pos   = int2(input.vpos.xy);
    float4 src = tex2Dfetch(SamplerBackBuffer, pos);

    int space         = (iColorSpaceOverride > 0) ? iColorSpaceOverride : BUFFER_COLOR_SPACE;
    float whitePt     = (space <= 1) ? SCRGB_WHITE_NITS : fWhitePoint;
    float3 lumaCoeffs = (space >= 3) ? Luma2020 : Luma709;

    // True bit-transparent bypass at factory defaults (TM off).
    [branch]
    if (iDebugMode == 0 &&
        abs(fExposure) < NEUTRAL_EPS && abs(fBlackPoint) < NEUTRAL_EPS &&
        abs(fContrast - 1.0) < NEUTRAL_EPS && abs(fShadows) < NEUTRAL_EPS &&
        abs(fHighlights) < NEUTRAL_EPS && abs(fTemperature) < NEUTRAL_EPS &&
        abs(fTint) < NEUTRAL_EPS && abs(fSaturation - 1.0) < NEUTRAL_EPS &&
        abs(fVibrance) < NEUTRAL_EPS && fBleaching < NEUTRAL_EPS &&
        fMConeCrosstalk < NEUTRAL_EPS && fAbneyCorrection < NEUTRAL_EPS &&
        fGamutGuardKnee < NEUTRAL_EPS && iGamutTarget == 0 &&
        iToneMapperMode == 0)
    {
        fragColor = src;
        return;
    }

    float3 original_lin = DecodeToLinear(src.rgb, space);
    bool is_invalid = any(IsNan3(original_lin)) || any(IsInf3(original_lin));
    original_lin = is_invalid ? (0.18 * whitePt).xxx : original_lin;

    float3x3 to_LMS, to_RGB, to_RGB_boundary;
    [branch]
    if (space >= 3)
    {
        to_LMS = RGB2020_to_LMS;  to_RGB = LMS_to_RGB2020;  to_RGB_boundary = LMS_to_RGB2020;
    }
    else if (space == 2)
    {
        to_LMS = RGB709_to_LMS;   to_RGB = LMS_to_RGB709;
        // scRGB policy: open-ended container; guard chromaticity against the Rec.2020 volume.
        to_RGB_boundary = LMS_to_RGB2020;
    }
    else
    {
        to_LMS = RGB709_to_LMS;   to_RGB = LMS_to_RGB709;   to_RGB_boundary = LMS_to_RGB709;
    }
    [branch]
    if (iGamutTarget == 1)      to_RGB_boundary = LMS_to_RGB709;
    else if (iGamutTarget == 2) to_RGB_boundary = LMS_to_P3D65;
    else if (iGamutTarget == 3) to_RGB_boundary = LMS_to_RGB2020;

    float2 mb_white = MB_WHITE_D65;

    // =============================================================================================
    // STAGE 1: PHOTOREAL GRADING (native HPE basis)
    // =============================================================================================
    float3 lms = mul(to_LMS, original_lin);

    // 1a. White balance + exposure
    lms *= input.wbScale;
    if (abs(fExposure) > NEUTRAL_EPS) lms *= exp2(fExposure);

    // 1b. Dehaze + filmic contrast (negative-luma scRGB excursions bypass untouched)
    float3 lms_pre_grading = lms;
    float luma = dot(lms_pre_grading, input.luma_LMS_coeffs);

    float bp_ratio = 1.0;
    if (fBlackPoint > NEUTRAL_EPS && luma > 0.0)
    {
        bp_ratio = ComputeBlackPointRatio(luma, fBlackPoint * whitePt, fShadowFloor);
    }

    float contrast_ratio = 1.0;
    float graded_luma = max(luma * bp_ratio, FLT_MIN);
    [branch]
    if (graded_luma > FLT_MIN && luma > 0.0)
    {
        float pivot = fContrastPivot * whitePt;
        float x = log2(graded_luma / pivot) * fContrast;
        float S = fShadows * 3.0;
        float H = fHighlights * 3.0;
        float rational_factor = (x * x) / (x * x + 6.0);
        float recovery = lerp(S, H, saturate(0.5 + x * 4.0));
        x += recovery * rational_factor;
        float ratio = (pivot * exp2(x)) / graded_luma;
        // Asymptotic soft limiter: monotonic, converges to x100 (documented honestly).
        float excess = max(ratio - 80.0, 0.0);
        contrast_ratio = min(ratio, 80.0) + (excess / (1.0 + excess / 20.0));
    }
    lms *= bp_ratio * contrast_ratio;

    // 1c. Troland bleaching (REINSTATED V1.2; lms_pre_bleach feeds debug modes 3 and Stage-3 crosstalk)
    float3 lms_pre_bleach = lms;
    lms = ApplyTrolandBleachingLMS(lms, fBleaching, input.luma_LMS_coeffs);

    // =============================================================================================
    // STAGE 2: TONE MAPPING DISPATCH
    // Both operators share the negative-luma bypass policy (extended-range scRGB WCG pixels are
    // routed around ALL tonemapping untouched).
    // =============================================================================================
    float3 color = float3(0.0, 0.0, 0.0);
    float3 pre_khronos_color = float3(0.0, 0.0, 0.0);
    float tone_comp_ratio = 1.0;
    float targetPeak = (space <= 1) ? 1.0 : max(1.0, fDisplayPeakNits / whitePt);
    float pix_luma = dot(lms, input.luma_LMS_coeffs);

    if (iToneMapperMode == 1) // Khronos PBR Neutral (container basis, as in V6.1.6)
    {
        [branch]
        if (pix_luma > 0.0)
        {
            color = mul(to_RGB, lms);
            pre_khronos_color = color;
            color /= max(whitePt, FLT_MIN);
            color = ApplyKhronosPBRNeutral(color, targetPeak, fCompressionStart, fDesaturationStrength);
            color *= whitePt;
            lms = mul(to_LMS, color);
        }
    }
    else if (iToneMapperMode == 2) // RenoDX PsychoV-22
    {
        [branch]
        if (pix_luma > 0.0)
        {
            // ---------------------------------------------------------------------------------
            // UNIT SEAM (see incident history before touching):
            //   IN: nits -> convert primaries -> NORMALIZE by diffuse white -> white-relative
            //   709 -> enforce 709 representability -> PSYCHOV22_ToneMapBT709 (white-relative)
            //   -> denormalize -> OUT: nits.
            // ---------------------------------------------------------------------------------
            float3 rgb_container = mul(to_RGB, lms);

            float3 c709_nits;
            if (space >= 3) c709_nits = mul(MAT_BT2020_TO_BT709, rgb_container);
            else            c709_nits = rgb_container;

            float paper_scale = max(whitePt, FLT_MIN);
            float3 c709_rel = c709_nits / paper_scale;

            // V1.2 FIX (hue mirroring): the primary conversion produces NEGATIVE components for
            // any 2020-in-gamut color outside 709. The observer model folds magnitudes through
            // abs(), which would reflect those chromaticities about the neutral axis (2020
            // green entering as magenta). Enforce the 709-representability contract with a
            // closed-form luminance-preserving pull toward achromatic: smallest blend t such
            // that every component is >= 0. Binding channel is the most-negative one:
            //   t >= c_min / (c_min - L),  valid because L > 0 > c_min here.
            float min_c = min(c709_rel.r, min(c709_rel.g, c709_rel.b));
            [branch]
            if (min_c < 0.0)
            {
                float lum709 = dot(c709_rel, Luma709);
                float t_fix = saturate(min_c / min(min_c - lum709, -FLT_MIN));
                c709_rel = lerp(c709_rel, lum709.xxx, t_fix);
            }
            // NOTE: residual chroma compression here is an upstream architectural limit -
            // PsychoV-22's domain is 709-representable signals. Full WCG passthrough would
            // require an upstream change, not a seam patch.

            float peak_white_rel;
            if (space <= 1) peak_white_rel = 1.0;   // SDR: ceiling == diffuse white
            else            peak_white_rel = max(fDisplayPeakNits / fWhitePoint, 1.0);

            // V1.2 GUARD (degenerate shoulder): anchor_out >= peak collapses the NR curve
            // (shoulder == 0 => everything saturates to peak). Clamp the anchor below peak.
            float anchor_out_v = min(fPsychoAnchorOut, 0.98 * peak_white_rel);

            int gmode;
            if (iPsychoGamutMode == 0) gmode = (space >= 3) ? 1 : 0;
            else                       gmode = iPsychoGamutMode - 1;

            float3 tm709_rel = PSYCHOV22_ToneMapBT709(
                c709_rel, peak_white_rel, fPsychoCompression,
                fPsychoAnchorIn, anchor_out_v,
                fPsychoGamutStrength, gmode);

            float3 rgb_back_rel;
            if (space >= 3) rgb_back_rel = mul(MAT_BT709_TO_BT2020, tm709_rel);
            else            rgb_back_rel = tm709_rel;

            float3 rgb_back_nits = rgb_back_rel * paper_scale;
            float3 lms_after = mul(to_LMS, rgb_back_nits);

            tone_comp_ratio = dot(lms_after, input.luma_LMS_coeffs) / max(pix_luma, FLT_MIN);
            lms = lms_after;
        }
    }

    // =============================================================================================
    // STAGE 3: APPEARANCE (Photoreal basis, post-TM, as in V6.1.6)
    // Hue-restore control removed: PsychoV restores hue intrinsically; Abney covers Stage 1 needs.
    // =============================================================================================
    lms = ApplyMConeCrosstalkLMS(lms, fMConeCrosstalk, lms_pre_bleach, input.luma_LMS_coeffs, mb_white, whitePt);

    lms = ApplyMBPurityAndGamutGuardLMS(
        lms, fSaturation, fVibrance, fSkinProtection, iGamutTarget, fGamutGuardKnee,
        fAbneyCorrection, input.luma_LMS_coeffs, to_RGB_boundary, mb_white, whitePt);

    // =============================================================================================
    // STAGE 4: RECONSTRUCT, DEBUG, ENCODE
    // =============================================================================================
    color = mul(to_RGB, lms);
    is_invalid = any(IsNan3(color)) || any(IsInf3(color));
    color = is_invalid ? original_lin : color;

    [branch]
    if (iDebugMode != 0)
    {
        float3 debug_out = float3(0.0, 0.0, 0.0);

        if (iDebugMode == 1)
        {
            float l = dot(color, lumaCoeffs);
            debug_out = StopsToFalseColor(log2(max(abs(l), FLT_MIN) / max(whitePt, FLT_MIN)));
        }
        else if (iDebugMode == 2)
        {
            debug_out = GetZoneColor(GetZone(dot(color, lumaCoeffs) / max(whitePt, FLT_MIN)));
        }
        else if (iDebugMode == 3)
        {
            // V1.2: wired to the live fBleaching control again (was permanently identity/blue).
            float k = ComputeBleachingKLMS(lms_pre_bleach, fBleaching);
            debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(k));
        }
        else if (iDebugMode == 4)
        {
            float3 lms_dbg = mul(to_LMS, color);
            if (lms_dbg.r + lms_dbg.g > 0.0)
            {
                float2 off = LMS_to_MB(lms_dbg).xy - mb_white;
                float v = saturate(SqrtIEEE(dot(off, off)) * 3.0);
                debug_out = float3(v, v * 0.7, v * 0.3);
            }
        }
        else if (iDebugMode == 5)
        {
            float3 lms_dbg = mul(to_LMS, color);
            if (lms_dbg.r + lms_dbg.g > 0.0)
            {
                float2 off = LMS_to_MB(lms_dbg).xy - mb_white;
                float psq = dot(off, off);
                if (psq > 1e-12)
                {
                    float hue = atan2(off.y, off.x) / (2.0 * PI) + 0.5;
                    debug_out = HueToRGB(saturate(hue)) * saturate(SqrtIEEE(psq) * 5.0);
                }
            }
        }
        else if (iDebugMode == 6)
        {
            float3 lms_dbg = mul(to_LMS, color);
            float mx = max(max(abs(lms_dbg.r), abs(lms_dbg.g)), abs(lms_dbg.b));
            if (mx > FLT_MIN) debug_out = abs(lms_dbg) / mx;
        }
        else if (iDebugMode == 7)
        {
            if (any(IsNan3(color)) || any(IsInf3(color)))
            {
                debug_out = float3(1.0, 1.0, 1.0);
            }
            else
            {
                float3 neg = float3(color.r < 0.0 ? 1.0 : 0.0, color.g < 0.0 ? 1.0 : 0.0, color.b < 0.0 ? 1.0 : 0.0);
                debug_out = (neg.r + neg.g + neg.b > 0.0) ? neg : float3(0.0, 0.15, 0.0);
            }
        }
        else if (iDebugMode == 8)
        {
            if (iToneMapperMode == 1)
            {
                float ratio = ComputeCompressionRatio(pre_khronos_color / max(whitePt, FLT_MIN), targetPeak, fCompressionStart);
                debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(ratio));
            }
            else if (iToneMapperMode == 2)
            {
                debug_out = lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.3, 1.0), saturate(tone_comp_ratio));
            }
            else
            {
                debug_out = float3(0.2, 0.2, 0.2);
            }
        }
        else if (iDebugMode == 9)
        {
            float3 lms_dbg = mul(to_LMS, color);
            if (lms_dbg.r + lms_dbg.g > FLT_MIN)
            {
                float mask = Evaluate3DSkinLocusMB(dot(lms_dbg, input.luma_LMS_coeffs) / max(whitePt, FLT_MIN), LMS_to_MB(lms_dbg));
                debug_out = lerp(float3(0.0, 0.1, 0.3), float3(1.0, 0.2, 0.8), mask);
            }
        }

        fragColor = float4(EncodeDebug(debug_out, space), src.a);
        return;
    }

    float3 encoded = EncodeFromLinear(color, space);
    [flatten]
    if (space <= 1) encoded = saturate(encoded);
    fragColor = float4(encoded, src.a);
}

// =================================================================================================
// 12. Technique
// =================================================================================================

technique Photoreal_PsychoV22_V12 <
    ui_label = "Photoreal HDR + RenoDX PsychoV-22 V1.2 (Seam Integrity Edition)";
    ui_tooltip = "Stage 1: Photoreal grading in native Ottoson-HPE MacLeod-Boynton space.\n"
                 "Stage 2: RenoDX PsychoV-22 observer cascade + device-hull mapping in\n"
                 "Stockman-Sharpe weighted LMS, composed at the 709-linear boundary.\n\n"
                 "V1.2:\n"
                 "  - FIX: WCG hues no longer mirror through the PsychoV handoff (HDR10/HLG).\n"
                 "  - GUARD: Anchor Out clamped below peak (degenerate-shoulder protection).\n"
                 "  - POLICY: PsychoV Gamut Strength defaults to 0.50 - keep >= 0.5 with any TM on.\n"
                 "  - RESTORED: Stage 1 Troland bleaching + its debug visualizer.\n\n"
                 "Standing guarantees:\n"
                 "  - Factory defaults (TM off) are a bit-transparent bypass.\n"
                 "  - Zero shared constants across the two color-science domains.\n"
                 "  - PsychoV hue restore / M-crosstalk intrinsic (faithful to upstream).\n"
                 "  - SDR tone-map ceiling == diffuse white; scRGB negatives bypass all stages.\n\n"
                 "Attribution: PsychoV-22 core adapted from RenoDX (C) Carlos Lopez, MIT.\n"
                 "Gamut-bound helper is a documented V1.1 reconstruction (see source notice).";
>
{
    pass
    {
        VertexShader      = VS_Master;
        PixelShader       = PS_Master;
        VertexCount       = 3;
        PrimitiveTopology = TRIANGLELIST;
    }
}