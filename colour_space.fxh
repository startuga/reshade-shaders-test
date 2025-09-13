/**
 * CORRECTED colour_space.fxh
 * 
 * Fixes critical scaling errors in HDR luminance representation:
 * 1. Proper absolute luminance scaling for all color spaces
 * 2. Correct luminance coefficient implementation
 * 3. Physically accurate transfer function conversions
 * 
 * Version: 2.1 (Physical Accuracy Fix)
 */

// ==============================================================================
// Color Space Definitions
// ==============================================================================

// Color space enumeration - DO NOT CHANGE THESE VALUES
// They're referenced by shaders that use this header
#define CSP_SRGB   0  // Standard SDR (sRGB, typically 80-100 nits)
#define CSP_SCRGB  1  // Extended SDR (scRGB, typically 80-100 nits)
#define CSP_HDR10  2  // HDR10 (PQ transfer function, 0.005-10,000 nits)
#define CSP_HLG    3  // HLG (Hybrid Log-Gamma, 0.001-1000+ nits)

// ==============================================================================
// Color Space Conversion Matrices
// ==============================================================================

namespace Csp {
    namespace Mat {
        // BT.709 to BT.2020 conversion matrix
        static const float3x3 Bt709To::Bt2020 = {
            float3(0.627404, 0.329056, 0.043540),
            float3(0.069097, 0.919685, 0.011218),
            float3(0.016392, 0.088122, 0.895486)
        };

        // BT.2020 to BT.709 conversion matrix
        static const float3x3 Bt2020To::Bt709 = {
            float3(1.660710, -0.587551, -0.073159),
            float3(-0.124542,  1.132903, -0.008361),
            float3(-0.018152, -0.100587,  1.118739)
        };

        // BT.2020 to XYZ conversion matrix (D65 white point)
        static const float3x3 Bt2020ToXYZ = {
            float3(0.636958, 0.262700, 0.000072),
            float3(0.144617, 0.678000, 0.072194),
            float3(0.165567, 0.059300, 0.991050)
        };

        // XYZ to BT.2020 conversion matrix
        static const float3x3 XYZTo::Bt2020 = {
            float3( 1.717269, -0.666079, -0.012956),
            float3(-0.371473,  1.331091, -0.003194),
            float3(-0.270796,  0.026543,  1.016150)
        };
    }
}

// ==============================================================================
// LUMINANCE COEFFICIENTS (CRITICAL FIX)
// ==============================================================================

// Correct luminance coefficients for different color spaces
static const float3 Rec709LumaCoefficients = float3(0.2126, 0.7152, 0.0722);
static const float3 Rec2020LumaCoefficients = float3(0.2627, 0.6780, 0.0593);
static const float3 DCI_P3LumaCoefficients = float3(0.2280, 0.7075, 0.0645);

// ==============================================================================
// TRANSFER FUNCTIONS (CRITICAL FIX)
// ==============================================================================

namespace Csp {
    namespace Trc {
        // SDR transfer functions (gamma 2.2 approximation)
        float3 LinearTo::SDR(float3 linear) {
            return pow(linear, 1.0/2.2);
        }
        
        float3 SDRTo::Linear(float3 sdr) {
            return pow(sdr, 2.2);
        }

        // HDR10 transfer functions (SMPTE ST.2084)
        float ST2084_EOTF(float m) {
            const float c1 = 0.8359375;
            const float c2 = 18.8515625;
            const float c3 = 18.6875;
            const float m1 = 0.1593017578125;
            const float m2 = 78.84375;
            
            float n = pow(m, 1.0/m2);
            return pow(max(n - c1, 0.0) / (c2 - c3*n), m1);
        }
        
        float ST2084_OETF(float n) {
            const float c1 = 0.8359375;
            const float c2 = 18.8515625;
            const float c3 = 18.6875;
            const float m1 = 0.1593017578125;
            const float m2 = 78.84375;
            
            float n_pow = pow(n, m1);
            return pow((c1 + c2*n_pow) / (1.0 + c3*n_pow), m2);
        }
        
        float3 LinearTo::Pq(float3 linear) {
            // CORRECTED: 1.0 linear = 10,000 nits for HDR10
            return float3(
                ST2084_EOTF(linear.r * 10000.0),
                ST2084_EOTF(linear.g * 10000.0),
                ST2084_EOTF(linear.b * 10000.0)
            );
        }
        
        float3 PqTo::Linear(float3 pq) {
            // CORRECTED: output values where 1.0 = 10,000 nits
            return float3(
                ST2084_OETF(pq.r) / 10000.0,
                ST2084_OETF(pq.g) / 10000.0,
                ST2084_OETF(pq.b) / 10000.0
            );
        }

        // HLG transfer functions
        float Hlg_OETF(float linear) {
            const float a = 0.17883277;
            const float b = 0.28466892;
            const float c = 0.55991073;
            
            if (linear <= 0.0) {
                return 0.0;
            } else if (linear <= 1.0/12.0) {
                return sqrt(3.0 * linear);
            } else {
                return a * log(12.0 * linear - b) + c;
            }
        }
        
        float Hlg_EOTF(float hlg) {
            const float a = 0.17883277;
            const float b = 0.28466892;
            const float c = 0.55991073;
            
            if (hlg <= 0.0) {
                return 0.0;
            } else if (hlg <= 0.5) {
                return (hlg * hlg) / 3.0;
            } else {
                return (exp((hlg - c) / a) + b) / 12.0;
            }
        }
        
        float3 LinearTo::Hlg(float3 linear) {
            // CORRECTED: 1.0 linear = 1,000 nits for HLG
            return float3(
                Hlg_OETF(linear.r * 1000.0),
                Hlg_OETF(linear.g * 1000.0),
                Hlg_OETF(linear.b * 1000.0)
            );
        }
        
        float3 HlgTo::Linear(float3 hlg) {
            // CORRECTED: output values where 1.0 = 1,000 nits
            return float3(
                Hlg_EOTF(hlg.r) / 1000.0,
                Hlg_EOTF(hlg.g) / 1000.0,
                Hlg_EOTF(hlg.b) / 1000.0
            );
        }
    }
}

// ==============================================================================
// UTILITY FUNCTIONS (CORRECTED)
// ==============================================================================

// Get reference white in nits for the current color space
float GetReferenceWhiteNits() {
    return 80.0; // Standard reference white (80 nits) for all color spaces
}

// Get luminance scale factor (nits per linear unit)
float GetLuminanceScale() {
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        return 80.0;  // 1.0 linear = 80 nits for SDR
    #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        return 10000.0;  // 1.0 linear = 10,000 nits for HDR10
    #elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
        return 1000.0;   // 1.0 linear = 1,000 nits for HLG
    #else
        return 80.0;     // Default to SDR reference
    #endif
}

// Calculate luminance using correct coefficients for the color space
float CalculateLuminance(float3 color, int colour_space = ACTUAL_COLOUR_SPACE) {
    #if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        return dot(color, Rec709LumaCoefficients);
    #else
        // For HDR content, use Rec. 2020 coefficients by default
        // (could be extended for DCI-P3 detection)
        return dot(color, Rec2020LumaCoefficients);
    #endif
}

// Convert linear luminance to log2 ratio space (exposure stops)
float LinearToLog2Ratio(float linear_luma) {
    // Convert to absolute nits
    float absolute_luma_nits = linear_luma * GetLuminanceScale();
    
    // Calculate ratio relative to reference white
    float ratio = absolute_luma_nits / GetReferenceWhiteNits();
    
    // Return log2 of the ratio (exposure stops from reference)
    return log2(max(ratio, 1e-6));
}

// Convert log2 ratio back to linear luminance
float Log2RatioToLinear(float log2_ratio) {
    // Calculate absolute luminance in nits
    float absolute_luma_nits = GetReferenceWhiteNits() * exp2(log2_ratio);
    
    // Convert back to linear RGB value for this color space
    return absolute_luma_nits / GetLuminanceScale();
}