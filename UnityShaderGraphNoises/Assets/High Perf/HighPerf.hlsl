//UNITY_SHADER_NO_UPGRADE
#ifndef MYHLSLINCLUDE_INCLUDED
#define MYHLSLINCLUDE_INCLUDED

// ---------------------------------------------------------------------
// With only 3 texture fetches, generates endless seamless non-repeating 
// same-properties texture from example. ( in the paper: +1 LUT fetch
// to handle non-Gaussian correlated color histograms).

// Simple implementation of our HPG'18 
// "High-Performance By-Example Noise using a Histogram-Preserving Blending Operator"
// https://hal.inria.fr/hal-01824773
// ( color-histogram Gaussianisation not possible in shadertoy ;-) 
//   or possibly via this approx: https://www.shadertoy.com/view/slX3zr  ). 
// ---------------------------------------------------------------------

#define CON 1      // contrast preserving interpolation. cf https://www.shadertoy.com/view/4dcSDr
#define Z   8.     // patch scale inside example texture

#define rnd22(p)    frac(sin(mul((p) , float2x2(127.1,311.7,269.5,183.3) ))*43758.5453)
#define srgb2rgb(V) pow( max(V,0.), float4( 2.2 )  )          // RGB <-> sRGB conversions
#define rgb2srgb(V) pow( max(V,0.), float4(1./2.2) )

// --- texture patch associated to vertex I in triangular/hexagonal grid. key 'P'
// (textureGrad handles MIPmap through patch borders)
#define C(I)  ( iChannel0.SampleGrad(Sampler, U/Z-rnd22(I) ,Gx,Gy) - m*float(CON) )
// --- for tests
#define S(v)       smoothstep( p,-p, v )                    // antialiased drawing
#define hue(v)   ( .6 + .6 * cos( v  + float4(0,23,21,0)  ) ) // from https://www.shadertoy.com/view/ll2cDc
#define H(I)       hue( (I).x + 71.3*(I).y )
//#define keyToggle(c) ( texelFetch(iChannel3,int2(64+c,2),0).x > 0.) // keyboard. from https://www.shadertoy.com/view/llySRh

float2x2 inverse(float2x2 A)
{
    float2x2 C;

    float det = determinant(A);
    C[0][0] = A._m11;
    C[1][0] = -A._m01;
    C[0][1] = -A._m10;
    C[1][1] = A._m00;

    return C / det;
}

void MyFunction_float(float2 u, float iMouse, float2 iTime, UnityTexture2D iChannel0, UnitySamplerState Sampler, out float4 _out)
{
    //float2 iTime = 1.0f;
    float2x2 M0 = float2x2(1, 0, .5, sqrt(3.) / 2.),
        M = inverse(M0);                           // transform matrix <-> tilted space
    float2 R = float2(800,800),
        z = iMouse,
        U = u*z + float2(2. * iTime),
        V = mul(M, U),                                    // pre-hexa tilted coordinates
        I = floor(V),                                 // hexa-tile id
        P = floor(mul(M, float2(2. * iTime)));              // center tile (to show patches)
    float p = .7 * ddy(U.y);                            // pixel size (for antialiasing)
    float2 Gx = ddx(U / Z), Gy = ddy(U / Z);               // (for cross-borders MIPmap)
    float4 m = iChannel0.SampleBias(Sampler, U, 99.);     // mean texture color // vec4 m = srgb2rgb( texture(iChannel0,U,99.) );

    float3 F = float3(frac(V), 0), A, W; F.z = 1. - F.x - F.y; // local hexa coordinates
    if (F.z > 0.)
        _out = (W.x = F.z) * C(I)                      // smart interpolation
        + (W.y = F.y) * C(I + float2(0, 1))            // of hexagonal texture patch
        + (W.z = F.x) * C(I + float2(1, 0));           // centered at vertex
    else                                               // ( = random offset in texture )
        _out = (W.x = -F.z) * C(I + 1.)
        + (W.y = 1. - F.y) * C(I + float2(1, 0))
        + (W.z = 1. - F.x) * C(I + float2(0, 1));
#if CON    
    _out = m + _out / length(W);  // contrast preserving interp. cf https://www.shadertoy.com/view/4dcSDr
#endif
    _out = clamp(_out, 0., 1.);
    if (m.g == 0.) _out = _out.rrrr;                           // handles B&W (i.e. "red") textures

    //if (keyToggle(7)) _out = mix(_out, float4(1), S(min(W.x, min(W.y, W.z)) - p)); // key 'G'; show grid  

	//_out = float4(1.0f, 1.0f, 1.0f, 1.0f);
}

#endif //MYHLSLINCLUDE_INCLUDED