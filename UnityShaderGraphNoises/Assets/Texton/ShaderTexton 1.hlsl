//UNITY_SHADER_NO_UPGRADE
#ifndef MYHLSLINCLUDE_INCLUDED
#define MYHLSLINCLUDE_INCLUDED

const float pi = 3.14159265358979323846f;

/*
 * From http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
 * Same strategy as in Gabor noise by example
 * Apply hashtable to create cellseed
 * Use a linear congruential generator as fast PRNG
 */
uint wang_hash(uint seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 668265261u;
    seed = seed ^ (seed >> 15u);
    return(seed);
}

uint cellseed(const in int2 c, const in uint offset)
{
    const uint period = 1024u;
    uint s = ((uint(c.y) % period) * period + (uint(c.x) % period)) * period + offset;
    if (s == 0u) s = 1u;
    return(s);
}

struct noise_prng
{
    uint state;
};

void mysrand(inout noise_prng p, const in uint seed)
{
    uint s = seed;
    p.state = wang_hash(s);
}

uint myrand(inout noise_prng p)
{
    // linear congruential generator: procudes correlated output. Similar patterns are visible
    // p.state = 1664525u * p.state + 1013904223u;
    // Xorshift algorithm from George Marsaglia's paper
    p.state ^= (p.state << 13u);
    p.state ^= (p.state >> 17u);
    p.state ^= (p.state << 5u);
    return(p.state);
}


float myrand_uniform_0_1(inout noise_prng p)
{
    return(((float)myrand(p)) / ((float)4294967295u));
}

uint myrand_uniform_integer(inout noise_prng p, const in uint N)
{
    /* return a random integer among uniformly distributed among 0,1,...,N-1 */
    return(uint(floor(N * myrand_uniform_0_1(p))));
}

float myrand_gaussian_0_1(inout noise_prng p)
{
    /* Box-Muller method for generating standard Gaussian variate */
    float u = myrand_uniform_0_1(p);
    float v = myrand_uniform_0_1(p);
    return(sqrt(-2.0 * log(u)) * cos(2.0 * pi * v));
}

uint my_rand_poisson(inout noise_prng p, const in float lmbd)
{
    /* This is a crude approximation of Poisson distribution that should be used only for large lmbd */
    return (uint(floor(lmbd + 0.5 + (sqrt(lmbd) * myrand_gaussian_0_1(p)))));
}

void MyFunction_float(float3 x_tex, int noiseOff, float3 noiseMean, UnityTexture2D textonTexture, float2 textonTextureSize, float scale, float2 half_window_size, uint offset, float meannbofimpacts, UnitySamplerState mySampler, float3 sumTexton, float fadeImage, float addImg, float newadd, out float4 gl_FragColor)
{
    if (noiseOff == 1u)
    {
        //gl_FragColor = float4(noiseMean, 1.0f);
        gl_FragColor.xyz = noiseMean;
        return;
    }

    //float2 textonTextureSize = textureSize(textonTexture, 0);
    float r = textonTextureSize.x;

    float nsubdiv = 1.6f;

    float2 xoverr = scale * half_window_size / r * (x_tex.xy + float2(1.0f, -1.0f));
    /* provide gradient for texture fetching */
    float2 dx = scale * half_window_size / r * ddx(x_tex.xy);
    float2 dy = scale * half_window_size / r * ddy(x_tex.xy);


    float2 vx = float2(0.5f, 0.5f);
    int2 mink = int2(floor(nsubdiv * (xoverr - vx)));
    int2 maxk = int2(floor(nsubdiv * (xoverr + vx)));

    /* Initialize fragment color with (0,0,0,0) */
    gl_FragColor = float4(0.0f, 0.0f, 0.0f, 0.0f);

    float3 minT = float3(-3.2468e-03, -3.5128e-03, -3.8760e-03);
    float3 maxT = float3(2.9405e-03, 3.5723e-03, 3.0620e-03);

    float lambda = (meannbofimpacts * fadeImage * sqrt(sqrt(sqrt(meannbofimpacts))) / (r * r));

    /* Simulate Poisson process on the 4 neighborhood cells of (x,y) */
    for (int ncx = mink.x; ncx <= maxk.x; ncx++) /* x-cell number */
    {
        for (int ncy = mink.y; ncy <= maxk.y; ncy++) /* y-cell number */
        {
            /* seed cell = (x/w, y/w) */
            uint seed = cellseed(int2(ncx, ncy), offset);
            noise_prng p;
            mysrand(p, seed);
            /* Draw number of point in the cell */
            uint Ncell = my_rand_poisson(p, meannbofimpacts / (nsubdiv * nsubdiv));
            /* Draw the points of the cell */
            for (uint i = 0u; i < Ncell; i++)
            {
                float2 uv = xoverr + float2(0.5f - (ncx + myrand_uniform_0_1(p)) / nsubdiv, 0.5f - (ncy + myrand_uniform_0_1(p)) / nsubdiv);
                if ((uv.x >= 0) && (uv.x <= 1) && (uv.y >= 0) && (uv.y <= 1))
                {
                    //gl_FragColor += texture2DGrad(textonTexture, uv, dx, dy);
                    float3 textonTemp = textonTexture.SampleGrad(mySampler, uv, dx, dy);
                    //gl_FragColor.xyz += float3(textonTemp.x*(abs(minT.x)+maxT.x)-abs(minT.x) + (maxT.x- minT.x)/addImg, textonTemp.y * (abs(minT.y) + maxT.y) - abs(minT.y)+ (maxT.y - minT.y)/addImg, textonTemp.z * (abs(minT.z) + maxT.z) - abs(minT.z)+ (maxT.z - minT.z)/addImg);
                    gl_FragColor.xyz += float3(textonTemp.x * (abs(minT.x) + maxT.x) - abs(minT.x) + (maxT.x - minT.x) * (addImg+lambda), textonTemp.y * (abs(minT.y) + maxT.y) - abs(minT.y) + (maxT.y - minT.y) * (addImg + lambda), textonTemp.z * (abs(minT.z) + maxT.z) - abs(minT.z) + (maxT.z - minT.z) * (addImg + lambda));
                }
            }
        }
    }
    //float3 textonTemp = textonTexture.SampleGrad(mySampler, x_tex, dx, dy);
    //gl_FragColor = textonTexture.SampleGrad(mySampler, x_tex, dx, dy);
    //gl_FragColor.xyz = float3(textonTemp.x * (abs(minT.x) + maxT.x) - abs(minT.x), textonTemp.y * (abs(minT.y) + maxT.y) - abs(minT.y), textonTemp.z * (abs(minT.z) + maxT.z) - abs(minT.z));
    //gl_FragColor.xyz = float3((gl_FragColor.x + abs(minT.x)) / (abs(minT.x) + maxT.x), (gl_FragColor.y + abs(minT.y)) / (abs(minT.y) + maxT.y), (gl_FragColor.z + abs(minT.z)) / (abs(minT.z) + maxT.z));
    //gl_FragColor = gl_FragColor + 0.5f;
    /* normalize and add mean */
    lambda = (meannbofimpacts * fadeImage / (r * r));
    gl_FragColor.xyz = 1.0f / sqrt(lambda) * (gl_FragColor.xyz - lambda * sumTexton) + noiseMean;
    //gl_FragColor.xyz = 1.0f / sqrt(lambda) * gl_FragColor.xyz + noiseMean;
    gl_FragColor.w = 1.0f;
    //gl_FragColor.xyz = fadeImage * textonTexture.SampleGrad(mySampler, x_tex.xy, dx, dy)+ addImg;
    //gl_FragColor = float4(x_tex.x, x_tex.y, 0., 1.);
}

#endif //MYHLSLINCLUDE_INCLUDED