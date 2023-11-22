//UNITY_SHADER_NO_UPGRADE
#ifndef MYHLSLINCLUDE_INCLUDED
#define MYHLSLINCLUDE_INCLUDED

#define getValue(p,v) textonTexture.SampleLevel(mySampler, float2(v + 0.5, p - 0.5) / r, 0).a
#define logit(v,alpha) log(1/v - 1)/-alpha

static const float pi = 3.14159265358979323846f;

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
    return((float(myrand(p))) / (float(4294967295u)));
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



void MyFunction_float(float3 x_tex, int noiseOff, UnityTexture2D textonTexture, float scale, uint offset, float meannbofimpacts, UnitySamplerState mySampler, out float4 gl_FragColor)
{
    float r = _textonTexture_TexelSize.z;//textonTextureSize.x;

    float3 noiseMean = float3(getValue(0.,0.), getValue(0., 1.), getValue(0., 2.));
    float3 sumTexton = float3(logit(getValue(-1., 0.), 1.5), logit(getValue(-1., 1.), 1.5), logit(getValue(-1., 2.), 1.5));

    if (noiseOff == 1u)
    {
        //gl_FragColor = float4(noiseMean, 1.0f);
        gl_FragColor.xyz = noiseMean;
        return;
    }

    x_tex.xy = float3(x_tex.xy-float2(0.5f, 0.5f), 0.)*2.f;
    //x_tex.xy = float3(newadd * ((x_tex.xy * x_tex.xy * x_tex.xy * x_tex.xy) + addImg), 0.f);

    //float2 textonTextureSize = textureSize(textonTexture, 0);
    

    float nsubdiv = 1.6f;

    float2 xoverr = scale / r * x_tex.xy;
    /* provide gradient for texture fetching */
    float2 dx = scale / r * ddx(x_tex.xy);
    float2 dy = scale / r * ddy(x_tex.xy);


    float2 vx = float2(0.5f, 0.5f);
    int2 mink = int2(floor(nsubdiv * (xoverr - vx)));
    int2 maxk = int2(floor(nsubdiv * (xoverr + vx)));

    /* Initialize fragment color with (0,0,0,0) */
    gl_FragColor = float4(0.0f, 0.0f, 0.0f, 0.0f);

    //float3 minT = float3(-5.3174e-03, -5.4951e-03, -5.9748e-03);
    //float3 maxT = float3(5.7197e-03, 6.1895e-03, 6.3055e-03);

    float alpha2 = 150.;
    float3 minT = float3(logit(getValue(-2., 0.), alpha2), logit(getValue(-2., 1.), alpha2), logit(getValue(-2., 2.), alpha2));
    float3 maxT = float3(logit(getValue(-3., 0.), alpha2), logit(getValue(-3., 1.), alpha2), logit(getValue(-3., 2.), alpha2));

    //float lambda = (meannbofimpacts * fadeImage * sqrt(sqrt(sqrt(meannbofimpacts))) / (r * r));

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
                    
                    //gl_FragColor.xyz += float3(textonTemp.x * (abs(minT.x) + maxT.x) - abs(minT.x), textonTemp.y * (abs(minT.y) + maxT.y) - abs(minT.y), textonTemp.z * (abs(minT.z) + maxT.z) - abs(minT.z));
                    gl_FragColor.xyz += float3(textonTemp.x * (maxT.x - minT.x) + minT.x, textonTemp.y * (maxT.y - minT.y) + minT.y, textonTemp.z * (maxT.z - minT.z) + minT.z);
                    //gl_FragColor.xyz += float3(textonTemp.x -0.5, textonTemp.y - 0.5, textonTemp.z - 0.5);

                    //gl_FragColor.xyz += float3(0.f, 0.f, 0.1f);
                }
            }
        }
    }
    

    /* normalize and add mean */
    float lambda = meannbofimpacts / (r*r);
    gl_FragColor.xyz = 1.0f / sqrt(lambda) * (gl_FragColor.xyz - lambda * sumTexton) + noiseMean;
    //gl_FragColor.xyz = noiseMean;
    gl_FragColor.xyz = pow(gl_FragColor.xyz, 2.2f); // go linear
    
    //gl_FragColor.xyz = float3(0., 0., 0.);
    //float3 textemp = textonTexture.SampleGrad(mySampler, x_tex.xy, ddx(x_tex.xy), ddy(x_tex.xy));
    //gl_FragColor.xyz = newadd * (float3(textemp.x * (maxT.x - minT.x) + minT.x + maxT.x/2, textemp.y * (maxT.y - minT.y) + minT.y + maxT.y/2, textemp.z * (maxT.z - minT.z) + minT.z + maxT.z/2) + addImg);

    //gl_FragColor.xyz = textonTexture.SampleLevel(mySampler, float2(fadeImage+0.5, addImg-0.5)/130., 0).a;
    //gl_FragColor.xyz = getValue(fadeImage, addImg);
    //gl_FragColor.xyz = textonTexture.SampleLevel(mySampler, x_tex.xy, 0).a;
    //gl_FragColor.xyz = tex2D(textonTexture, float2(fadeImage,addImg)).a;


    gl_FragColor.w = 1.0f;
    
    
}

#endif //MYHLSLINCLUDE_INCLUDED