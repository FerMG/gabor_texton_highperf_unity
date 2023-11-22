//UNITY_SHADER_NO_UPGRADE
#ifndef MYHLSLINCLUDE_INCLUDED
#define MYHLSLINCLUDE_INCLUDED

static const int IMPULSE_CAP = 128;

static const float pi = 3.14159265358979323846f;
//#define bbsm 1739.0f
//#define pp_epsilon (1.f / 289.f)
//#define poisson_epsilon .001f

struct gnoise_params { //struct for input params to gabor noise
    float a, density, filterSigma, octaves;
    float4 sector;
    float2x2 jacob;
};

struct gnoise_im_params { //struct to pass intermediate values within gnoise
    float2x2 filter, sigma_f_plus_g_inv;
    float ainv, a_prime_square, filt_scale;
};

//hash based on Blum, Blum & Shub 1986
//and Sharpe http://briansharpe.wordpress.com/2011/10/01/gpu-texture-free-noise/
static const float bbsm = 1739.0f;//magic product of two primes chosen to have high period without float precision issues
float2 bbsopt(const float2 a) {
    return frac(a * a * bbsm);
}
float bbsopt(const float a) {
    return frac(a * a * bbsm);
}
float2 mod1024(const float2 v) {
    return v - floor(v * 0.0009765625) * 1024.;
}
float seed(const float2 p) {
    float2 h0 = bbsopt(p.xy * (1.0 / bbsm)); //scale to small value for bbsopt; fp precision errors will make quasi infinite
    float2 h1 = bbsopt(mod1024(p.xy) * (1.0 / bbsm) + .5); //repeats every 1024 to destroy fp precision artifacts
    float2 h2 = h0 + h1;//best of both worlds
    return bbsopt(h2.y + bbsopt(h2.x));
}

//permutation polynomial
//based on Gustavson/McEwan https://github.com/ashima/webgl-noise/
//and Sharpe http://briansharpe.wordpress.com/2011/10/01/gpu-texture-free-noise/
// the permutation x <- 34x*x + x mod 289 maps 0 to 0.
// to prevent small values from getting stuck at zero, add a small constant term
//for efficiency, store all values in (0,1) and use x%y = frac(x/y)*y
static const float pp_epsilon = (1. / 289.);
float nextRand(inout float u) {//rng
    u = frac((u * 34.0 + 1.0) * u + pp_epsilon);
    return frac((7. * 289. / 288.) * u);
}

//approximate poisson distribution uniform u
//from Galerne, Lagae, Lefebvre, Drettakis
static const float poisson_epsilon = .001;
int poisson(inout float u, const float m) {
    float u1 = nextRand(u);
    float u2 = nextRand(u);
    float x = sqrt(-2. * log(u1 + poisson_epsilon)) * cos(2. * pi * u2);
    return int(m + sqrt(m) * x + .5);
}

//Gabor noise based on Lagae, Lefebvre, Drettakis, Dutre 2011
float eval_cell(const float2 cpos, const float2 gpos, const float2 dnbr, const gnoise_params params, const gnoise_im_params im_params) {
    float u = seed(gpos + dnbr); //deterministic seed for nbr cell
    int impulses = poisson(u, params.density * (1. / pi)); //number of impulses in nbr cell
    float4 h = params.sector; //annular sector
    float a = params.a; //bandwidth
    float aps = im_params.a_prime_square; //intermediate calculations for filtering
    float filt_scale = im_params.filt_scale; //aps*im_params.ainv*im_params.ainv;
    float2 fpos = cpos - dnbr;//fragment position in cell space

    float acc = 0.;
    //for impulses
    for (int k = 0; k < IMPULSE_CAP; k++) {
        if (k < impulses) {
            //position of impulse in cell space - uniform distribution
            float2 ipos = float2(nextRand(u), nextRand(u));
            //displacement to fragment
            float2 delta = (fpos - ipos) * im_params.ainv;
            //impulse frequency, orientation - distribution on input ranges
            float mfreq = pow(2., nextRand(u) * params.sector.y);
            float ifreq = h.x * mfreq;
            float iorientation = lerp(h.z - .5 * h.w, h.z + .5 * h.w, nextRand(u));
            //evaluate kernel, accumulate fragment value
            float2 mu = ifreq * float2(cos(iorientation), sin(iorientation));
            float phi = nextRand(u); //phase - uniform dist [0, 1]
            float filt_exp = -.5 * dot(mu, mul(im_params.sigma_f_plus_g_inv, mu));
            acc += filt_scale / mfreq * exp(-pi * aps * dot(delta, delta) + filt_exp) * cos(2. * pi * (dot(delta, mul(im_params.filter, mu)) + phi));
        }
        else { break; }
    }
    return acc;
}

float det2x2(const float2x2 m) {
    return (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
}
float2x2 inv2x2(const float2x2 m) {
    return (1. / det2x2(m)) * float2x2(m[1][1], -m[0][1], -m[1][0], m[0][0]);
}
float2x2 id2x2() {
    return float2x2(1., 0., 0., 1.);
}

//annular sector of pink noise
float gnoise(float2 pos, gnoise_params params) {
    gnoise_im_params im_params;
    im_params.ainv = 1. / params.a;

    //compute positions for this fragment
    float2 temp = pos * params.a;
    float2 cpos = frac(temp);
    float2 gpos = floor(temp);

    float2x2 jacob = params.jacob;
    float2x2 jacob_t = float2x2(jacob[0][0], jacob[1][0], jacob[0][1], jacob[1][1]);
    float2x2 sigma_f_inv = (4. * pi * pi * params.filterSigma * params.filterSigma) * (jacob * jacob_t);
    float2x2 sigma_f = inv2x2(sigma_f_inv);
    float2x2 sigma_g_inv = (2. * pi * im_params.ainv * im_params.ainv) * id2x2();
    float2x2 sigma_g = inv2x2(sigma_g_inv);
    float2x2 sigma_fg_inv = sigma_f_inv + sigma_g_inv;
    float2x2 sigma_fg = inv2x2(sigma_fg_inv);

    //filter params
    im_params.filter = sigma_fg * sigma_g_inv;
    im_params.sigma_f_plus_g_inv = inv2x2(sigma_f + sigma_g);
    im_params.a_prime_square = 2. * pi * sqrt(det2x2(sigma_fg));
    im_params.filt_scale = im_params.a_prime_square * im_params.ainv * im_params.ainv;

    float value =
        eval_cell(cpos, gpos, float2(-1., -1.), params, im_params) +
        eval_cell(cpos, gpos, float2(-1., 0.), params, im_params) +
        eval_cell(cpos, gpos, float2(-1., 1.), params, im_params) +
        eval_cell(cpos, gpos, float2(0., -1.), params, im_params) +
        eval_cell(cpos, gpos, float2(0., 0.), params, im_params) +
        eval_cell(cpos, gpos, float2(0., 1.), params, im_params) +
        eval_cell(cpos, gpos, float2(1., -1.), params, im_params) +
        eval_cell(cpos, gpos, float2(1., 0.), params, im_params) +
        eval_cell(cpos, gpos, float2(1., 1.), params, im_params);


    //ad hoc attempt to normalize
    value *= .5 * pow(params.density + 1., -.5);
    float octexp = pow(2., params.sector.y);
    value *= (1. + params.sector.y) * octexp / (2. * octexp - 1.);

    return value;


}




void MyFunction_float(float2 uv, float viewportWidth, float planeSize, float2 origin, float bandwidth, float density, float freq, float octaves, float orientation, float isotropy, float wbandwidth, float wfreq, float woctaves, float worientation, float wisotropy, float warp, float wdirection, float wspread, out float4 _out2)
{
    //flip uv
    uv = float2(1.-uv.x, 1. - uv.y);

    //float viewportWidth = 512.; 
    float viewportHeight = viewportWidth;
    //var origin = -.5;
    //float planeSize = 100.;

    float minfreq = planeSize * .5;
    float maxfreq = .25 * (min(viewportWidth, viewportHeight) * planeSize);

    float2 pos = uv.xy + origin.xy;

    freq = pow(2., freq * 6.) * .5 * planeSize;
    octaves = octaves * 8.;
    orientation = orientation * 3.14;
    isotropy = isotropy * 3.14;
    bandwidth = 1. / ((1. - bandwidth) * 4. / freq + 1. / maxfreq);
    
    gnoise_params params;
    params.a = bandwidth;
    params.filterSigma = 1.;
    params.jacob = float2x2(ddx(uv.xy), ddy(uv.xy));
    params.sector = float4(freq, octaves, orientation, isotropy);
    params.density = density * 127. + 1.; //mean impulses per grid cell

    wfreq = freq * wfreq * 2.;
    woctaves = woctaves * 8.;
    worientation = orientation + worientation * 3.14;
    wisotropy = wisotropy * 3.14;
    wbandwidth = 1. / ((1. - wbandwidth) * 4. / wfreq + 1. / maxfreq);
    
    gnoise_params wparams;
    wparams.a = wbandwidth;
    wparams.filterSigma = params.filterSigma;
    wparams.jacob = params.jacob;
    wparams.sector = float4(wfreq, woctaves, worientation, wisotropy);
    wparams.density = params.density;

    warp = warp / freq;
    wspread = wspread * 2. * pi;
    wdirection = wdirection * pi * 2.;

    float2 p0 = float2(1., 1.);
    float osx = gnoise(pos + p0, wparams);
    //float osy = gnoise(pos-p0, params);
    pos = pos + warp * float2(cos(wspread * osx + wdirection), sin(wspread * osx + wdirection));

    float value = gnoise(pos, params);
    value = value * .5 + .5;

    //monochrome
    float3 c = float3(value, value, value);

    c = pow(c, 2.2);;
    //draw fragment
    _out2 = float4(c, 1.0);
}

#endif //MYHLSLINCLUDE_INCLUDED