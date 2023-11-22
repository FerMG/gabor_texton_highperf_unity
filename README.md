Gabor Noise, Texton Noise e High-Performance By-Example Noise implementados no Shader Graph do Unity
=====================

Nesse repositório temos implementações do Gabor Noise, Texton Noise e High-Performance By-Example Noise no Shader Graph do Unity. Abaixo temos mais detalhes de como usá-los, créditos e licenças.

O projeto do Unity deste repositório possui uma cena com um exemplo de uso.

## Gabor Noise
Autor do código base original (em GLSL): Victor Shepardson

Link para seu repositório: https://github.com/victor-shepardson/webgl-gabor-noise

Licença: MIT license


## Texton Noise
Autores dos códigos base originais (GLSL e Matlab): B. Galerne, A. Leclaire, L. Moisan

Link para o artigo e código base original: https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13073

### Modo de uso:
Para produção das texturas de texton deve-se utilizar os scripts (Matlab ou Octave) presentes na pasta "compute_texton"

A textura texton pode ser produzida com o seguinte comando:
tn_compute_texton("C:/path/compute_texton/input_textures/wood_256.png", "C:/path/compute_texton/output_textons", 128);

O último argumento da função é o tamanho do texton em pixels: 64, 128, 256 ou 512

## High-Performance By-Example Noise
Autores do código base original (em GLSL): Eric Heitz, Fabrice Neyret

Link para o artigo: https://inria.hal.science/hal-01824773

Link para o código do shader original em GLSL: https://www.shadertoy.com/view/MdyfDV
