#version 330 core
uniform sampler2D uDepthTex, uSegTex;

in  vec2 vUv;

layout (location = 0) out vec4 oColor;

void main(void){
    float s  = texture(uSegTex, vUv).r;
    float d  = 65535. * texture(uDepthTex, vUv).r;
    float d0 = fract(d / 255.);
    float d1 = floor(d / 255.) / 255.; 
    oColor = vec4(s, d1, d1, d0);
}
