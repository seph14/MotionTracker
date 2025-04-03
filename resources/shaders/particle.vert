#version 330 core
layout(location = 0) in uint idx;

uniform mat4  ciModelViewProjection, unProjMat;

uniform sampler2D uRgbTex, uDepthTex, uMaskTex;
uniform uint      uWidth, uHeight;
uniform vec3      uRange;
uniform float     uLift, uAspect, uProj;

out vec3 vCol;

void main(void){
    ivec2 iuv   = ivec2(idx % uWidth, idx / uWidth);
    vec2 uv     = (vec2(iuv) + .5) / vec2(uWidth, uHeight);
    vec2 fuv    = vec2(uv.x, 1. - uv.y);
    float mask  = step(.25, textureLod(uMaskTex, fuv, 0.).r);
    float depth = textureLod(uDepthTex, fuv, 0.).r;
    vec3  opos  = vec3( uAspect * uv.x - .5 * uAspect, 0. - uv.y, depth );
    vec3  pos   = vec3(0.f, uLift, uRange.y) + vec3(3. * uRange.x, 3. * uRange.x, uRange.z) * opos;

    vec4 tmp    = vec4(opos, 1.f);
    tmp.x       = (tmp.x + uAspect / 2.f) / uAspect;
    tmp.y       = 1.f + tmp.y;
    tmp.z       = 2.f * tmp.z / 3.f;
    tmp         = 2.f * tmp - 1.f;
    vec4 obj    = unProjMat * tmp;
    obj        /= obj.w;

    gl_Position = ciModelViewProjection * vec4( mix(pos, 4.*obj.xyz, uProj), mask);
    vCol        = textureLod(uRgbTex, uv, 0.).rgb;
}
