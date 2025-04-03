#version 330 core

in  vec3 vCol;
layout (location = 0) out vec4 oColor;

void main(void){
    oColor = vec4(vCol, 1.);
}
