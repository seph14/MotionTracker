#version 330 core

in vec4 ciPosition;
in vec2 ciTexCoord0;

out vec2 vUv;

uniform mat4 ciModelViewProjection;

void main() {
    vUv         = vec2(0.,1.) + vec2(1.,-1.) * ciTexCoord0;
    gl_Position = ciModelViewProjection * ciPosition;
}
