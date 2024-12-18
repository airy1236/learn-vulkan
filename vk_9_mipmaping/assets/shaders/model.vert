#version 460 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec3 inTangent;

layout(location = 1) out vec2 fragTexCoord;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() {
    fragTexCoord = inTexCoord;

    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
}