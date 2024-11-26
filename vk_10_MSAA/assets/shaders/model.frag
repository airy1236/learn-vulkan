#version 460 core

layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D textureSampler;

void main() {
    outColor = texture(textureSampler, fragTexCoord);
}