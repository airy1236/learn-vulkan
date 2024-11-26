#include "texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

Texture::Texture(const std::string path) {
	//stbi_set_flip_vertically_on_load(true);
	pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	imageSize = texWidth * texHeight * 4; //每个像素需要4个字节存储

	mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1; //计算mipmap层级

	if (!pixels) {
		throw std::runtime_error("failed to load texture image!\n");
	}
}
Texture::~Texture() {}

void Texture::destroy() {
	stbi_image_free(pixels);
}

int Texture::getWidth() {
	return texWidth;
}
int Texture::getHeight() {
	return texHeight;
}
int Texture::getChannels() {
	return texChannels;
}

unsigned char* Texture::getPixels() {
	return pixels;
}

VkDeviceSize Texture::getImageSize() {
	return imageSize;
}

uint32_t Texture::getMipLevels() {
	return mipLevels;
}