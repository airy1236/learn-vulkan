#pragma once

#include <iostream>

#include <vulkan/vulkan.h>

class Texture {
public:

	Texture(const char* path);
	~Texture();

	void destroy();

	int getWidth();
	int getHeight();
	int getChannels();

	unsigned char* getPixels();

	VkDeviceSize getImageSize();

private:

	int texWidth{ 0 };
	int texHeight{ 0 };
	int texChannels{ 0 };

	unsigned char* pixels{ nullptr };

	VkDeviceSize imageSize{ 0 };

};