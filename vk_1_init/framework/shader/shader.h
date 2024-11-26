#pragma once

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

class Shader {
public:

	Shader(const std::string& filename, shaderc_shader_kind kind);
	~Shader();

	std::vector<uint32_t> data();

private:

	std::vector<uint32_t> spirvCode;

};