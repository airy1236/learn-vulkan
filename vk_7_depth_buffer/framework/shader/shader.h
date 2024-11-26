#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>

#include <vulkan/vulkan.h>

#include <shaderc/shaderc.hpp>

class Shader {
public:

	Shader(const std::string& filepath, shaderc_shader_kind kind);
	~Shader();

	std::vector<uint32_t> data();

private:

	std::vector<uint32_t> spirvCode;

};