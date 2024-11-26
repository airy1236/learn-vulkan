#include "shader.h"

Shader::Shader(const std::string& filepath, shaderc_shader_kind kind) {
	std::ifstream file(filepath, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open the shader file");
	}
	size_t fileSize = (size_t)file.tellg();
	std::string shaderCode(fileSize, '\0');
	file.seekg(0);
	file.read(&shaderCode[0], fileSize);
	file.close();

	shaderc::Compiler compiler;
	shaderc::CompileOptions options;
	shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(shaderCode, kind, filepath.c_str(), options);
	if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
		throw std::runtime_error(module.GetErrorMessage());
	}
	spirvCode = { module.begin(), module.end() };

}
Shader::~Shader() {}

std::vector<uint32_t> Shader::data() {
	return spirvCode;
}