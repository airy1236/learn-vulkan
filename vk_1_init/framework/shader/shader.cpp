#include "shader.h"

Shader::Shader(const std::string& filename, shaderc_shader_kind kind) {
    // 读取文件内容
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file!");
    }
    size_t fileSize = (size_t)file.tellg();
    std::string shaderCode(fileSize, '\0');
    file.seekg(0);
    file.read(&shaderCode[0], fileSize);
    file.close();
    // 编译GLSL代码为SPIR-V字节码
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    
    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(shaderCode, kind, filename.c_str(), options);
    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(module.GetErrorMessage());
    }

    spirvCode = { module.cbegin(), module.cend() };

}

Shader::~Shader() {}

std::vector<uint32_t> Shader::data() {
    return spirvCode;
}