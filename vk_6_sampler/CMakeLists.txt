cmake_minimum_required(VERSION 3.2)

project(Vulkan_6_sampler)

set(CMAKE_CXX_STANDARD 20)
# 源文件组
file(GLOB NO_GROUP_FILES "*.cpp")
file(GLOB HPP "*.hpp")
file(GLOB HEARDER "*.h")

file(GLOB ASSETS "./assets")

file(COPY ${ASSETS} DESTINATION ${CMAKE_BINARY_DIR})

include_directories (
	SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/include
)

link_directories (
	SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/lib
)

add_subdirectory(application)
add_subdirectory(framework)

add_executable(Vulkan_6_sampler "main.cpp" "vkInit.hpp")

target_link_libraries(Vulkan_6_sampler glfw3.lib vulkan-1.lib shaderc_shared.lib app fw)