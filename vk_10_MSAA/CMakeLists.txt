cmake_minimum_required(VERSION 3.2)

project(Vulkan_10_MSAA)

set(CMAKE_CXX_STANDARD 20)
# Դ�ļ���
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

add_executable(Vulkan_10_MSAA "main.cpp" "vkInit.hpp" "renderDataStruct.h")

target_link_libraries(Vulkan_10_MSAA glfw3.lib vulkan-1.lib shaderc_shared.lib assimp-vc143-mtd.lib app fw)