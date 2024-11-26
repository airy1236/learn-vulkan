#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include "../vkInit.hpp"

#define App Application::getInstance()

class Application {
public:

	static Application* getInstance();

	GLFWwindow* getWindow();
	void setWindowTitle(const char* title);
	int getWidth();
	int getHeight();

	void init(const int& w = 800, const int& h = 600);

	bool update();

	void end();

	~Application();

private:

	static Application* instance;

	GLFWwindow* window{ nullptr };
	const char* windowtitle = "Vulkan_1_init";
	int width{ 0 };
	int height{ 0 };
	bool framebufferResized = false;
	void TitleFPS();

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

	Application();

};