#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "../vkInit.hpp"

#define App Application::getInstance()

using inputcallback = void(*)(double, double);

class Application {
public:

	~Application();
	static Application* getInstance();

	GLFWwindow* getWindow();

	void init(const int& w = 800, const int& h = 600);

	bool update();

	void end();

	void setMouseCallback(inputcallback callback);
	void setScrollCallback(inputcallback callback);

private:

	Application();
	static Application* instance;

	GLFWwindow* window{ nullptr };
	const char* windowtitle = "vulkan_7_depth_buffer";
	int width{ 0 };
	int height{ 0 };
	bool framebufferResized = false;
	void TitleFPS();

	static void framebufferResizedCallback(GLFWwindow* window, int width, int height);


	inputcallback mouse{ nullptr };
	inputcallback scroll{ nullptr };

	static void MouseCallback(GLFWwindow* window, double xposIn, double yposIn);
	static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

};