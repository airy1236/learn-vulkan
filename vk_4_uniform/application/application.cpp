#include "application.h"

Application* Application::instance = nullptr;

Application* Application::getInstance() {
	if (instance == nullptr) instance = new Application();
	return instance;
}

Application::Application() {}
Application::~Application() {}

GLFWwindow* Application::getWindow() {
	return window;
}

void Application::init(const int& w, const int& h) {
	width = w;
	height = h;

	if (!glfwInit()) throw std::runtime_error("failed to init glfw\n");
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(width, height, windowtitle, nullptr, nullptr);
	if (window == nullptr) throw std::runtime_error("failed to create window\n");

	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizedCallback);

	vkInit::Base().initVulkan(window, framebufferResized);
}

bool Application::update() {
	if (glfwWindowShouldClose(window)) return false;

	glfwPollEvents();

	TitleFPS();

	return true;
}

void Application::end() {
	vkInit::Base().destroy();

	glfwDestroyWindow(window);
	glfwTerminate();
}

void Application::TitleFPS() {
	static double time0 = glfwGetTime();
	double time1 = glfwGetTime();
	static int dframe = 0;
	double dt = time1 - time0;
	++dframe;
	if (dt >= 1.0) {
		std::stringstream info;
		info.precision(1);
		info << windowtitle << "     " << std::fixed << dframe / dt << "FPS";
		glfwSetWindowTitle(window, info.str().c_str());

		time0 = time1;
		dframe = 0;
	}
}

void Application::framebufferResizedCallback(GLFWwindow* window, int width, int height) {
	App->framebufferResized = true;
}