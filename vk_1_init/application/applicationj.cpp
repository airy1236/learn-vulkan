#include "application.h"

Application::Application() {}

Application::~Application() {}

Application* Application::instance = nullptr;

Application* Application::getInstance() {
	if (instance == nullptr) instance = new Application();
	return instance;
}

GLFWwindow* Application::getWindow() {
	return window;
}

void Application::setWindowTitle(const char* title) {
	windowtitle = title;
}

int Application::getWidth() {
	return width;
}
int Application::getHeight() { 
	return height;
}

void Application::init(const int& w, const int& h) {
	width = w;
	height = h;

	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(width, height, windowtitle, nullptr, nullptr);
	if (!window) { throw std::runtime_error("failed to create window!\n"); }

	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	
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
	static double time0 = glfwGetTime();  // 使用 static 确保只初始化一次  
	double time1 = glfwGetTime();
	static int dframe = 0;  // 静态变量，用于跨函数调用保持状态  
	double dt = time1 - time0;

	// 每一帧都增加 dframe  
	++dframe;

	// 如果时间差足够大（例如 >= 1 秒），则计算并显示帧率  
	if (dt >= 1.0) {
		std::stringstream info;
		info.precision(1);
		info << windowtitle << "    " << std::fixed << dframe / dt << " FPS";
		glfwSetWindowTitle(window, info.str().c_str());

		// 重置时间和帧数计数  
		time0 = time1;
		dframe = 0;
	}
}

void Application::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
	App->framebufferResized = true;
}
