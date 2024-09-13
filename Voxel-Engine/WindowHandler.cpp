#include "WindowHandler.h"

GLFWwindow* WindowHandler::init() {
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, "Voxel Engine", nullptr, nullptr);
	//glfwSetWindowUserPointer(window, this);
	//glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

	return window;
}

GLFWwindow* WindowHandler::getWindow() {
	return window;
}

void WindowHandler::cleanup() {
	glfwDestroyWindow(window);

	glfwTerminate();
}