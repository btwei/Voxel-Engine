#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "Config.h"

class WindowHandler {
public:
	GLFWwindow* init();
	GLFWwindow* getWindow();
	void cleanup();
private:
	GLFWwindow* window;
};

