#pragma once
#include "WindowHandler.h"
#include "VulkanRenderer.h"
#include "Camera.h"

class VoxelEngine {
public:
	void registerEvent(std::string event, void* callback);
	void placeVoxel();
	void run();
private:
	WindowHandler windowHandler;
	VulkanRenderer vulkanRenderer;
	GLFWwindow* window;
	Camera camera;

	void init();
	void update();
	void mainLoop();
	void cleanup();
};

