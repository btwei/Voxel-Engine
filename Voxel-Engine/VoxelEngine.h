#pragma once
#include "WindowHandler.h"
#include "VulkanRenderer.h"
#include "ChunkHandler.h"
#include "Camera.h"
#include "VertexTypes.h"

class VoxelEngine {
public:
	void registerEvent(std::string event, void* callback);
	void placeVoxel();
	void run();
private:
	//WindowHandler windowHandler;
	VulkanRenderer vulkanRenderer;
	//ChunkHandler chunkHandler;
	GLFWwindow* window;   
	//Camera defaultCamera;

	void init();
	void update();
	void mainLoop();
	void cleanup();
};

