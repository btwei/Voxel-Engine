#include "VoxelEngine.h"

void VoxelEngine::registerEvent(std::string event, void* callback) {

}

void VoxelEngine::placeVoxel() {

}

void VoxelEngine::run() {
	init();
	mainLoop();
	cleanup();
}

void VoxelEngine::init() {
	//window = windowHandler.init();
	//vulkanRenderer.initWindow(window);
	//vulkanRenderer.setCamera(&defaultCamera);
	//chunkHandler.init(&vulkanRenderer);

	std::vector<Vertex> newVertices = {
	    Vertex({0.5f, -0.5f, 1.0f}, {0.5f, 1.0f, 0.0f}, {1.0f, 0.0f}),
		Vertex({-0.5f, -0.5f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}),
		Vertex({0.5f, 0.5f, 1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}),
		Vertex({-0.5f, 0.5f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}),
		Vertex({-0.75f, 0.75f, 1.0f}, {0.5f, 0.5f, 0.0f}, {1.0f, 1.0f}),
		Vertex({0.5f, -0.5f, 1.0f}, {0.5f, 1.0f, 0.0f}, {1.0f, 0.0f}),
		Vertex({-0.5f, -0.5f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}),
		Vertex({0.5f, 0.5f, 1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}),
		Vertex({-0.5f, 0.5f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}),
		Vertex({-0.75f, 0.75f, 1.0f}, {0.5f, 0.5f, 0.0f}, {1.0f, 1.0f})
	};

	vulkanRenderer.setStagingBuffer(newVertices,{
	0, 1, 2,
	2, 1, 3,
	2, 3, 4
		});

	vulkanRenderer.initVulkan();
}

void VoxelEngine::update() {
	//Game State updates first
	//camera.update();

	//Render updates second
	vulkanRenderer.update();
}

void VoxelEngine::mainLoop() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		update();
		vulkanRenderer.drawFrame();
	}
}

void VoxelEngine::cleanup() {
	vulkanRenderer.cleanup();
	//windowHandler.cleanup();
}