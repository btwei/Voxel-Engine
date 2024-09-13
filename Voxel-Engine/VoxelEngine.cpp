#include "VoxelEngine.h"

void VoxelEngine::registerEvent(std::string event, void* callback) {
	//onTick
	//onStartup
	//onCleanup
}

void VoxelEngine::placeVoxel() {

}

void VoxelEngine::run() {
	init();
	mainLoop();
	//vulkanRenderer.run();
	cleanup();
}

void VoxelEngine::init() {
	window = windowHandler.init();
	vulkanRenderer.init(window);
}

void VoxelEngine::update() {
	//camera.update();
	//vulkanRenderer.drawFrame();
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
	windowHandler.cleanup();
}