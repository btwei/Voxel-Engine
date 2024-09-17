#include "ChunkHandler.h"

void ChunkHandler::init(VulkanRenderer* vr) {
	vulkanRenderer = vr;
	vr->setStagingBuffer(getVertices(), getIndices());
}

void ChunkHandler::update() {

}

void ChunkHandler::cleanup() {

}

std::vector<Vertex> ChunkHandler::getVertices() {
	return {
	{{0.5f, -0.5f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
	{{-0.5f, -0.5f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
	{{0.5f, 0.5f, 1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
	{{-0.5f, 0.5f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}}
	};
}

std::vector<uint16_t> ChunkHandler::getIndices() {
	return {
	0, 1, 2,
	2, 1, 3
	};
}