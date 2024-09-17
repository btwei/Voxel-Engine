#pragma once

#include <glm/glm.hpp>
#include "VulkanRenderer.h"
#include "VertexTypes.h"

#define CHUNK_SIZE 32

class ChunkHandler {
public:
	void init(VulkanRenderer* vr);
	void update();
	void cleanup();

private:
	VulkanRenderer* vulkanRenderer;
	std::vector<Vertex> getVertices();
	std::vector<uint16_t> getIndices();
};

/*
class Mesh {
public:
	glm::vec3 m_model;
	glm::vec3 position;

	void render();
private:
	void getVao();
	void getVertices();
};

class Chunk : public Mesh{
public:
	glm::vec3 m_model;

private:
	glm::vec3 position;
};

class Block {
public:
	bool isActive();
	void setActive(bool active);
private:
	bool active;
	BlockType blockType;
};
*/