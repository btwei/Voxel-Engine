#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
	Camera();
	Camera(glm::vec3 position, float yaw, float pitch);

	glm::mat4 m_proj;
	glm::mat4 m_view;

	void update();

	float fovy;
	float near;
	float far;

private:
	glm::vec3 position;
	float yaw;
	float pitch;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec3 forward;
};