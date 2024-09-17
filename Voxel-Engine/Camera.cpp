#include "Camera.h"

Camera::Camera() {
	fovy = 45.0f;
	near = 0.1f;
	far = 10.0f;

	position = glm::vec3(0.0f, 0.0f, 0.0f);
	yaw = 0.0f;
	pitch = 0.0f;

	up = glm::vec3(0, -1, 0);
	right = glm::vec3(1, 0, 0);
	forward = glm::vec3(0, 0, 1);

	m_view = glm::lookAt(position, forward, up);
}

Camera::Camera(glm::vec3 position, float yaw, float pitch) : position(position) {
	m_view = glm::lookAt(position, position + forward, up);
}

void Camera::update() {

}