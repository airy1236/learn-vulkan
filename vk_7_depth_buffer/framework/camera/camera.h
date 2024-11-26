#pragma once

#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.1f;
const float FOV = 45.0f;

class Camera {
public:

	Camera(const glm::vec3& posiition = glm::vec3(0.0, 0.0, 0.0), const glm::vec3& up = glm::vec3(0.0, 1.0, 0.0),
		const float& yaw = YAW, const float& pitch = PITCH);

	Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);

	~Camera();

	void KeyMove(int dir, float dletaTime);

	void MouseMove(double xposIn, double yposIn);

	void MouseScroll(double yoffset);

	glm::mat4 getViewMatrix();

	glm::vec3 getPosition();

	glm::vec3 getFront();

	glm::vec3 getUp();

	glm::vec3 getRight();

	float getFov();

private:

	void updateCameraVector();

	glm::vec3 Position;
	glm::vec3 WorldUp;
	glm::vec3 Front;
	glm::vec3 Right;
	glm::vec3 Up;

	float Yaw;
	float Pitch;
	float Speed;
	float Sensitivity;
	float Fov;

};