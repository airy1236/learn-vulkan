#include "application/application.h"
#include "vkInit.hpp"
#include "framework/camera/camera.h"

const int scr_width = 800;
const int scr_height = 600;

Camera camera(glm::vec3(0.0, 0.0, 5.0));

float lastX = scr_width / 2.0f;
float lastY = scr_height / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

void processInput() {
    if (glfwGetKey(App->getWindow(), GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(App->getWindow(), true);

    if (glfwGetKey(App->getWindow(), GLFW_KEY_W) == GLFW_PRESS) camera.KeyMove(1, deltaTime);
    if (glfwGetKey(App->getWindow(), GLFW_KEY_A) == GLFW_PRESS) camera.KeyMove(2, deltaTime);
    if (glfwGetKey(App->getWindow(), GLFW_KEY_S) == GLFW_PRESS) camera.KeyMove(3, deltaTime);
    if (glfwGetKey(App->getWindow(), GLFW_KEY_D) == GLFW_PRESS) camera.KeyMove(4, deltaTime);
}

void MouseCallback(double xposIn, double yposIn) {
    float xpos = xposIn;
    float ypos = yposIn;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.MouseMove(xoffset, yoffset);
}

void ScrollCallback(double xoffset, double yoffset) {
    camera.MouseScroll(yoffset);
}


int main() {
    App->init(scr_width, scr_height);
    App->setMouseCallback(MouseCallback);
    App->setScrollCallback(ScrollCallback);

    while (App->update()) {

        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput();

        vkInit::Base().drawFrame(camera);

    }

    App->end();

    return 0;
}