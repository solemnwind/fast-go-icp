#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilityCore.hpp"
#include "glslUtility.hpp"
#include "icp_kernel.h"
#include "kernel.h"

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>


//====================================
// GL Stuff
//====================================

GLuint positionLocation = 0;   // Match results from glslUtility::createProgram.
GLuint colorsLocation = 1; // Also see attribtueLocations below.
const char *attributeLocations[] = { "Position", "Velocity" };

GLuint pointVAO = 0;
GLuint pointVBO_positions = 0;
GLuint pointVBO_colors = 0;
GLuint pointIBO = 0;
GLuint displayImage;
GLuint program[2];

const unsigned int PROG_POINT = 0;

const float fovy = (float) (PI / 4);
const float zNear = 0.0001f;
const float zFar = 10000.0f;
int width = 1280;
int height = 720;
int pointSize = 2;

// For camera controls
bool leftMousePressed = false;
bool rightMousePressed = false;
double lastX;
double lastY;
float theta = 1.5f;
float phi = 1.0f;
float zoom = 5.0f;
glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.5f);
glm::vec3 cameraPosition;
glm::mat4 projection;

//====================================
// Main
//====================================

inline int ilog2(int x) {
	int lg = 0;
	while (x >>= 1) {
		++lg;
	}
	return lg;
}

inline int ilog2ceil(int x) {
	return x == 1 ? 0 : ilog2(x - 1) + 1;
}


const char *projectName;

int main(int argc, char* argv[]);

void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void updateCamera();
void runCUDA();

bool init(int argc, char **argv);
void initVAO();
void initShaders(GLuint *program);
