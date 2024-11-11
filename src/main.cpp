/**
* @file      main.cpp
* @brief     GPU Accelerated Go-ICP
* @authors   Zhaojin Sun, Mufeng Xu
* @copyright 2024 Zhaojin & Mufeng. All rights reserved.
* @note      This code framework is based on CIS5650 Project 1
*/

#include "main.hpp"
#include "tinyply.h"

#define VISUALIZE 1
#define CUDA_NAIVE 1

int numPoints = 0;
int numDataPoints = 0;
int numModelPoints = 0;

std::vector<glm::vec3> dataBuffer;
std::vector<glm::vec3> modelBuffer;

glm::vec3* dev_pos;
glm::vec3* dev_col;

glm::vec3* dev_dataBuffer;
glm::vec3* dev_modelBuffer;
glm::vec3* dev_corrBuffer;

glm::vec3* dev_centeredCorrBuffer;
glm::vec3* dev_centeredDataBuffer;
glm::mat3* dev_ABtBuffer;

// Helper function to determine the file extension
std::string getFileExtension(const std::string& filename) {
	size_t pos = filename.find_last_of(".");
	if (pos != std::string::npos) {
		std::string ext = filename.substr(pos + 1);
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
		return ext;
	}
	return "";
}

// Function to load a TXT point cloud file
bool loadTXTPointCloud(const std::string& FName, int& N, std::vector<glm::vec3>& buffer) {
	std::ifstream ifile(FName);
	if (!ifile.is_open()) {
		std::cerr << "Unable to open point file '" << FName << "'" << std::endl;
		return false;
	}

	ifile >> N;
	if (N <= 0) {
		std::cerr << "Invalid number of points in the file: " << FName << std::endl;
		return false;
	}

	buffer.resize(N);
	for (int i = 0; i < N; i++) {
		float x, y, z;
		ifile >> x >> y >> z;
		buffer[i] = glm::vec3(x, y, z);
	}

	ifile.close();
	std::cout << "Loaded " << N << " points from TXT file '" << FName << "'" << std::endl;
	return true;
}

// Function to load a PLY point cloud file
bool loadPLYPointCloud(const std::string& FName, int& N, std::vector<glm::vec3>& buffer) {
	try {
		std::ifstream file_stream(FName, std::ios::binary);
		if (!file_stream.is_open()) {
			std::cerr << "Unable to open point file '" << FName << "'" << std::endl;
			return false;
		}

		tinyply::PlyFile ply_file;
		ply_file.parse_header(file_stream);

		std::shared_ptr<tinyply::PlyData> vertices;
		try {
			vertices = ply_file.request_properties_from_element("vertex", { "x", "y", "z" });
		}
		catch (const std::exception& e) {
			std::cerr << "PLY file is missing 'x', 'y', or 'z' vertex properties: " << e.what() << std::endl;
			return false;
		}

		ply_file.read(file_stream);

		if (vertices && vertices->count > 0) {
			N = static_cast<int>(vertices->count);
			buffer.resize(N);

			const float* vertex_buffer = reinterpret_cast<const float*>(vertices->buffer.get());
			for (int i = 0; i < N; ++i) {
				buffer[i] = glm::vec3(
					vertex_buffer[3 * i + 0],
					vertex_buffer[3 * i + 1],
					vertex_buffer[3 * i + 2]
				);
			}

			file_stream.close();
			std::cout << "Loaded " << N << " points from PLY file '" << FName << "'" << std::endl;
			return true;
		}
		else {
			std::cerr << "No vertices found in the PLY file: " << FName << std::endl;
			return false;
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error reading PLY file '" << FName << "': " << e.what() << std::endl;
		return false;
	}
}

// Main function to load point cloud from a file with automatic type detection
bool loadPointCloud(const std::string& FName, int& N, std::vector<glm::vec3>& buffer) {
	std::string extension = getFileExtension(FName);

	if (extension == "txt") {
		return loadTXTPointCloud(FName, N, buffer);
	}
	else if (extension == "ply") {
		return loadPLYPointCloud(FName, N, buffer);
	}
	else {
		std::cerr << "Unsupported file format: " << extension << std::endl;
		return false;
	}
}

/*
*C main function.
*/
int main(int argc, char* argv[]) {
	projectName = "Fast Globally Optimal ICP";

	if (init(argc, argv)) {
		mainLoop();
		PointCloud::cleanupBuffers();
		return 0;
	}
	else {
		return 1;
	}
}

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
	std::cout << "Loading data points..." << std::endl;
	if (!loadPointCloud(argv[1], numDataPoints, dataBuffer)) {
		return false;
	}

	std::cout << "Loading model(target) points..." << std::endl;
	if (!loadPointCloud(argv[2], numModelPoints, modelBuffer)) {
		return false;
	}

	// Initialize drawing state
	numPoints = dataBuffer.size() + modelBuffer.size();
	std::cout << "Total " << numPoints << " points loaded" << std::endl;

	cudaDeviceProp deviceProp;
	int gpuDevice = 0;
	int device_count = 0;

	cudaGetDeviceCount(&device_count);

	if (gpuDevice > device_count) {
		std::cout
			<< "Error: GPU device number is greater than the number of devices!"
			<< " Perhaps a CUDA-capable GPU is not installed?"
			<< std::endl;
		return false;
	}
	cudaGetDeviceProperties(&deviceProp, gpuDevice);
	int major = deviceProp.major;
	int minor = deviceProp.minor;

	std::ostringstream ss;
	ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
	deviceName = ss.str();

	// Window setup stuff
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		std::cout
			<< "Error: Could not initialize GLFW!"
			<< " Perhaps OpenGL 3.3 isn't available?"
			<< std::endl;
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}

	initVAO();

	// Default to device ID 0. If you have more than one GPU and want to test a non-default one,
	// change the device ID.
	cudaGLSetGLDevice(0);

	cudaGLRegisterBufferObject(pointVBO_positions);
	cudaGLRegisterBufferObject(pointVBO_colors);

	PointCloud::initBuffers(dataBuffer, modelBuffer);

	updateCamera();
	initShaders(program);
	glEnable(GL_DEPTH_TEST);
	return true;
}

void initVAO() {

	std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (numPoints)] };
	std::unique_ptr<GLuint[]> bindices{ new GLuint[numPoints] };

	glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
	glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

	for (int i = 0; i < numPoints; i++) {
		bodies[4 * i + 0] = 0.0f;
		bodies[4 * i + 1] = 0.0f;
		bodies[4 * i + 2] = 0.0f;
		bodies[4 * i + 3] = 1.0f;
		bindices[i] = i;
	}


	glGenVertexArrays(1, &pointVAO); // Attach everything needed to draw a particle to this
	glGenBuffers(1, &pointVBO_positions);
	glGenBuffers(1, &pointVBO_colors);
	glGenBuffers(1, &pointIBO);

	glBindVertexArray(pointVAO);

	// Bind the positions array to the pointVAO by way of the pointVBO_positions
	glBindBuffer(GL_ARRAY_BUFFER, pointVBO_positions); // bind the buffer
	glBufferData(GL_ARRAY_BUFFER, 4 * (numPoints) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data
	glEnableVertexAttribArray(positionLocation);
	glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	// Bind the colors array to the pointVAO by way of the pointVBO_colors
	glBindBuffer(GL_ARRAY_BUFFER, pointVBO_colors);
	glBufferData(GL_ARRAY_BUFFER, 4 * (numPoints) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(colorsLocation);
	glVertexAttribPointer((GLuint)colorsLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pointIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (numPoints) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void initShaders(GLuint * program) {
	GLint location;

	program[PROG_POINT] = glslUtility::createProgram(
		"shaders/point.vert.glsl",
		"shaders/point.geom.glsl",
		"shaders/point.frag.glsl", attributeLocations, 2);
	glUseProgram(program[PROG_POINT]);

	if ((location = glGetUniformLocation(program[PROG_POINT], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
	if ((location = glGetUniformLocation(program[PROG_POINT], "u_cameraPos")) != -1) {
		glUniform3fv(location, 1, &cameraPosition[0]);
	}
}

void runCUDA() {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
	// use this buffer

	float4 *dptr = NULL;
	float *dptrVertPositions = NULL;
	float *dptrVertcolors = NULL;

	cudaGLMapBufferObject((void**)&dptrVertPositions, pointVBO_positions);
	cudaGLMapBufferObject((void**)&dptrVertcolors, pointVBO_colors);

	// execute the kernel
#if CUDA_NAIVE
	ICP::naiveGPUStep();
#else
	ICP::CPUStep(dataBuffer, modelBuffer);
#endif

#if VISUALIZE
	PointCloud::copyPointsToVBO(dptrVertPositions, dptrVertcolors);
#endif
	// unmap buffer object
	cudaGLUnmapBufferObject(pointVBO_positions);
	cudaGLUnmapBufferObject(pointVBO_colors);
}

void mainLoop() {
	double fps = 0;
	double timebase = 0;
	int frame = 0;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		frame++;
		double time = glfwGetTime();

		if (time - timebase > 1.0) {
			fps = frame / (time - timebase);
			timebase = time;
			frame = 0;
		}

		runCUDA();

		std::ostringstream ss;
		ss << "[";
		ss.precision(1);
		ss << std::fixed << fps;
		ss << " fps] " << deviceName;
		glfwSetWindowTitle(window, ss.str().c_str());

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if VISUALIZE
		glUseProgram(program[PROG_POINT]);
		glBindVertexArray(pointVAO);
		glPointSize((GLfloat)pointSize);
		glDrawElements(GL_POINTS, numPoints + 1, GL_UNSIGNED_INT, 0);
		glPointSize(1.0f);

		glUseProgram(0);
		glBindVertexArray(0);

		glfwSwapBuffers(window);
#endif
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}


void errorCallback(int error, const char *description) {
	fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (leftMousePressed) {
		// compute new camera parameters
		phi += (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
		updateCamera();
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, std::fmin(zoom, 10000.0f));
		updateCamera();
	}

	lastX = xpos;
	lastY = ypos;
}

void updateCamera() {
	cameraPosition.x = zoom * sin(phi) * sin(theta);
	cameraPosition.z = zoom * cos(theta);
	cameraPosition.y = zoom * cos(phi) * sin(theta);
	cameraPosition += lookAt;

	projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
	glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
	projection = projection * view;

	GLint location;

	glUseProgram(program[PROG_POINT]);
	if ((location = glGetUniformLocation(program[PROG_POINT], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
}
