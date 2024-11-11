#pragma once
#include "kernel.h"

void matSVD(glm::mat3& ABt, glm::mat3& U, glm::mat3& S, glm::mat3& V);

namespace ICP {
	void CPUStep(std::vector<glm::vec3>& dataBuffer, std::vector<glm::vec3>& modelBuffer);
	void naiveGPUStep();
}
