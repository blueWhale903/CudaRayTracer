#pragma once

#include <glm/vec3.hpp>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Ray {
public:
    __device__ Ray() {}

    __device__ Ray(const vec3& origin, const glm::vec3& direction) : origin_point(origin), dir(direction) {}

    __device__ const glm::vec3& origin() const { return origin_point; }
    __device__ const glm::vec3& direction() const { return dir; }

    __device__ glm::dvec3 at(float t) const {
        return origin_point + t * dir;
    }

private:
    glm::vec3 origin_point;
    glm::vec3 dir;
};
