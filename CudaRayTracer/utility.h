#pragma once

#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <math.h>
#include <stdlib.h>
#include <float.h>

#include <glm/glm.hpp>

#include <curand_kernel.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using glm::vec3;

// Constants

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA check errors
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * M_PI / 180.0f;
}

__device__ float random_float(curandState* local_rand_state, float min, float max) {
    return curand_uniform(local_rand_state) * (max - min) + min;
}

__device__ vec3 random_vec3(curandState* local_rand_state, float start, float end) {
    return vec3(random_float(local_rand_state, start, end),
            random_float(local_rand_state, start, end),
            random_float(local_rand_state, start, end));
}

__device__ vec3 random_unit_vector(curandState* local_rand_state) {
    float theta = random_float(local_rand_state, 0.0, M_PI); 
    float phi = random_float(local_rand_state, 0.0, 2.0 * M_PI);
    float x = sinf(theta) * cosf(phi);
    float y = sinf(theta) * sinf(phi);
    float z = cosf(theta);
 
    return vec3(x, y, z);
}

__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    float theta = random_float(local_rand_state, 0, 2 * M_PI);
    float r = random_float(local_rand_state, 0.001f, 1.0f);

    return vec3(r*cosf(theta), r*sinf(theta), 0);
}

__device__ vec3 random_on_hemisphere(curandState* local_rand_state, const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector(local_rand_state);
    if (glm::dot(on_unit_sphere, normal) > 0.0f) {
        return on_unit_sphere;
    }
    else {
        return -on_unit_sphere;
    }
}

__device__ vec3 reflect(const vec3& v, const vec3& normal) {
    return v - 2.0f * glm::dot(v, normal) * normal;
}

__device__ vec3 refract(const vec3& uv, const vec3& normal, float etai_over_etat) {
    float cos_theta = std::fmin(glm::dot(-uv, normal), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * normal);
    vec3 r_out_parallel = -std::sqrtf(std::fabs(1.0f - std::pow(glm::length(r_out_perp),2.0f))) * normal;
    return r_out_perp + r_out_parallel;
}

// Common Headers

#include "color.h"
#include "ray.h"
#include "interval.h"

