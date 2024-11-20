#pragma once

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
static std::mt19937 gen(std::random_device{}());

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA check errors
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, uint32_t const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<uint32_t>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Utility Functions

// Helper min/max functions
__device__ float min(std::initializer_list<float> list) {
    float min_val = *list.begin();
    for (auto val : list) {
        min_val = fminf(min_val, val);
    }
    return min_val;
}

__device__ float max(std::initializer_list<float> list) {
    float max_val = *list.begin();
    for (auto val : list) {
        max_val = fmaxf(max_val, val);
    }
    return max_val;
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1903, 0, 0, rand_state);
    }
}

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * M_PI / 180.0f;
}

__device__ float random_float(curandState* local_rand_state, float min, float max) {
    return curand_uniform(local_rand_state) * (max - min) + min;
}

float random_float(float min, float max) {
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(gen);
}

__device__ vec3 random_vec3(curandState* local_rand_state, float start, float end) {
    return vec3(random_float(local_rand_state, start, end),
            random_float(local_rand_state, start, end),
            random_float(local_rand_state, start, end));
}

vec3 random_vec3(float start, float end) {
    return vec3(random_float(start, end), random_float(start, end), random_float(start, end));
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
    for (int i = 0; i < 10; i++) {
        auto p = vec3(random_float(local_rand_state, -1, 1), random_float(local_rand_state, -1, 1), 0);
        if (glm::length(p) < 1)
            return p;
    }

    auto p = vec3(random_float(local_rand_state, -1, 1), random_float(local_rand_state, -1, 1), 0);
    return p;
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

static void export_framebuffer_to_png(vec3* device_framebuffer, uint32_t width, uint32_t height, const char* filename) {
    // Allocate host memory for the framebuffer
    vec3* host_framebuffer = new vec3[width * height];

    // Copy data from device to host
    cudaMemcpy(host_framebuffer, device_framebuffer, width * height * sizeof(vec3), cudaMemcpyDeviceToHost);

    // Allocate memory for the image data in the format expected by stb_image_write
    unsigned char* image_data = new unsigned char[width * height * 3];

    // Convert vec3 float data to unsigned char
    for (uint32_t i = 0; i < width * height; ++i) {
        image_data[i * 3 + 0] = static_cast<unsigned char>(std::min(std::max(host_framebuffer[i].x * 255.0f, 0.0f), 255.0f));
        image_data[i * 3 + 1] = static_cast<unsigned char>(std::min(std::max(host_framebuffer[i].y * 255.0f, 0.0f), 255.0f));
        image_data[i * 3 + 2] = static_cast<unsigned char>(std::min(std::max(host_framebuffer[i].z * 255.0f, 0.0f), 255.0f));
    }

    // Write the image to a file
    stbi_write_png(filename, width, height, 3, image_data, width * 3);

    // Free allocated memory
    delete[] host_framebuffer;
    delete[] image_data;
}

__device__ void swap(float& a, float& b) {
    float t = a;
    a = b;
    b = t;
}

// Common Headers

#include "color.h"
#include "ray.h"
#include "interval.h"

