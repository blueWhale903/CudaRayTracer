#pragma once

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
using glm::vec2;

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

__device__ __forceinline__ float random_float(curandState* local_rand_state, float min, float max) {
    return curand_uniform(local_rand_state) * (max - min) + min;
}

float random_float(float min, float max) {
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(gen);
}

__device__ __forceinline__ vec3 random_vec3(curandState* local_rand_state, float start, float end) {
    return vec3(random_float(local_rand_state, start, end),
            random_float(local_rand_state, start, end),
            random_float(local_rand_state, start, end));
}

vec3 random_vec3(float start, float end) {
    return vec3(random_float(start, end), random_float(start, end), random_float(start, end));
}

__device__ __forceinline__ vec3 random_unit_vector(curandState* local_rand_state) {
    //vec3 p;
    //for (int i = 0; i < 2; i++) {
    //    p = vec3(random_float(local_rand_state, -1, 1), random_float(local_rand_state, -1, 1), random_float(local_rand_state, -1, 1));
    //    if (glm::length(p) <= 1)
    //        return p;
    //}

    //return p;
    float theta = 2.0f * M_PI * curand_uniform(local_rand_state);
    float z = 2.0f * curand_uniform(local_rand_state) - 1.0f;
    float r = sqrtf(1.0f - z * z);
    return vec3(r * cosf(theta), r * sinf(theta), z);
}

__device__ __forceinline__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    vec3 p;
    for (int i = 0; i < 2; i++) {
        p = vec3(random_float(local_rand_state, -1, 1), random_float(local_rand_state, -1, 1), 0);
        if (glm::length(p) <= 1)
            return p;
    }

    return p;
}

__device__ __forceinline__ vec3 reflect(const vec3& v, const vec3& normal) {
    return v - 2.0f * glm::dot(v, normal) * normal;
}

__device__ __forceinline__ vec3 refract(const vec3& uv, const vec3& normal, float etai_over_etat) {
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

__device__ __forceinline__ vec3 min_vec(const vec3& a, const vec3& b) {
    return vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ vec3 max_vec(const vec3& a, const vec3& b) {
    return vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

unsigned char* LoadTexture(const std::string& filename, int width, int height, int channels) {
    return stbi_load(filename.c_str(), &width, &height, &channels, 3); // Force 3 channels (RGB)
}

__device__ vec2 interpolate_uv(const vec2& uv0, const vec2& uv1, const vec2& uv2,
    float u, float v) {
    vec2 uv = (1.0f - u - v) * uv0 + u * uv1 + v * uv2;
    // Scale UVs to texture dimensions
    uv.y = 1.0f - uv.y;
    return uv;
}
// Common Headers

#include "color.h"
#include "ray.h"
#include "interval.h"

