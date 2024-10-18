#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <glm/vec3.hpp>

#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "model_loader.h"

using glm::vec3;

#define RND (curand_uniform(&local_rand_state))

class Scene {
public:
    Scene();
    void launch_random_scene_kernel(Hittable** d_world);
    void launch_triangles_scene(Hittable** d_world);
private:
    curandState* d_rand_state;

};

__global__ void kernel_create_world(Hittable** d_list, Hittable** d_world, curandState* rand_state) {
    uint32_t i = 0;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        d_list[i++] = new Sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, new Lambertian(vec3(0.5, 0.5, 0.5)));

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new Sphere(center, 0.2,
                        new Lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = new Sphere(center, 0.2,
                        new Metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[i++] = new Sphere(center, 0.2f, new Dielectric(1.5));
                }
            }
        }
        d_list[i++] = new Sphere(vec3(0, 1, 0), 1.0, new Dielectric(1.5));
        d_list[i++] = new Sphere(vec3(-4, 1, 0), 1.0, new Lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new Sphere(vec3(4, 1, 0), 1.0, new Metal(vec3(0.7, 0.6, 0.5), 0.0));

        *rand_state = local_rand_state;
        *d_world = new HittableList(d_list, 22 * 22 + 1 + 3);
    }
}

__global__ void kernel_triangles_scene(Vertex* d_vertex, uint32_t* d_indices,
    Hittable** d_list, Hittable** d_world,
    uint32_t num_triangles, curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (uint32_t i = 0; i < (num_triangles / 3); i++) {
            uint32_t i0 = d_indices[3 * i];
            uint32_t i1 = d_indices[3 * i + 1];
            uint32_t i2 = d_indices[3 * i + 2];

            d_list[i] = new Triangle(
                d_vertex[i0].position, d_vertex[i1].position, d_vertex[i2].position,
                d_vertex[i0].normal, d_vertex[i1].normal, d_vertex[i2].normal
            );
        }

        curandState local_rand_state = *rand_state;

        uint32_t i = num_triangles / 3;
        d_list[i++] = new Sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, new Lambertian(vec3(0.5, 0.5, 0.5)));

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new Sphere(center, 0.2,
                        new Lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = new Sphere(center, 0.2,
                        new Metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[i++] = new Sphere(center, 0.2f, new Dielectric(1.5));
                }
            }
        }

        *rand_state = local_rand_state;
        printf("Number of hittable objects: %d\n", i);
        *d_world = new HittableList(d_list, i);
    }
}

Scene::Scene() {
    cudaMalloc((void**)&d_rand_state, sizeof(curandState));

    rand_init << <1, 1 >> > (d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Scene::launch_random_scene_kernel(Hittable** d_world) {
    Hittable** d_list = nullptr;
    uint32_t n = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void**)&d_list, n * sizeof(Hittable*)));

    kernel_create_world<<<1, 1 >>>(d_list, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_list));
}

void Scene::launch_triangles_scene(Hittable** d_world) {
    std::vector<Vertex> h_vertex;
    std::vector<uint32_t> h_indices;
    if (!LoadModel("../models/cats.obj", h_vertex, h_indices)) {
        std::cerr << "Failed to load OBJ model!" << std::endl;
        return;
    }
    uint32_t num_triangles = h_indices.size();
    uint32_t num_hittable = 22 * 22 + 1 + (num_triangles / 3);
    Hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hittable * sizeof(Hittable*)));

    Vertex* d_vertex;
    uint32_t* d_indices;
    checkCudaErrors(cudaMalloc((void**)&d_vertex, h_vertex.size() * sizeof(Vertex)));
    checkCudaErrors(cudaMalloc((void**)&d_indices, num_triangles * sizeof(uint32_t)));

    checkCudaErrors(cudaMemcpy(d_vertex, h_vertex.data(), h_vertex.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_indices, h_indices.data(), num_triangles * sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_triangles_scene<<<1, 1 >>>(d_vertex, d_indices, d_list, d_world, num_triangles, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
