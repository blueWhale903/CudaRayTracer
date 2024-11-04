#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <glm/vec3.hpp>

#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "model_loader.h"
#include "model.h"

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

__global__ void kernel_triangles_scene(Model* d_models, uint16_t num_models, 
    Hittable** d_list, Hittable** d_world, curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t ith_hittable = 0;

        Material* cat_materials[3] = { 
            new Dielectric(1.5),
            new Metal(vec3(0.7, 0.6, 0.5), 0.0), 
            new Lambertian(vec3(0.1, 0.2, 0.3)) 
        };

        for (int i = 0; i < num_models; i++) {
            Model model = d_models[i];
            uint32_t num_triangles = model.num_triangles;
            uint32_t* indices = model.d_indices;
            Vertex* vertex = model.d_vertex;
            for (uint32_t j = 0; j < num_triangles; j++) {
				uint32_t i0 = indices[3 * j];
				uint32_t i1 = indices[3 * j + 1];
				uint32_t i2 = indices[3 * j + 2];

                d_list[ith_hittable++] = new Triangle(
                    vertex[i0].position, vertex[i1].position, vertex[i2].position,
                    vertex[i0].normal, vertex[i1].normal, vertex[i2].normal,
                    cat_materials[i]
                );
			}
        }
        

        curandState local_rand_state = *rand_state;

        d_list[ith_hittable++] = new Sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, new Lambertian(vec3(0.5, 0.5, 0.5)));

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[ith_hittable++] = new Sphere(center, 0.2,
                        new Lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_list[ith_hittable++] = new Sphere(center, 0.2,
                        new Metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[ith_hittable++] = new Sphere(center, 0.2f, new Dielectric(1.5));
                }
            }
        }

        *rand_state = local_rand_state;
        printf("Number of hittable objects: %d\n", ith_hittable);
        *d_world = new HittableList(d_list, ith_hittable);
    }
}

Scene::Scene() {
    cudaMalloc((void**)&d_rand_state, sizeof(curandState));

    rand_init<<<1, 1>>>(d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Scene::launch_random_scene_kernel(Hittable** d_world) {
    Hittable** d_list = nullptr;
    uint32_t n = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void**)&d_list, n * sizeof(Hittable*)));

    kernel_create_world<<<1, 1>>>(d_list, d_world, d_rand_state);
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

    std::vector<Model> models;
    Model* d_models;
    Model cat1("../models/cat1.obj");
    Model cat2("../models/cat2.obj");
    Model cat3("../models/cat3.obj");

    models.push_back(cat1);
    models.push_back(cat2);
    models.push_back(cat3);

    checkCudaErrors(cudaMalloc((void**)&d_models, models.size() * sizeof(Model)));
    checkCudaErrors(cudaMemcpy(d_models, models.data(), models.size() * sizeof(Model), cudaMemcpyHostToDevice));
    
    uint32_t num_indices = h_indices.size();
    uint32_t num_triangles = num_indices / 3;
    uint32_t num_hittable = 22 * 22 + 1 + num_triangles;
    Hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hittable * sizeof(Hittable*)));

    Vertex* d_vertex;
    uint32_t* d_indices;
    checkCudaErrors(cudaMalloc((void**)&d_vertex, h_vertex.size() * sizeof(Vertex)));
    checkCudaErrors(cudaMalloc((void**)&d_indices, num_indices * sizeof(uint32_t)));

    checkCudaErrors(cudaMemcpy(d_vertex, h_vertex.data(), h_vertex.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_indices, h_indices.data(), num_indices * sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_triangles_scene<<<1, 1>>>(d_models, 3, d_list, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
