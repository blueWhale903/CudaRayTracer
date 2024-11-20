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
#include "bvh.h"

using glm::vec3;

#define RND (curand_uniform(&local_rand_state))

class Scene {
public:
    Scene();
    void launch_random_scene_kernel(Hittable** d_world);
    void launch_triangles_scene(Hittable** d_world);
    void load_models(const std::vector<std::string>& modelPaths);
private:
    curandState* d_rand_state;
    struct DeviceResources {
        Model* models;
        Hittable** hittableList;
        BVHNode* bvhNodes;
        uint32_t* mortonCodes;
        uint32_t* indices;
        curandState* randState;
        void* triangleMemory;

        size_t numModels;
        size_t numTriangles;
        size_t numHittables;
    } m_resources;

    // Memory allocation with error checking
    void allocateDeviceMemory();
    void deallocateDeviceMemory();

    // Scene construction steps
    void preprocessTriangles();
    void constructBVH();
    void addGroundPlane();

};

__global__ void kernel_create_world(Hittable** d_list, Hittable** d_world, curandState* rand_state) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t ith_hittable = 0;
        curandState local_rand_state = *rand_state;

        d_list[ith_hittable++] = new Sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, new Lambertian(vec3(0.5, 0.5, 0.5)));

        for (int k = 0; k < 1; k++) {
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
        }
        d_list[ith_hittable++] = new Sphere(vec3(0, 1, 0), 1.0, new Dielectric(1.5));
        d_list[ith_hittable++] = new Sphere(vec3(-4, 1, 0), 1.0, new Lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[ith_hittable++] = new Sphere(vec3(4, 1, 0), 1.0, new Metal(vec3(0.7, 0.6, 0.5), 0.0));
        AABB scene_bbox = d_list[0]->bounding_box();
        for (uint32_t i = 1; i < ith_hittable; ++i) {
            scene_bbox = AABB(scene_bbox, d_list[i]->bounding_box());
        }
        *d_world = new HittableList(d_list, ith_hittable, scene_bbox);
        *rand_state = local_rand_state;
    }
}

__global__ void load_triangles(Hittable** d_list,
                               Model* d_models, 
                               uint32_t num_models,
                               uint32_t num_triangles,
                               char* d_obj_memory) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_triangles) return;

    // Safe model and triangle index calculation
    int model_index = 0, triangle_index = tid;
    for (int i = 0; i < num_models; i++) {
        if (triangle_index >= d_models[i].num_triangles) {
            triangle_index -= d_models[i].num_triangles;
            model_index = i + 1;
        }
        else {
            break;
        }
    }

    // Bounds checking
    if (model_index >= num_models) return;

    Model& model = d_models[model_index];

    // Use pre-allocated memory for Triangle
    char* triangle_memory = d_obj_memory + (tid * sizeof(Triangle));
    Triangle* triangle = new (triangle_memory) Triangle();

    // Safe triangle data loading
    uint32_t* indices = model.d_indices;
    Vertex* vertex = model.d_vertex;

    // Bounds checking for indices
    if (triangle_index * 3 + 2 >= model.num_triangles * 3) return;

    uint32_t i0 = indices[3 * triangle_index];
    uint32_t i1 = indices[3 * triangle_index + 1];
    uint32_t i2 = indices[3 * triangle_index + 2];

    uint32_t num_vertices = model.num_vertices;
    // Bounds checking for vertices
    if (i0 >= num_vertices ||
        i1 >= num_vertices ||
        i2 >= num_vertices) return;

    // Populate triangle data
    triangle->a = vertex[i0].position;
    triangle->b = vertex[i1].position;
    triangle->c = vertex[i2].position;
    triangle->normal_a = vertex[i0].normal;
    triangle->normal_b = vertex[i1].normal;
    triangle->normal_c = vertex[i2].normal;

    // Use pre-allocated or indexed material
    char* material_memory = d_obj_memory + (num_triangles * sizeof(Triangle)) + (tid * sizeof(Lambertian));
    triangle->material = new (material_memory) Lambertian(vec3(0.4, 0.3, 0.3));

    // Initialize triangle (AABB, etc.)
    triangle->init();

    // Store in list with atomic to prevent race conditions
    d_list[tid] = triangle;
 }


__global__ void kernel_triangles_scene(Hittable** d_list, void* d_hittable_memory, Hittable** d_world, uint32_t num_triangles, curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t ith_hittable = num_triangles;
        curandState local_rand_state = *rand_state;

        d_list[ith_hittable++] = new Sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, new Lambertian(vec3(0.5, 0.5, 0.5)));

        AABB scene_bbox = d_list[0]->bounding_box();
        for (uint32_t i = 1; i < ith_hittable; ++i) {
            scene_bbox = AABB(scene_bbox, d_list[i]->bounding_box());
        }

        *d_world = new HittableList(d_list, ith_hittable, scene_bbox);

        *rand_state = local_rand_state;
        printf("- Scene Created with %d objects\n", ith_hittable);
    }
}

__global__ void test_list(Hittable** d_list, uint32_t n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < n) {
        Ray r(vec3(0, 0, 0), vec3(1, 2, 3));
        HitRecord rec;
        d_list[id]->hit(r, Interval(0.001f, FLT_MAX), rec);
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
    uint32_t num_hittable = 22*22+1+3;
    uint32_t num_nodes = 2 * num_hittable - 1;

    checkCudaErrors(cudaMalloc((void**)&d_list, num_hittable * sizeof(Hittable*)));
    // Zero out the memory to ensure clean state
    checkCudaErrors(cudaMemset(d_list, 0, num_hittable * sizeof(Hittable*)));
    
    kernel_create_world<<<1, 1>>>(d_list, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    uint32_t* d_morton_codes;
    uint32_t* d_indices;
    BVHNode* d_nodes;

    checkCudaErrors(cudaMalloc((void**)&d_morton_codes, num_hittable * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc((void**)&d_indices, num_hittable * sizeof(uint32_t)));
    checkCudaErrors(cudaMalloc((void**)&d_nodes, num_nodes * sizeof(BVHNode)));

    uint32_t threads = 256;
    uint32_t blocks = (num_hittable + threads - 1) / threads;
    assign_morton_codes<<<blocks,threads>>>((HittableList**)d_world, d_indices, d_morton_codes, num_hittable);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::device_ptr<uint32_t> morton_codes_ptr(d_morton_codes);
    thrust::device_ptr<uint32_t> d_indices_ptr(d_indices);
    thrust::sort_by_key(morton_codes_ptr, morton_codes_ptr + num_hittable, d_indices_ptr);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    build_bvh<<<1, 1>>>(d_nodes, d_morton_codes, d_indices, num_hittable, (HittableList**)d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_list));
}

void Scene::launch_triangles_scene(Hittable** d_world) {
    std::vector<Model> models;
    Model* d_models;

    Model rabbit1("../models/dragon.obj");
    models.push_back(rabbit1);

    checkCudaErrors(cudaMalloc((void**)&d_models, models.size() * sizeof(Model)));
    checkCudaErrors(cudaMemcpy(d_models, models.data(), models.size() * sizeof(Model), cudaMemcpyHostToDevice));

    uint32_t num_spheres = 1; // Number of Spheres
    uint32_t num_triangles = 0;
    for (Model model : models) {
        num_triangles += model.num_triangles;
    }
    uint32_t num_hittable = num_triangles + num_spheres;

    void* d_hittable_memory;
    size_t hittable_memory_size = (num_hittable * sizeof(Hittable) + num_hittable * sizeof(Material));
    checkCudaErrors(cudaMalloc(&d_hittable_memory, hittable_memory_size + 256));
    d_hittable_memory = (void*)((uintptr_t(d_hittable_memory) + 255) & ~255);

    uint32_t num_nodes = 2 * num_hittable - 1;
    Hittable** d_list;
    checkCudaErrors(cudaMallocManaged((void**)&d_list, (num_hittable+1024) *sizeof(Hittable*)));
    // Zero out the memory to ensure clean state
    checkCudaErrors(cudaMemset(d_list, 0, (num_hittable+1024) *  sizeof(Hittable*)));

    size_t triangle_memory_size = num_triangles * sizeof(Triangle) * 2; // Double allocation
    void* d_triangle_memory;
    checkCudaErrors(cudaMalloc(&d_triangle_memory, triangle_memory_size));

    uint32_t threads = 256;
    uint32_t blocks = (num_hittable + threads - 1) / threads;

    load_triangles<<<blocks, threads>>>(d_list,
                                        d_models,
                                        models.size(),
                                        num_triangles,
                                        static_cast<char*>(d_triangle_memory));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    kernel_triangles_scene<<<1, 1>>>(d_list, d_hittable_memory, d_world, num_triangles, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    test_list<<<blocks, threads>>>(d_list, num_hittable);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    uint32_t* d_morton_codes;
    uint32_t* d_index;
    BVHNode* d_nodes;
    checkCudaErrors(cudaMallocManaged((void**)&d_morton_codes, num_hittable * sizeof(uint32_t)));
    checkCudaErrors(cudaMallocManaged((void**)&d_index, num_hittable * sizeof(uint32_t)));
    checkCudaErrors(cudaMallocManaged((void**)&d_nodes, num_nodes * sizeof(BVHNode)));

    assign_morton_codes<<<blocks, threads>>>((HittableList**)d_world, d_index, d_morton_codes, num_hittable);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::device_ptr<uint32_t> morton_codes_ptr(d_morton_codes);
    thrust::device_ptr<uint32_t> d_indices_ptr(d_index);
    thrust::sort_by_key(morton_codes_ptr, morton_codes_ptr + num_hittable, d_indices_ptr);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    build_bvh<<<1, 1>>>(d_nodes, d_morton_codes, d_index, num_hittable, (HittableList**)d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
