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
    void create_random_spheres_scene(Hittable** d_world);
    void create_cb_bunny(Hittable** d_world);
private:
    curandState* d_rand_state;
    Material* materials;

    void construct_bvh(Hittable** d_world, uint32_t num_hittables);
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
        d_list[ith_hittable++] = new Sphere(vec3(-20, 70, 0), 50.0, new DiffuseLight(vec3(1, 1, 1)));

        AABB scene_bbox = d_list[0]->bounding_box();
        for (uint32_t i = 1; i < ith_hittable; ++i) {
            scene_bbox = AABB(scene_bbox, d_list[i]->bounding_box());
        }
        *d_world = new HittableList(d_list, ith_hittable, scene_bbox);
        *rand_state = local_rand_state;

        printf("- Total primitives: %d\n", ith_hittable);
    }
}

Scene::Scene() {
    cudaMalloc((void**)&d_rand_state, sizeof(curandState));

    rand_init<<<1, 1>>>(d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Scene::create_random_spheres_scene(Hittable** d_world) {
    Hittable** d_list = nullptr;
    uint32_t num_hittable = 22*22+1+3+1;

    checkCudaErrors(cudaMalloc((void**)&d_list, num_hittable * sizeof(Hittable*)));
    // Zero out the memory to ensure clean state
    checkCudaErrors(cudaMemset(d_list, 0, num_hittable * sizeof(Hittable*)));
    
    kernel_create_world<<<1, 1>>>(d_list, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    construct_bvh(d_world, num_hittable);
}

__device__ void load_triangles(Hittable** d_list, char* d_triangles, Material** d_materials,
                               Texture** d_textures, Model* d_models, int num_models) {
    uint32_t ith_hittable = 0;

    // Initialize materials using placement new
    for (int i = 0; i < num_models; i++) {
        if (d_models[i].d_texture) {
            unsigned char* texture = d_models[i].d_texture;
             d_materials[i] = new Lambertian(new ImageTexture(texture, 2048, 2048));
            continue;
        }

        switch (i) {
        case 0:
            d_materials[i] = new Lambertian(vec3(0.73, 0.3, 0.3));
            break;
        case 1:
            d_materials[i] = new Lambertian(vec3(0.3, 0.73, 0.3));
            break;
        case 2:
            d_materials[i] = new DiffuseLight(vec3(14, 14, 14));
            break;
        case 6:
            d_materials[i] = new Metal(vec3(1, 1, 1), 0.35);
            break;
        default:
            d_materials[i] = new Lambertian(vec3(0.7, 0.7, 0.7));
            break;
        }
    }

    // Initialize triangles using placement new
    for (int i = 0; i < num_models; i++) {
        Model model = d_models[i];
        vec3* vertices = model.d_vertex;
        uint32_t* indices = model.d_indices;
        vec2* tex_coords = model.d_uvs; // Texture coordinates

        Material* mat = d_materials[i];

        for (int j = 0; j < model.num_triangles; j++) {
            char* triangle_memory = d_triangles + (ith_hittable * sizeof(Triangle));
            Triangle* triangle = new (triangle_memory) Triangle();

            int i0 = indices[3 * j];
            int i1 = indices[3 * j + 1];
            int i2 = indices[3 * j + 2];

            vec2 uv0 = tex_coords[i0];
            vec2 uv1 = tex_coords[i1];
            vec2 uv2 = tex_coords[i2];

            vec3 p0 = vertices[i0];
            vec3 p1 = vertices[i1];
            vec3 p2 = vertices[i2];

            triangle->uv_a = uv0;
            triangle->uv_b = uv1;
            triangle->uv_c = uv2;

            triangle->a = p0;
            triangle->b = p1;
            triangle->c = p2;
            triangle->material = mat;
            triangle->create_bbox();

            d_list[ith_hittable] = triangle;
            ith_hittable++;
        }
    }
}

__global__ void build_cb_bunny(Hittable** d_world, Hittable** d_list, char* d_triangles, Material** d_materials,
                               Texture** d_textures, Model* d_models, int num_models, int num_triangles) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int ith_hittable = num_triangles;

        load_triangles(d_list, d_triangles, d_materials, d_textures, d_models, num_models);

        //d_list[ith_hittable++] = new Sphere(vec3(-3, 3, 4), 3.0f, new Metal(vec3(1, 1, 1), 0.0f));
        //d_list[ith_hittable++] = new Sphere(vec3(2, 3, -4), 3.0f, new Dielectric(1.5));
        d_list[ith_hittable++] = new Sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, new Lambertian(vec3(0.5, 0.5, 0.5)));
        d_list[ith_hittable++] = new Sphere(vec3(-20, 100, 0), 50.0, new DiffuseLight(vec3(1, 1, 1)));

        AABB scene_bbox = d_list[0]->bounding_box();
        for (uint32_t i = 1; i < ith_hittable; ++i) {
            AABB bbox = d_list[i]->bounding_box();
            scene_bbox = AABB(scene_bbox, bbox);
        }

        *d_world = new HittableList(d_list, ith_hittable, scene_bbox);

        printf("- Total primitives: %d\n", ith_hittable);
    }
}

void Scene::create_cb_bunny(Hittable** d_world) {
    std::vector<Model> models;
    Model* d_models;

    //Model back("../models/back.obj", "../texture_images/tex1.png");
    //Model left("../models/left.obj");
    //Model right("../models/right.obj");
    //Model top("../models/top.obj");
    //Model bot("../models/bot.obj");
    //Model light("../models/light.obj");
    Model bunny("../models/texture_cat.obj", "../texture_images/texture_cat.png");

    //models.push_back(left);
    //models.push_back(right);
    //models.push_back(light);
    //models.push_back(back);
    //models.push_back(bot);
    //models.push_back(top);
    models.push_back(bunny);

    checkCudaErrors(cudaMalloc((void**)&d_models, models.size() * sizeof(Model)));
    checkCudaErrors(cudaMemcpy(d_models, models.data(), models.size() * sizeof(Model), cudaMemcpyHostToDevice));

    uint32_t num_triangles = 0;
    uint32_t num_spheres = 2;
    for (Model model : models) {
        num_triangles += model.num_triangles;
    }
    uint32_t num_hittable = num_triangles + num_spheres;

    Hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hittable * sizeof(Hittable*)));

    Material** d_materials;
    Texture** d_textures;
    checkCudaErrors(cudaMallocManaged((void**)&d_materials, models.size() * sizeof(Material*)));
    checkCudaErrors(cudaMallocManaged((void**)&d_textures, models.size() * sizeof(Texture*)));

    size_t triangle_memory_size = num_triangles * (sizeof(Triangle));
    void* d_triangle_memory;
    checkCudaErrors(cudaMalloc((void**)&d_triangle_memory, triangle_memory_size));


    build_cb_bunny<<<1, 1>>>(d_world, d_list, static_cast<char*>(d_triangle_memory), d_materials, d_textures, d_models, models.size(), num_triangles);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    construct_bvh(d_world, num_hittable);
}

void Scene::construct_bvh(Hittable** d_world, uint32_t num_hittable) {
    uint32_t num_nodes = 2 * num_hittable - 1;

    uint32_t threads = 256;
    uint32_t blocks = (num_hittable + threads - 1) / threads;

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

    checkCudaErrors(cudaFree(d_morton_codes));
    checkCudaErrors(cudaFree(d_index));
}
