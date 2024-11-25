#include <iostream>
#include <chrono>

#include "utility.h"

#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "renderer.h"

#include "model_loader.h"

#include "scene.h"

#include <cuda.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Scene;


__global__ void free_world(Hittable** d_world, Camera** d_camera) {
    delete* d_world;
    delete* d_camera;
}

__global__ void init_camera(Camera** d_camera, uint32_t width, float aspect_ratio) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
		float vfov = 25.0f;
		vec3 look_from = vec3(8.f, 5.f, -6.0f);
		vec3 look_at = vec3(-4, 0, 4);

		float defocus_angle = 0.6f;
		float focus_distance = glm::length(look_from - vec3(0,0,0));

		*d_camera = new Camera(width, look_from, look_at, vec3(0, 1, 0),
            vfov, aspect_ratio, defocus_angle, focus_distance);
    }
}

int main()
{
    const float ASPECT_RATIO = 16.0f / 9.0f;
    uint32_t IMAGE_WIDTH = 1000;
    uint32_t IMAGE_HEIGHT = 1000;
    const uint32_t spp = 10;
    const uint32_t tx = 8;
    const uint32_t ty = 8;

    std::cerr << "Rendering a " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << " image with " << spp << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    // Initialize frame buffer
    uint32_t num_pixels = IMAGE_WIDTH * IMAGE_HEIGHT;
    size_t framebuffer_size = num_pixels * sizeof(vec3);
    vec3* framebuffer;
    checkCudaErrors(cudaMallocManaged((void**)&framebuffer, framebuffer_size));

    // Initialize camera on GPU
    Camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
    init_camera<<<1, 1>>>(d_camera, IMAGE_WIDTH, ASPECT_RATIO);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Create list of Spheres on GPU
    Hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
    Scene scene;
    //scene.launch_random_scene_kernel(d_world);
    scene.launch_triangles_scene(d_world);
    std::clog << "- Scene Created\n";

    // Create renderer
    Renderer renderer(IMAGE_WIDTH, IMAGE_HEIGHT, tx, ty, spp);
 
    // Start rendering
    clock_t start, stop;
    start = clock();
    
    renderer.launch_kernel_render(framebuffer, d_camera, d_world);

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "\nRender time: " << timer_seconds << " seconds.\n";

    // Export to png
    export_framebuffer_to_png(framebuffer, IMAGE_WIDTH, IMAGE_HEIGHT, "output.png");

    free_world<<<1, 1>>>(d_world, d_camera);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(framebuffer));

    cudaDeviceReset();

    return 0;
}
