#include <iostream>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "utility.h"

#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#include "model_loader.h"

#include <cuda.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void export_framebuffer_to_png(vec3* device_framebuffer, int width, int height, const char* filename) {
    // Allocate host memory for the framebuffer
    vec3* host_framebuffer = new vec3[width * height];

    // Copy data from device to host
    cudaMemcpy(host_framebuffer, device_framebuffer, width * height * sizeof(vec3), cudaMemcpyDeviceToHost);

    // Allocate memory for the image data in the format expected by stb_image_write
    unsigned char* image_data = new unsigned char[width * height * 3];

    // Convert vec3 float data to unsigned char
    for (int i = 0; i < width * height; ++i) {
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

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Hittable** d_list, Hittable** d_world, Camera** d_camera, int width, int height, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        d_list[0] = new Sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, new Lambertian(vec3(0.5, 0.5, 0.5)));

        int i = 1;
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
        *d_world = new HittableList(d_list, 22*22+1+3);

        float aspect_ratio = 16.0f/9.0f;

        float vfov = 30.0f;
        vec3 look_from = vec3(13.0f, 2.0f, 3.0f);
        vec3 look_at = vec3(0, 0, 0);

        float defocus_angle = 0.6f;
        float focus_distance = 10.0f;

        *d_camera = new Camera(width, look_from, look_at, vec3(0,1,0), vfov, aspect_ratio, defocus_angle, focus_distance);
    }
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, Camera** cam, Hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    uint32_t pixel_index = j * max_x + i;

    curandState local_rand_state = rand_state[pixel_index];

    vec3 col(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < ns; s++) {
        Ray r = (*cam)->get_ray(i, j, &local_rand_state);
        col += (*cam)->ray_color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);

    col.x = sqrtf(col.x);
    col.y = sqrtf(col.y);
    col.z = sqrtf(col.z);

    fb[pixel_index] = col;
}

__global__ void free_world(Hittable** d_list, Hittable** d_world, Camera** d_camera) {
    for (int i = 0; i < 22*22+1+3; i++) {
        delete ((Sphere*)d_list[i])->material;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

int main()
{
    const float ASPECT_RATIO = 16.0f / 9.0f;
    int IMAGE_WIDTH = 1280;
    int IMAGE_HEIGHT = 720;
    const int spp = 10;
    const int tx = 8;
    const int ty = 8;

    std::cerr << "Rendering a " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << " image with " << spp << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = IMAGE_WIDTH * IMAGE_HEIGHT;
    size_t framebuffer_size = num_pixels * sizeof(glm::vec3);

    vec3* framebuffer;
    checkCudaErrors(cudaMallocManaged((void**)&framebuffer, framebuffer_size));
        
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Hittable** d_list;
    int num_hittables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hittables * sizeof(Hittable*)));
    Hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
    Camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, IMAGE_WIDTH, IMAGE_HEIGHT, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "- WORLD CREATED\n";

    dim3 blocks(IMAGE_WIDTH / tx + 1, IMAGE_HEIGHT / ty + 1);
    dim3 threads(tx, ty);


    clock_t start, stop;
    start = clock();
    render_init<<<blocks, threads>>>(IMAGE_WIDTH, IMAGE_HEIGHT, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "- RENDERER SUCCESSFULLY INITIALIZED\n- RENDERING...";
    render<<<blocks, threads>>>(framebuffer, IMAGE_WIDTH, IMAGE_HEIGHT, spp, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "\nRender time: " << timer_seconds << " seconds.\n";

    export_framebuffer_to_png(framebuffer, IMAGE_WIDTH, IMAGE_HEIGHT, "output.png");

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(framebuffer));

    cudaDeviceReset();

    return 0;
}

