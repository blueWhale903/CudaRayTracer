#pragma once

#include <curand_kernel.h>

#include "camera.h"

#define CPU 0

class Renderer {
public:
	Renderer(uint32_t width, uint32_t height, uint16_t tx, uint16_t ty, uint32_t spp);
    void launch_kernel_render(vec3* framebuffer, Camera** d_camera, Hittable** d_world);
private:
	uint32_t m_width, m_height;
	dim3 m_blocks, m_threads;
	uint32_t m_samples_per_pixel;
	curandState* m_d_rand_state;
};

__global__ void kernel_render(vec3* fb, uint32_t max_x, uint32_t max_y, uint32_t ns, Camera** cam, Hittable** world, curandState* rand_state) {
#if CPU == 0
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    uint32_t pixel_index = j * max_x + i;

    curandState local_rand_state = rand_state[pixel_index];

    vec3 col(0.0f, 0.0f, 0.0f);
    for (uint32_t s = 0; s < ns; s++) {
        Ray r = (*cam)->get_ray(i, j, &local_rand_state);
        col += (*cam)->ray_color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    write_color(fb, pixel_index, col);
#else
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    curandState local_rand_state;
    for (uint32_t x = 0; x < max_x; x++) {
        for (uint32_t y = 0; y < max_y; y++) {
            uint32_t pixel_index = y * max_x + x;
            local_rand_state = rand_state[pixel_index];
            vec3 col(0.0f, 0.0f, 0.0f);
            for (uint32_t s = 0; s < ns; s++) {
                Ray r = (*cam)->get_ray(x, y, &local_rand_state);
                col += (*cam)->ray_color(r, world, &local_rand_state);
            }
            rand_state[pixel_index] = local_rand_state;
            col /= float(ns);
            write_color(fb, pixel_index, col);
        }
    }

#endif
}

__global__ void kernel_render_init(uint32_t max_x, uint32_t max_y, curandState* rand_state) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) return;

    uint32_t pixel_index = j * max_x + i;
    curand_init(pixel_index + 2003, 0, 0, &rand_state[pixel_index]);
}

Renderer::Renderer(uint32_t width, uint32_t height, uint16_t tx, uint16_t ty, uint32_t spp) :
	m_width(width), m_height(height), m_samples_per_pixel(spp)
{
	m_threads = dim3(tx, ty);
    m_blocks = dim3((width + tx - 1) / tx, (height + ty - 1) / ty);

    uint32_t n = width * height;
	checkCudaErrors(cudaMalloc((void**)&m_d_rand_state, n * sizeof(curandState)));

	kernel_render_init<<<m_blocks, m_threads>>>(width, height, m_d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::clog << "- Renderer Initialized" << std::endl;
}

void Renderer::launch_kernel_render(vec3* framebuffer, Camera** d_camera, Hittable** d_world) {

#if CPU == 0
    std::clog << "- GPU Renderering" << std::endl;
    kernel_render<<<m_blocks, m_threads>>>(framebuffer, m_width, m_height, m_samples_per_pixel,
                                           d_camera, d_world, m_d_rand_state);
#else
    std::clog << "- CPU Renderering" << std::endl;
    kernel_render<<<1, 1>>>(framebuffer, m_width, m_height, m_samples_per_pixel,
        d_camera, d_world, m_d_rand_state);
#endif
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(m_d_rand_state));
}


