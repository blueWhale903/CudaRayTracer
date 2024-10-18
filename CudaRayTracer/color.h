#pragma once

#include <glm/glm.hpp>

__device__ static float linear_to_gamma(double linear_component) {
    if (linear_component > 0) {
        return std::sqrtf(linear_component);
    }

    return 0;
}

__device__ static void write_color(vec3* framebuffer, uint32_t index, const vec3 pixel_color) {
    auto r = pixel_color.x;
    auto g = pixel_color.y;
    auto b = pixel_color.z;

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);


    framebuffer[index] = vec3(r,g,b);
}
