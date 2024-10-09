#pragma once

#include <glm/glm.hpp>

using glm::vec3;

__device__ float linear_to_gamma(double linear_component) {
    if (linear_component > 0) {
        return std::sqrtf(linear_component);
    }

    return 0;
}

__device__ void write_color(unsigned char image[], int index, const vec3 pixel_color) {
    auto r = pixel_color.x;
    auto g = pixel_color.y;
    auto b = pixel_color.z;

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Translate the [0,1] component values to the byte range [0,255].
    int rbyte = int(256 * glm::clamp(r, 0.0f, 0.999f));
    int gbyte = int(256 * glm::clamp(g, 0.0f, 0.999f));
    int bbyte = int(256 * glm::clamp(b, 0.0f, 0.999f));

    image[index + 0] = static_cast<unsigned char>(rbyte);
    image[index + 1] = static_cast<unsigned char>(gbyte);
    image[index + 2] = static_cast<unsigned char>(bbyte);
}
