#pragma once

#include <glm/glm.hpp>

#include "hittable.h"
#include "hittable_list.h"
#include "material.h"

#include <curand_kernel.h>

class Camera {
public:
	float aspect_ratio = 16.0f / 9.0f;
	float vfov = 20;
	uint32_t image_width = 1280;
	vec3 look_from = vec3(13.0f,2.0f,3.0f);
	vec3 look_at = vec3(0, 0, 0);
	vec3 vup = vec3(0, 1, 0);

	float defocus_angle = 0.6f;
	float focus_distance = 10.0f;

	__host__ __device__ Camera(uint32_t width, vec3& lookfrom, vec3& lookat, vec3& vup, float vfov, float aspect, float defocus_angle, float focus_dist) {
		image_width = width;
		image_height = (uint32_t)(image_width / aspect_ratio);
		image_height = (image_height < 1) ? 1 : image_height;

		camera_center = lookfrom;
		look_from = lookfrom;

		//float focal_length = glm::length(look_from - look_at);
		float theta = degrees_to_radians(vfov);
		float h = std::tanf(theta / 2.0f);

		const float VIEWPORT_HEIGHT = 2.0f * h * focus_distance;
		const float VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (static_cast<float>(image_width) / image_height);

		w = glm::normalize(look_from - look_at);
		u = glm::normalize(glm::cross(vup, w));
		v = glm::cross(w, u);

		vec3 viewport_horizontal = VIEWPORT_WIDTH * u;
		vec3 viewport_vertical = VIEWPORT_HEIGHT * -v;

		pixel_delta_horizontal = viewport_horizontal / (float)image_width;
		pixel_delta_vertical = viewport_vertical / (float)image_height;

		vec3 viewport_upper_left = camera_center -
			(focus_distance * w) - (0.5f * viewport_horizontal) - (0.5f * viewport_vertical);
		zeroth_pixel_location = viewport_upper_left + 0.5f * (pixel_delta_horizontal + pixel_delta_vertical);

		float defocus_radius = focus_distance * std::tanf(degrees_to_radians(defocus_angle / 2.0f));
		defocus_disk_u = u * defocus_radius;
		defocus_disk_v = v * defocus_radius;
	}

	__device__ vec3 ray_color(const Ray& ray, Hittable** world, curandState* local_rand_state)
	{
		Ray cur_ray = ray;
		vec3 cur_attenuation(1.0f, 1.0f, 1.0f);

		for (uint32_t i = 0; i < 50; i++) {
			HitRecord rec;
			
			if ((*world)->hit(cur_ray, Interval(0.001f, FLT_MAX), rec)) {
				Ray scattered;
				vec3 attenuation;
				if (rec.material->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
					cur_attenuation *= attenuation;
					cur_ray = scattered;
				}
				else {
					return vec3(0, 0, 0);
				}
			}
			else {
				vec3 unit_direction = glm::normalize(cur_ray.direction());
				float t = 0.5f * (unit_direction.y + 1.0f);
				vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
				return cur_attenuation * c;
			}
		}

		return vec3(0.0f, 0.0f, 0.0f);
	}
	__device__ Ray get_ray(float x, float y, curandState* local_rand_state) {
		glm::vec2 offset = sample_square(local_rand_state);
		vec3 pixel_center = zeroth_pixel_location +
			((offset.x + x) * pixel_delta_horizontal) +
			((offset.y + y) * pixel_delta_vertical);

		vec3 ray_origin = defocus_angle <= 0.0f ? camera_center : defocus_disk_sample(local_rand_state);
		vec3 direction = pixel_center - ray_origin;

		return Ray(ray_origin, direction);
	}

private:
	uint32_t image_height;
	vec3 camera_center;
	vec3 zeroth_pixel_location;
	vec3 pixel_delta_horizontal;
	vec3 pixel_delta_vertical;
	vec3 u, v, w;

	vec3 defocus_disk_u;
	vec3 defocus_disk_v;

	__device__ glm::vec2 sample_square(curandState* local_rand_state) {
		return glm::vec2(random_float(local_rand_state, -0.5f, 0.5f), random_float(local_rand_state, -0.5f, 0.5f));
	}

	__device__ vec3 defocus_disk_sample(curandState* local_rand_state) const {
		// Returns a random point in the camera defocus disk.
		vec3 p = random_in_unit_disk(local_rand_state);
		return camera_center + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
	}
};
