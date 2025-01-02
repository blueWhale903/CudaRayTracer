#pragma once

#include "utility.h"

class Texture {
public:
	virtual ~Texture() = default;

	__device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class SolidColor : public Texture {
public:
	__device__ __host__ SolidColor(const vec3& albedo) : albedo(albedo) {}

	__device__ vec3 value(float u, float v, const vec3& p) const override {
		return albedo;
	}
private:
	vec3 albedo;
};

class CheckerTexture : public Texture {
public:
	__device__ __host__ CheckerTexture(float scale, Texture* even, Texture* odd)
		: inv_scale(1.f / scale), even(even), odd(odd) {}

	__device__ __host__ CheckerTexture(float scale, const vec3& c1, const vec3& c2)
		: CheckerTexture(scale, new SolidColor(c1), new SolidColor(c2)) {}

	__device__ vec3 value(float u, float v, const vec3& p) const override {
		auto x_integer = int(floorf(inv_scale * p.x));
		auto y_integer = int(floorf(inv_scale * p.y));
		auto z_integer = int(floorf(inv_scale * p.z));

		bool is_even = (x_integer + y_integer + z_integer) % 2;

		return is_even ? even->value(u, v, p) : odd->value(u, v, p);
	}
private:
	float inv_scale;
	Texture* even;
	Texture* odd;
};

class ImageTexture : public Texture {
public:
	__device__ __host__ ImageTexture(const unsigned char* pixels, int width, int height)
		: data(pixels), width(width), height(height) {
	}

	__device__ vec3 value(float u, float v, const vec3& p) const override {
		// Clamp UV coordinates
		u = u - floorf(u);
		v = v - floorf(v);

		int i = static_cast<int>(u * (width - 1));
		int j = static_cast<int>((1.0f - v) * (height - 1));

		// Fetch pixel data
		auto pixel_index = 3 * (j * width + i);

		auto r = data[pixel_index + 0] / 255.0f;
		auto g = data[pixel_index + 1] / 255.0f;
		auto b = data[pixel_index + 2] / 255.0f;

		return vec3(r, g, b);
	}

private:
	const unsigned char* data;
	int width, height;
};