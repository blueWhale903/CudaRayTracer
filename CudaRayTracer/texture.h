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
