#pragma once

#include "interval.h"

class AABB {
public:
	Interval x, y, z;

	__device__ AABB() {}
	__device__ AABB(const Interval x, const Interval& y, const Interval& z): x(x), y(y), z(z) {}
	__device__ AABB(const vec3& point_a, const vec3& point_b) {
		x = point_a.x < point_b.x ? Interval(point_a.x, point_b.x) : Interval(point_b.x, point_a.x);
		y = point_a.y < point_b.y ? Interval(point_a.y, point_b.y) : Interval(point_b.y, point_a.y);
		z = point_a.z < point_b.z ? Interval(point_a.z, point_b.z) : Interval(point_b.z, point_a.z);
	}
	__device__ AABB(const AABB& bbox0, const AABB& bbox1) {
		x = Interval(bbox0.x, bbox1.x);
		y = Interval(bbox0.y, bbox1.y);
		z = Interval(bbox0.z, bbox1.z);
	}

	__device__ Interval get_axis(uint8_t i) {
		if (i == 1) return y;
		if (i == 2) return z;

		return x;
	}

	__device__ bool hit(const Ray& ray, Interval ray_t) {
		vec3 ray_origin = ray.origin();
		vec3 ray_dir = ray.direction();

		for (uint8_t axis; axis < 3; axis++) {
			const Interval& ax = get_axis(axis);
			const double adinv = 1.0 / ray_dir[axis];

			auto t0 = (ax.min - ray_origin[axis]) * adinv;
			auto t1 = (ax.max - ray_origin[axis]) * adinv;

			if (t0 < t1) {
				if (t0 > ray_t.min) ray_t.min = t0;
				if (t1 < ray_t.max) ray_t.max = t1;
			}
			else {
				if (t1 > ray_t.min) ray_t.min = t1;
				if (t0 < ray_t.max) ray_t.max = t0;
			}

			if (ray_t.max <= ray_t.min)
				return false;
		}

		return true;
	}

};
