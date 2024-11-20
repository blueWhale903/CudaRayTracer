#pragma once

#include "interval.h"
#include "utility.h"

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

	__device__ vec3 min() const {
		return vec3(x.min, y.min, z.min);
	}

	__device__ vec3 max() const {
		return vec3(x.max, y.max, z.max);
	}

	__device__ vec3 centroid() {
		return (min() + max()) * 0.5f;
	}

	__device__ bool hit(const Ray& ray, Interval ray_t) {
		vec3 ray_origin = ray.origin();
		vec3 ray_dir = ray.direction();

		for (uint8_t axis = 0; axis < 3; axis++) {
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

			ray_t.min -= 1e-7;
			ray_t.max += 1e-7;

			if (ray_t.max <= ray_t.min)
				return false;
		}

		return true;
	}

	__device__ bool fastAABBIntersect(const Ray& ray, Interval ray_t, float& t_min) {
		glm::vec3 invDir = 1.0f / ray.direction(); // Calculate the reciprocal of the direction vector
		const float eps = 1e-7;

		// Calculate t values for each axis
		glm::vec3 t0 = (min() - ray.origin()) * invDir;
		glm::vec3 t1 = (max() - ray.origin()) * invDir;

		// Ensure t0 is the entry point and t1 is the exit point
		glm::vec3 tNear = glm::min(t0, t1) - eps; // Component-wise min
		glm::vec3 tFar = glm::max(t0, t1) + eps;  // Component-wise max

		// Find the largest tNear and the smallest tFar
		ray_t.min = glm::max(glm::max(tNear.x, tNear.y), tNear.z);
		ray_t.max = glm::min(glm::min(tFar.x, tFar.y), tFar.z);

		t_min = ray_t.min;

		// Ray intersects the AABB if tMin <= tMax and tMax is positive
		return ray_t.min <= ray_t.max && ray_t.max > 0.0f;
	}
};
