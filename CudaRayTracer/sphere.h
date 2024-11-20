#pragma once

#include "hittable.h"
#include <algorithm>

#include "utility.h"

class Sphere : public Hittable {
public:
	__device__ Sphere(const vec3 center, float radius, Material* material)
		: center(center), radius(std::max(0.0f, radius)), material(material) {
		vec3 rvec(radius, radius, radius);
		bbox = AABB(center - rvec, center + rvec);
		minbbox = bbox.min();
		maxbbox = bbox.max();
	}

	__device__ __host__ ~Sphere() {
		delete material;
	}

	__device__ bool hit(const Ray& ray, Interval ray_t, HitRecord& record) const override {
		vec3 oc = center - ray.origin();

		float a = dot(ray.direction(), ray.direction());
		float b = dot(ray.direction(), oc);
		float c = dot(oc, oc) - radius * radius;
		float discriminant = b * b - a * c;

		if (discriminant < 0) {
			return false;
		}

		float sqrt_discriminant = std::sqrtf(discriminant);
		float root = (b - sqrt_discriminant) / a;
		if (!ray_t.contains(root)) {
			root = (b + sqrt_discriminant) / a;
			if (!ray_t.contains(root)) {
				return false;
			}
		}

		record.t = root;
		record.point = ray.at(root);
		vec3 outward_normal = (record.point - center) / radius;
		record.set_face_normal(ray, outward_normal);
		record.material = material;

		return true;
	}

	__device__ AABB bounding_box() const override { return bbox; }

	Material* material;
private:
	AABB bbox;
	vec3 center;
	float radius;
};
