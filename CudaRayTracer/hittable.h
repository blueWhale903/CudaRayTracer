#pragma once

#include "aabb.h"
#include "utility.h"

class Material;

class HitRecord {
public:
	vec3 point;
	vec3 normal;
	Material* material;
	float t;
	bool front_face;

	__device__ void set_face_normal(const Ray& ray, const vec3& outward_normal) {
		front_face = glm::dot(ray.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class Hittable {
public:
	vec3 minbbox = vec3(0, 0, 0);
	vec3 maxbbox = vec3(0, 0, 0);
	__device__ virtual bool hit(const Ray& r, Interval ray_t, HitRecord& record) const = 0;

	__device__ virtual AABB bounding_box() const = 0;
};

