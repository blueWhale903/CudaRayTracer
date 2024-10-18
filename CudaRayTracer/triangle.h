#pragma once

#include "hittable.h"
#include "material.h"

class Triangle : public Hittable {
public:
	__device__ __host__ Triangle() {}
	__device__ __host__ ~Triangle() {
		delete material;
	}
	__device__ __host__ Triangle(vec3 v1, vec3 v2, vec3 v3, vec3 n1, vec3 n2, vec3 n3) : a(v1), b(v2), c(v3), normal_a(n1), normal_b(n2), normal_c(n3) {
		material = new Metal(vec3(0.7, 0.6, 0.5), 0.0);
	}
		
	__device__ bool hit(const Ray& ray, Interval ray_t, HitRecord& record) const override {
		vec3 edge_ab = b - a;
		vec3 edge_ac = c - a;

		vec3 normal = glm::cross(edge_ab, edge_ac);

		vec3 ao = ray.origin() - a;
		vec3 dao = glm::cross(ao, ray.direction());

		const double epsilon = 1e-8;

		double determinant = -glm::dot(ray.direction(), normal);
		if (fabs(determinant) < epsilon) {
			return false;
		}
		double invert_determinant = 1 / determinant;

		float t = glm::dot(ao, normal) * invert_determinant;
		if (t < ray_t.min || t > ray_t.max) {
			return false;
		}

		float u = glm::dot(edge_ac, dao) * invert_determinant;
		float v = -glm::dot(edge_ab, dao) * invert_determinant;
		float w = 1 - u - v;

		if (determinant > epsilon && t >= 0 && u >= 0 && v >= 0 && w >= 0) {
			record.point = ray.origin() + ray.direction() * t;
			record.t = t;
			record.normal = glm::normalize(normal_a * w + normal_b * u + normal_c * v);
			record.material = material;
			return true;
		}

		return false;
	}

	AABB bounding_box() const override { return bbox; }

	Material* material;
	vec3 a;
	vec3 b;
	vec3 c;
	vec3 normal_a;
	vec3 normal_b;
	vec3 normal_c;
	AABB bbox;
};
