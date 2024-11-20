#pragma once

#include "hittable.h"
#include "material.h"

class Triangle : public Hittable {
public:
	__device__ Triangle() {}
	__device__ __host__ ~Triangle() {
		delete material;
	}
	__device__ Triangle(vec3 v1, vec3 v2, vec3 v3, vec3 n1, vec3 n2, vec3 n3, Material* mat) 
		: a(v1), b(v2), c(v3), normal_a(n1), normal_b(n2), normal_c(n3), material(mat) {
		float min_x = std::fminf(std::fminf(v1.x, v2.x), v3.x);
		float min_y = std::fminf(std::fminf(v1.y, v2.y), v3.y);
		float min_z = std::fminf(std::fminf(v1.z, v2.z), v3.z);
		float max_x = std::fmaxf(std::fmaxf(v1.x, v2.x), v3.x);
		float max_y = std::fmaxf(std::fmaxf(v1.y, v2.y), v3.y);
		float max_z = std::fmaxf(std::fmaxf(v1.z, v2.z), v3.z);

		float epsilon = 1e-4;
        bbox = AABB(vec3(min_x, min_y, min_z) - vec3(epsilon), vec3(max_x, max_y, max_z) + vec3(epsilon));
        pad_to_minimums();
		minbbox = bbox.min();
		maxbbox = bbox.max();
	}

    __device__ void init() {
        const float EPSILON = 1e-4;
        bbox = AABB(
            vec3(
                min({ a.x, b.x, c.x }) - EPSILON,
                min({ a.y, b.y, c.y }) - EPSILON,
                min({ a.z, b.z, c.z }) - EPSILON
            ),
            vec3(
                max({ a.x, b.x, c.x }) + EPSILON,
                max({ a.y, b.y, c.y }) + EPSILON,
                max({ a.z, b.z, c.z }) + EPSILON
            )
        );

        // Ensure consistent AABB calculation
        minbbox = bbox.min();
        maxbbox = bbox.max();
    }
		
	__device__ bool hit(const Ray& ray, Interval ray_t, HitRecord& record) const override {
        vec3 edge_ab = b - a;
        vec3 edge_ac = c - a;
        vec3 pvec = glm::cross(ray.direction(), edge_ac);
        float det = glm::dot(edge_ab, pvec);

        const float epsilon = 1e-7f;  // Smaller epsilon for more precision

        // If determinant is near zero, ray lies in plane of triangle
        if (fabs(det) < epsilon)
            return false;

        float inv_det = 1.0f / det;
        vec3 tvec = ray.origin() - a;
        float u = glm::dot(tvec, pvec) * inv_det;

        if (u < -epsilon || u > 1.0f + epsilon)
            return false;

        vec3 qvec = glm::cross(tvec, edge_ab);
        float v = glm::dot(ray.direction(), qvec) * inv_det;

        if (v < -epsilon || u + v > 1.0f + epsilon)
            return false;

        float t = glm::dot(edge_ac, qvec) * inv_det;

        if (t < ray_t.min - epsilon || t > ray_t.max + epsilon)
            return false;

        // Compute final intersection point and normal
        record.t = t;
        record.point = ray.origin() + t * ray.direction();
        vec3 normal = glm::normalize(glm::cross(edge_ab, edge_ac));
        record.set_face_normal(ray, normal);
        record.material = material;

        return true;
	}
    __device__ void pad_to_minimums() {
        // Adjust the AABB so that no side is narrower than some delta, padding if necessary.

        double delta = 0.0001;
        if (bbox.x.size() < delta) bbox.x = bbox.x.expand(delta);
        if (bbox.y.size() < delta) bbox.y = bbox.y.expand(delta);
        if (bbox.z.size() < delta) bbox.z = bbox.z.expand(delta);
    }
	__device__ AABB bounding_box() const override { return bbox; }

	Material* material;
	vec3 a;
	vec3 b;
	vec3 c;
	vec3 normal_a;
	vec3 normal_b;
	vec3 normal_c;
	AABB bbox;
};
