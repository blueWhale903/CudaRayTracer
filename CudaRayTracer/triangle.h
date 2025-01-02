#pragma once

#include "hittable.h"
#include "material.h"

class Triangle : public Hittable {
public:
    __device__ Triangle() {}
    __device__ __host__ ~Triangle() {}
	__device__ Triangle(vec3 v1, vec3 v2, vec3 v3, Material* mat) 
		: a(v1), b(v2), c(v3), material(mat) {
        create_bbox();
    }

    __device__ void create_bbox() {
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

        vec2 uv = interpolate_uv(uv_a, uv_b, uv_c, u, v);

        // Compute final intersection point and normal
        record.t = t;
        record.point = ray.origin() + t * ray.direction();
        vec3 normal = glm::normalize(glm::cross(edge_ab, edge_ac));
        record.set_face_normal(ray, normal);
        record.material = material;
        record.u = uv.x;
        record.v = uv.y;

        return true;
	}

    __device__ AABB bounding_box() const override { return bbox; }

	Material* material;
	vec3 a;
	vec3 b;
	vec3 c;
	AABB bbox;
    vec2 uv_a;
    vec2 uv_b;
    vec2 uv_c;
};
