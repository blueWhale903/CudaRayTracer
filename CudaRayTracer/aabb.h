#pragma once

#include "interval.h"
#include "utility.h"

class AABB {
public:
    Interval x, y, z;

    __device__ AABB() = default;

    __device__ AABB(const Interval& x, const Interval& y, const Interval& z)
        : x(x), y(y), z(z) {}

    __device__ AABB(const vec3& point_a, const vec3& point_b)
        : x(Interval(fminf(point_a.x, point_b.x), fmaxf(point_a.x, point_b.x))),
        y(Interval(fminf(point_a.y, point_b.y), fmaxf(point_a.y, point_b.y))),
        z(Interval(fminf(point_a.z, point_b.z), fmaxf(point_a.z, point_b.z))) {
        minbbox = vec3(x.min, y.min, z.min);
        maxbbox = vec3(x.max, y.max, z.max);
    }

    __device__ AABB(const AABB& bbox0, const AABB& bbox1)
        : x(Interval(bbox0.x, bbox1.x)),
        y(Interval(bbox0.y, bbox1.y)),
        z(Interval(bbox0.z, bbox1.z)) {
        minbbox = vec3(x.min, y.min, z.min);
        maxbbox = vec3(x.max, y.max, z.max);
    }

    __device__ __forceinline__ Interval get_axis(uint8_t i) const {
        return (i == 1) ? y : (i == 2) ? z : x;
    }

    __device__ __forceinline__ vec3 min() const {
        return minbbox;
    }

    __device__ __forceinline__ vec3 max() const {
        return maxbbox;
    }

    __device__ __forceinline__ vec3 centroid() const {
        return (minbbox + maxbbox) * 0.5f;
    }

    __device__ bool hit(const Ray& ray, Interval ray_t) const {
        const vec3 ray_origin = ray.origin();
        const vec3 inv_dir = 1.0f / ray.direction();

        vec3 t0 = (minbbox - ray_origin) * inv_dir;
        vec3 t1 = (maxbbox - ray_origin) * inv_dir;

        vec3 tNear = glm::min(t0, t1) - EPSILON;
        vec3 tFar = glm::max(t0, t1) + EPSILON;

        ray_t.min = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
        ray_t.max = fminf(fminf(tFar.x, tFar.y), tFar.z);

        //t_min = ray_t.min;
        return ray_t.min <= ray_t.max && ray_t.max > 0.0f;
    }

    __device__ __forceinline__ bool fastAABBIntersect(const Ray& ray, float& t_min) const {
        const vec3 ray_origin = ray.origin();
        const vec3 inv_dir = 1.0f / ray.direction();

        vec3 t0 = (minbbox - ray_origin) * inv_dir;
        vec3 t1 = (maxbbox - ray_origin) * inv_dir;

        vec3 tNear = min_vec(t0, t1) - EPSILON;
        vec3 tFar = max_vec(t0, t1) + EPSILON;

        float min_t = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
        float max_t = fminf(fminf(tFar.x, tFar.y), tFar.z);

        t_min = min_t;
        return min_t <= max_t && max_t > 0.0f;
    }

private:
    vec3 minbbox, maxbbox;
    static constexpr float EPSILON = 1e-7f;
};
