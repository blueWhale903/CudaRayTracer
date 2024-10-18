#pragma once

#include "hittable.h"
#include "triangle.h"

class HittableList : public Hittable {
public:
    Hittable** list;
    uint32_t size = 0;

    __host__ __device__ HittableList() {}
    __host__ __device__ HittableList(Hittable** list, uint32_t n):list(list), size(n) {}
	__host__ __device__ ~HittableList() {}

	__device__ virtual bool hit(const Ray& ray, Interval ray_t, HitRecord& record) const{
        HitRecord temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;
        for (uint32_t i = 0; i < size; i++) {
            Interval interval(ray_t.min, closest_so_far);
            if (list[i]->hit(ray, interval, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                record = temp_rec;
            }
        }

        return hit_anything;
	}

    AABB bounding_box() const override { return bbox; }

private:
    AABB bbox;
};
