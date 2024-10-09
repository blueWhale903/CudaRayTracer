#pragma once

#include "hittable.h"
#include <thrust/device_vector.h>

class HittableList : public Hittable {
public:
    Hittable** list;
    int size = 0;

    __host__ __device__ HittableList() {}
	__device__ HittableList(Hittable** list, int n):list(list), size(n) {}
	__host__ __device__ ~HittableList() {}

 //   __device__ void clear() { objects.clear(); size = 0; }
	//__device__ void add(Hittable* object) { 
 //       objects.push_back(object); 
 //       size++;
 //   }

	__device__ virtual bool hit(const Ray& ray, Interval ray_t, HitRecord& record) const{
        HitRecord temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < size; i++) {
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
