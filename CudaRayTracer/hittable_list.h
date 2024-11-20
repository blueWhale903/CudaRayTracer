#pragma once

#include "hittable.h"
#include "triangle.h"
#include "node.h"

class HittableList : public Hittable {
public:
    Hittable** list;
    BVHNode* d_nodes;
    uint32_t size = 0;

    __device__ HittableList() {}
    __device__ HittableList(Hittable** list, uint32_t n, AABB scene_bbox):list(list), size(n), bbox(scene_bbox) {}
	__host__ __device__ ~HittableList() {}

	__device__ virtual bool hit(const Ray& ray, Interval ray_t, HitRecord& record) const{
        HitRecord temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;
#define BVH 1
#if BVH == 0
        for (uint32_t i = 0; i < size; i++) {
            Interval interval(ray_t.min, closest_so_far);
            if (list[i]->hit(ray, interval, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                record = temp_rec;
            }
        }
#else
        BVHNode* stack[64];
        uint32_t stack_size = 0;
        
        stack[stack_size++] = &d_nodes[size];
        
        while (stack_size > 0) {
            BVHNode* node = stack[--stack_size];
            
            float t_min = ray_t.min;
            if (!node->bbox.fastAABBIntersect(ray, Interval(ray_t.min, closest_so_far), t_min)) {
                continue;
            }

            if (node->children[0] == nullptr && node->children[1] == nullptr) {
                int id = node->id;
                if (list[id]->hit(ray, Interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    record = temp_rec;
                }
            }

            if (node->children[1]) {
				stack[stack_size++] = node->children[1];
            }
            if (node->children[0]) {
				stack[stack_size++] = node->children[0];
            }
        }
#endif
        return hit_anything;
	}

    __device__ AABB bounding_box() const override { return bbox; }

private:
    AABB bbox;
};
