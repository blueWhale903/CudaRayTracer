#pragma once

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"
#include "node.h"

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

/// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__forceinline__ __device__ uint32_t expand_bits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

/// Calculates a 30-bit Morton code for the given 3D point located
/// within the unit cube [0,1].
__forceinline__ __device__ uint32_t morton_3d(const vec3& pos)
{
	float x = fminf(fmaxf(pos.x * 1024.0f, 0.0f), 1023.0f);
	float y = fminf(fmaxf(pos.y * 1024.0f, 0.0f), 1023.0f);
	float z = fminf(fmaxf(pos.z * 1024.0f, 0.0f), 1023.0f);
	uint32_t xx = expand_bits((uint32_t)x);
	uint32_t yy = expand_bits((uint32_t)y);
	uint32_t zz = expand_bits((uint32_t)z);
	return xx * 4 + yy * 2 + zz;
}

__global__ void assign_morton_codes(HittableList** d_world, uint32_t* d_indices, uint32_t* d_mortons, uint32_t num_primitives) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx > num_primitives - 1) return;

	Hittable** list = (*d_world)->list;
	AABB scene_bbox = (*d_world)->bounding_box();
	AABB object_bbox = list[idx]->bounding_box();
	vec3 centroid = object_bbox.centroid();
	vec3 normalized_centroid = (centroid - scene_bbox.min()) / (scene_bbox.max() - scene_bbox.min());

	d_mortons[idx] = morton_3d(normalized_centroid);
	d_indices[idx] = idx;
}

__device__ uint32_t find_split(unsigned int* sortedMortonCodes,
	int           first,
	int           last)
{
	// Identical Morton codes => split the range in the middle.

	unsigned int firstCode = sortedMortonCodes[first];
	unsigned int lastCode = sortedMortonCodes[last];

	if (firstCode == lastCode)
		return (first + last) >> 1;

	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.

	int commonPrefix = __builtin_clz(firstCode ^ lastCode);

	// Use binary search to find where the next bit differs.
	// Specifically, we are looking for the highest object that
	// shares more than commonPrefix bits with the first one.

	int split = first; // initial guess
	int step = last - first;

	do
	{
		step = (step + 1) >> 1; // exponential decrease
		int newSplit = split + step; // proposed new position

		if (newSplit < last)
		{
			unsigned int splitCode = sortedMortonCodes[newSplit];
			int splitPrefix = __builtin_clz(firstCode ^ splitCode);
			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	} while (step > 1);

	return split;
}

__global__ void build_bvh(BVHNode* nodes, uint32_t* d_morton_codes,
						  uint32_t* d_indices, uint32_t num_primitives, HittableList** d_world) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;

	uint32_t num_nodes = 2 * num_primitives - 1;
	
	struct Range {
		uint32_t first;
		uint32_t last;
		int parent_index;
		bool is_left_child;
	};

	Range stack[32];
	int stack_size = 0;

	stack[stack_size++] = { 0, num_primitives-1, -1, true };
	
	Hittable** d_list = (*d_world)->list;
	int idx = 0;
	while (stack_size > 0) {
		Range range = stack[--stack_size];

		uint32_t first = range.first;
		uint32_t last = range.last;
		int parent_index = range.parent_index;
		bool is_left_child = range.is_left_child;

		if (last == first) {
			BVHNode* leaf = &nodes[first];
			vec3 minbbox = d_list[d_indices[first]]->minbbox;
			vec3 maxbbox = d_list[d_indices[first]]->maxbbox;

			leaf->id = d_indices[first];
			leaf->children[0] = nullptr;
			leaf->children[1] = nullptr;
			leaf->bbox = AABB(minbbox, maxbbox);

			if (parent_index != -1) {
				if (is_left_child) {
					nodes[parent_index].children[0] = leaf;
				}
				else {
					nodes[parent_index].children[1] = leaf;
				}
			}
		}
		else {
			uint32_t split = find_split(d_morton_codes, first, last);
			int internal_node_index = num_primitives + idx;
			idx++;

			BVHNode* internal_node = &nodes[internal_node_index];

			internal_node->id = -1;

			if (parent_index != -1) {
				if (is_left_child) {
					nodes[parent_index].children[0] = internal_node;
				}
				else {
					nodes[parent_index].children[1] = internal_node;
				}
			}

			stack[stack_size++] = { first, split, internal_node_index, true };
			stack[stack_size++] = { split + 1, last, internal_node_index, false };
		}
	}

	for (int i = num_primitives - 1; i >= 0; i--) {
		if (num_primitives + i >= num_nodes) continue;

		BVHNode* internal_node = &nodes[num_primitives + i];
		if (!internal_node) continue;

		if (internal_node->children[0] && internal_node->children[1]) {
			// Both children exist; combine bounding boxes
			AABB left_bbox = internal_node->children[0]->bbox;
			AABB right_bbox = internal_node->children[1]->bbox;
			internal_node->bbox = AABB(left_bbox, right_bbox);
		}
		else if (internal_node->children[0]) {
			// Only left child exists; use its bounding box
			internal_node->bbox = internal_node->children[0]->bbox;
		}
		else if (internal_node->children[1]) {
			// Only right child exists; use its bounding box
			internal_node->bbox = internal_node->children[1]->bbox;
		}
	}

	(*d_world)->d_nodes = nodes;
	printf("- BVH constructed\n");
}


