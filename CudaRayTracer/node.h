#pragma once

#include "aabb.h"

struct alignas(32) BVHNode {
	AABB bbox;
	int left;
	int right;
	int id;
};
