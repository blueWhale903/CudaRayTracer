#pragma once

#include "aabb.h"

struct BVHNode {
	AABB bbox;
	BVHNode* children[2];
	int id;
	bool visited = false;
};
