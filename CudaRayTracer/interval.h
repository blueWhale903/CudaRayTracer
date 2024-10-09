#pragma once

#include "utility.h"

class Interval {
public:
	float min, max;

	 __device__ Interval() : min(+FLT_MAX), max(-FLT_MAX) {}
	 __device__ Interval(float min, float max) : min(min), max(max) {}
	 __device__ Interval(const Interval& a, const Interval& b) {
		 min = a.min <= b.min ? min = a.min : min = b.min;
		 max = a.max >= b.max ? max = a.max : max = b.max;
	 }
	 __device__ inline float size() const {
		return max - min;
	}

	 __device__ inline bool contains(float x) const {
		return x >= min && x <= max;
	}

	 __device__ inline bool surrounds(float x) const {
		return x > min && x < max;
	}

	 __device__ Interval expand(float delta) const {
		float padding = delta / 2.0f;
		return Interval(min - padding, max + padding);
	}

	 //__device__ static const Interval empty, universe;
};

 //__device__ const Interval Interval::empty = Interval(+infinity, -infinity);
 //__device__ const Interval Interval::universe = Interval(-infinity, +infinity);

