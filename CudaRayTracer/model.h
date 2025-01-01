#pragma once

#include "triangle.h"
#include "model_loader.h"

class Model {
public:
	vec3* d_vertex;
	uint32_t* d_indices;
	Material* material;
	uint32_t num_triangles;
	uint32_t num_vertices;
	Model(const std::string& modelFilePath) {
		std::vector<vec3> h_vertex;
		std::vector<uint32_t> h_indices;

		if (!LoadModel(modelFilePath, h_vertex, h_indices)) {
			std::cerr << "Failed to load OBJ model!" << std::endl;
			return;
		}

		num_triangles = h_indices.size() / 3;
		num_vertices = h_vertex.size();
			
		checkCudaErrors(cudaMalloc((void**)&d_vertex, h_vertex.size() * sizeof(vec3)));
		checkCudaErrors(cudaMalloc((void**)&d_indices, h_indices.size() * sizeof(uint32_t)));

		checkCudaErrors(cudaMemcpy(d_vertex, h_vertex.data(), h_vertex.size() * sizeof(vec3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));;
	}
};