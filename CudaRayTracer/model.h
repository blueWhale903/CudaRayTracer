#pragma once

#include "triangle.h"
#include "model_loader.h"

class Model {
public:
	vec3* d_vertex;
	uint32_t* d_indices;
	vec2* d_uvs;
	Material* material;
	unsigned char* d_texture = nullptr;

	uint32_t num_triangles;
	uint32_t num_vertices;

	Model(const std::string& modelFilePath, const std::string& texturePath = "") {
		std::vector<vec3> h_vertex;
		std::vector<uint32_t> h_indices;
		std::vector<vec2> h_uvs;

		if (!LoadModel(modelFilePath, h_vertex, h_indices, h_uvs)) {
			std::cerr << "Failed to load OBJ model!" << std::endl;
			return;
		}

		if (texturePath != "") {
			unsigned char* h_texture = LoadTexture(texturePath, 2048, 2048, 3);
			size_t texture_size = 2048 * 2048 * 3;
			checkCudaErrors(cudaMalloc((void**)&d_texture, texture_size));
			checkCudaErrors(cudaMemcpy(d_texture, h_texture, texture_size, cudaMemcpyHostToDevice));
			free(h_texture);  // Free host memory
		}

		num_triangles = h_indices.size() / 3;
		num_vertices = h_vertex.size();
	
		uint32_t num_indices = h_indices.size();
		uint32_t num_uvs = h_uvs.size();
			
		checkCudaErrors(cudaMalloc((void**)&d_vertex, num_vertices * sizeof(vec3)));
		checkCudaErrors(cudaMalloc((void**)&d_indices, num_indices * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc((void**)&d_uvs, num_uvs * sizeof(vec2)));

		checkCudaErrors(cudaMemcpy(d_vertex, h_vertex.data(), num_vertices * sizeof(vec3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_indices, h_indices.data(), num_indices * sizeof(uint32_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_uvs, h_uvs.data(), num_uvs * sizeof(vec2), cudaMemcpyHostToDevice));
	}
};