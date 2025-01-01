#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <vector>
#include "triangle.h"

struct Vertex {
    vec3 position;
    vec3 normal;
};

bool LoadModel(const std::string& modelFilePath, std::vector<vec3>& outVertices, std::vector<uint32_t>& outIndices) {
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(modelFilePath, aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "Error: " << importer.GetErrorString() << std::endl;
        return false;
    }

    // Loop through all meshes in the scene
    for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];

        // Extract vertices and normals
        for (uint32_t j = 0; j < mesh->mNumVertices; j++) {
            aiVector3D p = mesh->mVertices[j];

            vec3 v1 = vec3(p.x, p.y, p.z);
            outVertices.push_back(v1);
        }

        for (uint32_t j = 0; j < mesh->mNumFaces; j++) {
            aiFace face = mesh->mFaces[j];

            for (uint32_t k = 0; k < face.mNumIndices; k++) {
                outIndices.push_back(face.mIndices[k]);
            }
        }
    }

    printf("Model loaded\n");

    return true;
}

vec3 convertToGLM(const aiVector3D& v) {
    return glm::vec3(v.x, v.y, v.z);
}

Material* createMaterial(aiMaterial* aiMat) {
    aiColor3D color(1.0f, 1.0f, 1.0f);
    aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color);

    return new Lambertian(glm::vec3(color.r, color.g, color.b));
}

bool loadModel2(const std::string& path, std::vector<Triangle>& triangles, std::vector<Material*>& materials) {
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(
        path,
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_GenNormals |
        aiProcess_FlipUVs
    );

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "Error loading model: " << importer.GetErrorString() << std::endl;
        return false;
    }

    // Process all materials
    for (unsigned int i = 0; i < scene->mNumMaterials; i++) {
        aiMaterial* aiMat = scene->mMaterials[i];
        Material* material = createMaterial(aiMat);
        materials.push_back(material);
    }

    // Process all meshes in the scene
    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];

        // Use the material for the current mesh
        Material* material = materials[mesh->mMaterialIndex];

        // Extract all triangles
        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            aiFace face = mesh->mFaces[j];

            if (face.mNumIndices == 3) { // Ensure it's a triangle
                glm::vec3 v1 = convertToGLM(mesh->mVertices[face.mIndices[0]]);
                glm::vec3 v2 = convertToGLM(mesh->mVertices[face.mIndices[1]]);
                glm::vec3 v3 = convertToGLM(mesh->mVertices[face.mIndices[2]]);

                // Add triangle to list
                triangles.emplace_back(v1, v2, v3, material);
            }
        }
    }

    return true;
}

//Hittable** loadModelToDevice(const std::string& path, uint32_t& numTriangles) {
//    std::vector<Triangle> hostTriangles;
//
//    // Load model using ASSIMP
//    loadModel2(path, hostTriangles);
//
//    numTriangles = hostTriangles.size();
//
//    // Allocate device memory
//    Triangle** d_triangles;
//    cudaMalloc(&d_triangles, numTriangles * sizeof(Triangle*));
//
//    // Copy triangles to the device
//    cudaMemcpy(d_triangles, hostTriangles.data(), numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
//
//    return d_triangles;
//}

void copyTrianglesToDevice(const std::vector<Triangle>& triangles, Hittable*** d_list) {
    uint32_t d_list_size = triangles.size();

    // Allocate memory for the device array of Hittable* pointers
    checkCudaErrors(cudaMalloc((void**)d_list, d_list_size * sizeof(Hittable*)));

    // Loop through each triangle, allocate memory for it on the device, and copy it
    for (size_t i = 0; i < d_list_size; i++) {
        Triangle* d_triangle;
        checkCudaErrors(cudaMalloc((void**)&d_triangle, sizeof(Triangle)));
        checkCudaErrors(cudaMemcpy(d_triangle, &triangles[i], sizeof(Triangle), cudaMemcpyHostToDevice));

        // Set the pointer in d_list
        checkCudaErrors(cudaMemcpy(*d_list + i, &d_triangle, sizeof(Hittable*), cudaMemcpyHostToDevice));
    }
}
