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

bool LoadModel(const std::string& modelFilePath, std::vector<Vertex>& outVertices, std::vector<uint32_t>& outIndices) {
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(modelFilePath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "Error: " << importer.GetErrorString() << std::endl;
        return false;
    }

    // Loop through all meshes in the scene
    for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];

        // Extract vertices and normals
        for (uint32_t j = 0; j < mesh->mNumVertices; j++) {
            Vertex vertex;
            aiVector3D p = mesh->mVertices[j];
            aiVector3D n = mesh->mNormals[j];

            vec3 v1 = vec3(p.x, p.y, p.z);
            vec3 n1 = vec3(n.x, n.y, n.z);

            vertex.position = v1;   // Get vertex position
            vertex.normal = n1;      // Get vertex normal
            outVertices.push_back(vertex);
        }

        for (uint32_t j = 0; j < mesh->mNumFaces; j++) {
            aiFace face = mesh->mFaces[j];

            for (uint32_t k = 0; k < face.mNumIndices; k++) {
                outIndices.push_back(face.mIndices[k]);
            }
        }
    }

    return true;
}