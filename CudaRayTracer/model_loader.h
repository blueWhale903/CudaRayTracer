#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <functional>
#include <vector>
#include "triangle.h"

struct Vertex {
    vec3 position;
    vec3 normal;
};

bool LoadModel(const std::string& modelFilePath, std::vector<vec3>& outVertices,
    std::vector<uint32_t>& outIndices, std::vector<vec2>& outUVs) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(modelFilePath, aiProcess_JoinIdenticalVertices |
        aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "Error: " << importer.GetErrorString() << std::endl;
        return false;
    }

    // Process all nodes recursively
    std::function<void(aiNode*)> processNode = [&](aiNode* node) {
        // Process all meshes in current node
        for (uint32_t i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            uint32_t indexOffset = outVertices.size();

            // Extract vertices and UVs
            for (uint32_t j = 0; j < mesh->mNumVertices; j++) {
                aiVector3D p = mesh->mVertices[j];
                outVertices.push_back(vec3(p.x, p.y, p.z));

                if (mesh->HasTextureCoords(0)) {
                    aiVector3D uv = mesh->mTextureCoords[0][j];
                    outUVs.emplace_back(vec2(uv.x, uv.y));
                }
                else {
                    outUVs.emplace_back(vec2(0.0f, 0.0f));
                }
            }

            // Extract faces
            for (uint32_t j = 0; j < mesh->mNumFaces; j++) {
                aiFace face = mesh->mFaces[j];
                for (uint32_t k = 0; k < face.mNumIndices; k++) {
                    outIndices.push_back(indexOffset + face.mIndices[k]);
                }
            }
        }

        // Process children nodes
        for (uint32_t i = 0; i < node->mNumChildren; i++) {
            processNode(node->mChildren[i]);
        }
        };

    processNode(scene->mRootNode);
    return true;
}

