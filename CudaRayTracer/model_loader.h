//#pragma once
//
//#include <iostream>
//
//#include <assimp/Importer.hpp>
//#include <assimp/scene.h>
//#include <assimp/postprocess.h>
//
//#include <vector>
//#include <string>
//
//#include "triangle.h"
//
//std::vector<Triangle> load(std::string path) {
//	Assimp::Importer importer;
//    std::vector<Triangle> triangles = {};
//
//	const aiScene* scene = importer.ReadFile(path,
//		aiProcess_Triangulate | aiProcess_FlipUVs);
//
//	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
//		std::cerr << "Error::ASSIMP::" << importer.GetErrorString() << std::endl;
//        return {};
//	}
//
//    int n = 0;
//    for (int i = 0; i < scene->mNumMeshes; i++) {
//        aiMesh* mesh = scene->mMeshes[i];
//
//        n += mesh->mNumFaces;
//    }
//
//    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
//        aiMesh* mesh = scene->mMeshes[i];
//
//        // Loop over each face (after triangulation, all faces will be triangles)
//        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
//            aiFace face = mesh->mFaces[j];
//
//            // Ensure that the face is a triangle (this should always be true due to aiProcess_Triangulate)
//            if (face.mNumIndices == 3) {
//                aiVector3D vertex1 = mesh->mVertices[face.mIndices[0]];
//                aiVector3D vertex2 = mesh->mVertices[face.mIndices[1]];
//                aiVector3D vertex3 = mesh->mVertices[face.mIndices[2]];
//                
//                aiVector3D normal_a = mesh->mNormals[face.mIndices[0]];
//                aiVector3D normal_b = mesh->mNormals[face.mIndices[0]];
//                aiVector3D normal_c = mesh->mNormals[face.mIndices[0]];
//
//                glm::vec3 v1 = glm::vec3(vertex1.x, vertex1.y, vertex1.z);
//                glm::vec3 v2 = glm::vec3(vertex2.x, vertex2.y, vertex2.z);
//                glm::vec3 v3 = glm::vec3(vertex3.x, vertex3.y, vertex3.z);
//
//                glm::vec3 n1 = glm::vec3(normal_a.x, normal_a.y, normal_a.z);
//                glm::vec3 n2 = glm::vec3(normal_b.x, normal_b.y, normal_b.z);
//                glm::vec3 n3 = glm::vec3(normal_c.x, normal_c.y, normal_c.z);
//
//                // Store the triangle
//                Triangle triangle = Triangle(v1, v2, v3, n1, n2, n3);
//                triangles.push_back(triangle);
//            }
//        }
//    }
//    return triangles;
//}
