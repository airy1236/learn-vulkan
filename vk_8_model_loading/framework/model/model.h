#pragma once

#include <iostream>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include <assimp/texture.h>

#include "../../renderDataStruct.h"

struct MaterialInfo {
	unsigned int diffuseMaps;
};

class Mesh {
public:
	struct MeshInfo {
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
	};

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	uint32_t materialIndex;

	Mesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices, uint32_t materialIndex) {
		this->vertices = vertices;
		this->indices = indices;
		this->materialIndex = materialIndex;
	}
};

class Model {
public:
	Model(std::string path);
	std::vector<Mesh> meshes;
	std::vector<MaterialInfo> materials;
	std::vector<Mesh::MeshInfo> meshInfo;
	std::vector<std::string> texturePath;

private:
	std::string directory;

	void ProcessNode(const aiScene* scene, aiNode* node);
	Mesh ProcessMesh(const aiScene* scene, aiMesh* mesh);
	uint32_t SetupMaterial(std::vector<uint32_t> diffuseMaps);
	void SetupRenderInfo();
	std::vector<uint32_t> LoadMaterialTextures(aiMaterial* mat, aiTextureType type);
};

bool CompareMaterial(MaterialInfo dest, MaterialInfo source);