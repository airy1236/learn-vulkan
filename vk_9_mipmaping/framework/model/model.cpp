#include "model.h"

Model::Model(std::string path) {
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
	directory = path.substr(0, path.find_last_of('/'));
	ProcessNode(scene, scene->mRootNode);
	SetupRenderInfo();
}

void Model::ProcessNode(const aiScene* scene, aiNode* node) {
	for (size_t i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		meshes.push_back(ProcessMesh(scene, mesh));
	}

	for (size_t i = 0; i < node->mNumChildren; i++)
	{
		ProcessNode(scene, node->mChildren[i]);
	}
}

Mesh Model::ProcessMesh(const aiScene* scene, aiMesh* mesh) {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	for (size_t i = 0; i < mesh->mNumVertices; ++i) {
		Vertex vertex;
		vertex.Position.x = mesh->mVertices[i].x;
		vertex.Position.y = mesh->mVertices[i].y;
		vertex.Position.z = mesh->mVertices[i].z;

		vertex.Normal.x = mesh->mNormals[i].x;
		vertex.Normal.y = mesh->mNormals[i].y;
		vertex.Normal.z = mesh->mNormals[i].z;

		vertex.Tangent.x = mesh->mTangents[i].x;
		vertex.Tangent.y = mesh->mTangents[i].y;
		vertex.Tangent.z = mesh->mTangents[i].z;

		if (mesh->mTextureCoords[0]) {
			vertex.TexCoords.x = mesh->mTextureCoords[0][i].x;
			vertex.TexCoords.y = mesh->mTextureCoords[0][i].y;
		}
		else {
			vertex.TexCoords = { 0.0f, 0.0f };
		}

		vertex.Color = { 1.0, 0.2, 0.3 };

		vertices.push_back(vertex);
	}

	for (size_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (size_t j = 0; j < face.mNumIndices; j++)
			indices.push_back(face.mIndices[j]);
	}

	std::vector<uint32_t> diffuseMaps;
	if (mesh->mMaterialIndex >= 0) {
		aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
		diffuseMaps = LoadMaterialTextures(material, aiTextureType_DIFFUSE);
	}
	uint32_t materialIndex = SetupMaterial(diffuseMaps);

	return Mesh(vertices, indices, materialIndex);
}

std::vector<uint32_t> Model::LoadMaterialTextures(aiMaterial* mat, aiTextureType type) {
	std::vector<uint32_t> textures;
	for (uint32_t i = 0; i < mat->GetTextureCount(type); i++)
	{
		aiString str;
		mat->GetTexture(type, 0, &str);
		bool skip = false;
		std::string texturePath = directory + '/' + str.C_Str();

		for (uint32_t j = 0; j < this->texturePath.size(); j++)
		{
			if (this->texturePath[j] == texturePath) {
				textures.push_back(j);
				skip = true;
				break;
			}
		}
		if (!skip) {
			textures.push_back(this->texturePath.size());
			this->texturePath.push_back(texturePath);
		}
	}
	return textures;
}

uint32_t Model::SetupMaterial(std::vector<uint32_t> diffuseMaps) {
	MaterialInfo material;
	if (diffuseMaps.size() > 0)
		material.diffuseMaps = diffuseMaps[0];
	else
		material.diffuseMaps = 0;

	uint32_t materialIndex;
	bool skip = false;
	for (uint32_t i = 0; i < materials.size(); i++) {
		if (CompareMaterial(materials[i], material)) {
			materialIndex = i;
			skip = true;
			break;
		}
	}
	if (!skip) {
		materialIndex = materials.size();
		materials.push_back(material);
	}
	return materialIndex;
}

void Model::SetupRenderInfo() {
	meshInfo.resize(materials.size());

	for (uint32_t i = 0; i < meshes.size(); i++)
	{
		uint32_t index = meshes[i].materialIndex;
		uint32_t indexOffset = meshInfo[index].vertices.size();

		for (uint32_t j = 0; j < meshes[i].indices.size(); j++)
			meshInfo[index].indices.push_back(meshes[i].indices[j] + indexOffset);

		meshInfo[index].vertices.insert(meshInfo[index].vertices.end(), meshes[i].vertices.begin(), meshes[i].vertices.end());
	}
}

bool CompareMaterial(MaterialInfo dest, MaterialInfo source) {
	bool isSame = false;
	if (dest.diffuseMaps == source.diffuseMaps)
		isSame = true;
	else
		isSame = false;
	return isSame;
}