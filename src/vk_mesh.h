#pragma once

#include <vk_types.h>
#include <vector>
#include <glm/vec3.hpp>

struct VertexInputDescription
{
	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;

	VkPipelineVertexInputStateCreateFlags flag = 0;
};

struct  Vertex
{
	glm::vec3 positon;
	glm::vec3 normal;
	glm::vec3 color;

	static VertexInputDescription get_vertex_description();
};

struct Mesh
{
	std::vector<Vertex> _vertices;

	AllocatedBuffer _vertexBuffer;

	bool load_from_obj(const char* filename);
};