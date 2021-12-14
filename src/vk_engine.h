// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vector>
#include <deque>
#include <functional>  
#include <vk_mesh.h>
#include <glm/glm.hpp>

struct MeshPushConstants
{
	glm::vec4 data;
	glm::mat4 render_matrix;
};

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function)
	{
		deletors.push_back(function);
	}

	void flush() {
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++)
		{
			(*it)();
		}

		deletors.clear();
	}
};

struct Material
{
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct RenderObject
{
	Mesh* mesh;
	Material* material;
	glm::mat4 transformMatrix;
};

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber {0};

	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };

	// --- omitted ---
	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;

	VkSwapchainKHR _swapchain;

	VkFormat _swapchainImageFormat;

	std::vector<VkImage> _swapchainImages;

	std::vector<VkImageView> _swapchainImageViews;

	// --- vulkan command ---
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	// --- vulkan renderpass --- 
	VkRenderPass _renderPass;

	std::vector<VkFramebuffer> _framebuffers;

	// --- vulkan mainloop ---
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	// --- vulkan pipeline ---
	VkPipelineLayout _trianglePipelineLayout;
	// try to set two pipeline to switch shader
	VkPipeline _trianglePipeline;
	VkPipeline _redTrianglePipeline;

	// --- 标记变量,用来切换shader ---
	int _selectedShader{ 0 };

	// --- init delete queue ---
	DeletionQueue _mainDeletionQueue;

	// --- init vulkan memory allocator ---
	VmaAllocator _allocator;

	// --- mesh pipeline ---
	VkPipeline _meshPipeline;
	Mesh _trangleMesh;

	VkPipelineLayout _meshPipelineLayout;

	//add object mesh
	Mesh _monkeyMesh;

	//add depth texture
	VkImageView _depthImageView;
	AllocatedImage _depthImage;

	//depth format
	VkFormat _depthFormat;

	//Add Render Object
	std::vector<RenderObject> _renderables;

	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Mesh> _meshes;

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

private:
	void init_vulkan();
	
	void init_swapchain();

	void init_commands();

	// --- vulkan renderpass method ---
	void init_default_renderpass();

	void init_framebuffers();

	// --- vulkan mainloop ---
	void init_sync_structures();

	// --- load shader file ---
	bool load_shader_module(const char* filePath, VkShaderModule* outShaderModule);

	// --- init pipelines ---
	void init_pipelines();

	// --- mesh pipelines ---
	void load_meshes();
	void upload_mesh(Mesh& mesh);

	// --- Add material ---
	Material* create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);

	Material* get_material(const std::string& name);

	Mesh* get_mesh(const std::string& name);

	void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);

	// --- init scene ---
	void init_scene();
};

class PipelineBuilder {
public:
	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	VkPipelineColorBlendAttachmentState _colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineLayout _pipelineLayout;
	//add depth 
	VkPipelineDepthStencilStateCreateInfo _depthStencil;

	VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};