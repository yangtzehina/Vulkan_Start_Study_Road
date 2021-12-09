﻿// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vector>

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
};