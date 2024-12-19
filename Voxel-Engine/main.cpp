#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <glm/glm.hpp>
#include <array>
#include <unordered_map>
#include <thread>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "config.h"

#define CHUNK_SIZE 32
#define CHUNK_AREA CHUNK_SIZE*CHUNK_SIZE
#define CHUNK_VOL CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE

#define VOXEL_SIZE 1.0f

#define MOUSE_SENSITIVITY 0.1
#define PLAYER_SPEED 10
#define GRAVITY 1
#define PLAYER_HEIGHT 1.5

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;
const auto fixedTimeStep = std::chrono::duration<double>( 1.0f / 60.0f);

bool noClip = false;

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif


/* Utility Functions and Callbacks */

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

/* Structs and Global Variables (for access outside the engine, subject to change) */

struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct Voxel {
	bool isActive = false;
	short blockId;
};

struct Array3Hash {
	std::size_t operator()(const std::array<int, 3>& arr) const {
		std::size_t hash = 17;
		hash = hash * 31 + std::hash<int>()(arr[0]);
		hash = hash * 31 + std::hash<int>()(arr[1]);
		hash = hash * 31 + std::hash<int>()(arr[2]);
		return hash;
	}
};

struct ChunkResources {
	VkBuffer vertexBuffer;
	VkBuffer indexBuffer;
	VmaAllocation vertexAllocation;
	VmaAllocation indexAllocation;
	uint32_t indexCount;
};

struct GameState {
	glm::vec3 playerPos;
	glm::vec1 playerYaw;
	glm::vec1 playerPitch;
	std::vector<std::array<int, 3>> loadChunks;
	std::vector<std::array<int, 3>> unloadChunks; 
};

// loadedChunks stores a list of chunks to render as keys, and their corresponding memory handles as values
std::unordered_map<std::array<int, 3>, ChunkResources, Array3Hash> loadedChunks;

std::unordered_map<int, bool> keyState;

// Voxel* is a pointer to std::array<Voxel, CHUNK_VOL>
std::unordered_map<std::array<int, 3>, Voxel*, Array3Hash> chunks;

/* End Structs and Global Vars */

// Voxel Engine class
class VoxelEngine{
public:
	void run(){
		initWindow();
		initWorld();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:

	/* Private Variables - Vulkan Handles and Configuration */
	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	VkDeviceMemory chunkMemory;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;
	uint32_t mipLevels;
	VkImage colorImage;
	VkDeviceMemory colorImageMemory;
	VkImageView colorImageView;
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

	VmaAllocator allocator;
	int texWidth, texHeight, texChannels;

	uint32_t currentFrame = 0;
	bool firstUpdate = true;

	/* Private Variables - Game State */
	std::atomic<bool> framebufferResized = false;
	GameState state0, state1;
	std::mutex stateMutex0, stateMutex1;
	int updateState = 0;
	int renderState = 0;
	bool stateUpdate0 = false;
	bool stateUpdate1 = false;
	double lastX = 0, lastY = 0;
	float yaw;
	float pitch;
	glm::vec3 velocity;
	glm::vec3 position;
	glm::vec3 forward = glm::vec3(1.0f, 0.0f, 0.0f);
	glm::vec3 right = glm::vec3(0.0f, 0.0f, 1.0f);
	std::array<int, 3> playerChunk;

	std::vector<std::array<int, 3>> updateLoadChunkList;
	std::vector<std::array<int, 3>> updateUnloadChunkList;

	/* 
	 * Initialize Window and GLFW callbacks
	 */
	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		glfwSetKeyCallback(window, keyCallback);
		glfwSetCursorPosCallback(window, cursorPosCallback);
	}

	// GLFW Callback - framebufferResize
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<VoxelEngine*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	// GLFW Callback - keyCallback
	// Currently unused; I use glfwGetKey in the game update function instead
	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {

	}

	// GLFW Callback - cursorPosCallback
	// This function updates the player yaw, pitch, forward, and right vectors
	bool first = true;

	static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {

		auto app = reinterpret_cast<VoxelEngine*>(glfwGetWindowUserPointer(window));
		if (app->first == true) {
			app->lastX = xpos;
			app->lastY = ypos;
			app->first = false;
		}

		double dx = app->lastX - xpos;
		double dy = ypos - app->lastY;

		app->lastX = xpos;
		app->lastY = ypos;

		dx *= MOUSE_SENSITIVITY;
		dy *= MOUSE_SENSITIVITY;

		app->yaw += dx;
		app->pitch += dy;

		if (app->pitch > 89.0f) {
			app->pitch = 89.0f;
		}
		if (app->pitch < -89.0f) {
			app->pitch = -89.0f;
		}

		app->forward.x = cos(glm::radians(app->yaw)) * cos(glm::radians(app->pitch));
		app->forward.y = sin(glm::radians(app->pitch));
		app->forward.z = sin(glm::radians(app->yaw)) * cos(glm::radians(app->pitch));
		app->forward = glm::normalize(app->forward);

		app->right = glm::normalize(glm::cross(app->forward, glm::vec3(0.0f, -1.0f, 0.0f)));
	}

	// Vulkan Setup and Initialization
	void initVulkan(){
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createColorResources();
		createDepthResources();
		createFramebuffers();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		createVertexAndIndexBuffers();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
	}

	// Define Required Validation Layers
	const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

	/*
	 * This function creates the Vulkan Instance for my voxel engine,
	 * It also adds validation layer support for instance creation API calls
	 */
	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Voxel Engine";
		appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 1, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_API_VERSION_1_3;
		appInfo.apiVersion = VK_API_VERSION_1_3;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else {
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create instance!");
		}
	}

	/*
	 * Creates a debug messenger, which handles debug messages if validation layers are enabled.
	 */
	void setupDebugMessenger() {
		if (!enableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	// Callback function used by the debug utils messenger - Prints message contents to cerr
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData) {

		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

	/*
	 * Creates a window surface using glfw
	 */
	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	// Define Required Device Extensions
	const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	/*
	 * Selects a physical device. 
	 */
	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				msaaSamples = getMaxUsableSampleCount();
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}

	}

	/*
	 * Creates a logical device from the chosen physical device.
	 * Also gets queues for presenting and graphics
	 */
	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.sampleRateShading = VK_TRUE;

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	// Struct to hold queue family indices
	// isComplete returns true if at least one is sufficient for rendering and presenting
	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete() {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	// Struct to hold swap chain details
	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	/*
	 * Creates a swap chain and the swap chain images
	 */
	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	/*
	 * Creates an image view for each swap chain image
	 */
	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		}
	}

	/*
	 * Creates a render pass, which specifies the attachments accessed by the graphics pipeline
	 */
	void createRenderPass() {
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = msaaSamples;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = msaaSamples;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription colorAttachmentResolve{};
		colorAttachmentResolve.format = swapChainImageFormat;
		colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 2;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;
		subpass.pResolveAttachments = &colorAttachmentResolveRef;

		std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	/*
	 * Create a descriptor set layout, which specifies the resources accessed by the graphics pipeline
	 */
	void createDescriptorSetLayout() {
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		uboLayoutBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	/*
	 * Create a graphics pipeline, based on my earlier render pass and descriptor set layout
	 */
	void createGraphicsPipeline() {
		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds = 0.0f;
		depthStencil.maxDepthBounds = 1.0f;
		depthStencil.stencilTestEnable = VK_FALSE;
		depthStencil.front = {};
		depthStencil.back = {};

		auto vertShaderCode = readFile("shaders/vert.spv");
		auto fragShaderCode = readFile("shaders/frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDesciptions = Vertex::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDesciptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDesciptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_TRUE;
		multisampling.rasterizationSamples = msaaSamples;
		multisampling.minSampleShading = 0.2f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPushConstantRange range{};
		range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		range.offset = 0;
		range.size = 16;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		pipelineLayoutInfo.pPushConstantRanges = &range;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;

		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;

		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	/*
	 * Create a command pool
	 */
	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	/*
	 * Create Framebuffers for swapchain images, color, and depth image views
	 */
	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 3> attachments = {
				colorImageView,
				depthImageView,
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	/*
	 * Create Resources for a color buffer to render to with one sample per pixel
	 */
	void createColorResources() {
		VkFormat colorFormat = swapChainImageFormat;

		createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat,
			VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
		colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
	}

	/*
	 * Create Resources for the Depth Buffer such as an image and an image view
	 */
	void createDepthResources() {
		VkFormat depthFormat = findDepthFormat();
		createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage,
			depthImageMemory);
		depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
	}

	/*
	 * Create Resources for the Texture
	 */
	void createTextureImage() {
		stbi_uc* pixels = stbi_load("textures/atlas.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		if (!pixels) {
			throw std::runtime_error("failed to load texture image!");
		}

		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		stbi_image_free(pixels);

		createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			textureImage, textureImageMemory);

		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);
	}

	/*
	 * Creates an Image View for the Texture
	 */
	void createTextureImageView() {
		textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
	}

	/*
	 * 
	 */
	void createTextureSampler() {
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_NEAREST;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.anisotropyEnable = VK_TRUE;
		
		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);

		samplerInfo.maxAnisotropy = 2.0f;//properties.limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST; // TODO: Enable Mipmapping with texture atlases (or swap to texture arrays)
		samplerInfo.mipLodBias = -0.75f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 4.0f;//static_cast<float>(mipLevels);

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	/* Create Initial Vertex and Index Buffers for first time loading */
	/* This is based on initial loadedChunks and saves their corresponding chunkResources (buffers and allocations) */
	void createVertexAndIndexBuffers() {
		//Initialize VMA

		VmaAllocatorCreateInfo allocatorCreateInfo{};
		allocatorCreateInfo.device = device;
		allocatorCreateInfo.instance = instance;
		allocatorCreateInfo.physicalDevice = physicalDevice;
		allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;

		vmaCreateAllocator(&allocatorCreateInfo, &allocator);

		//Allocate a buffer for each loadedChunk
		//Load each buffer with vertex or index data

		for (auto& [coords, chunkBuffers] : loadedChunks) {
			//Generate vertex and index data from voxel data
			std::vector<Vertex> chunkVertices{};
			std::vector<uint16_t> chunkIndices{};
			constructChunk(coords, chunkVertices, chunkIndices);

			//Skip empty chunks
			if (chunkVertices.size() == 0 || chunkIndices.size() == 0) continue;

			VkDeviceSize chunkVerticesSize = sizeof(chunkVertices[0]) * chunkVertices.size();
			VkDeviceSize chunkIndicesSize = sizeof(chunkIndices[0]) * chunkIndices.size();

			//Use a staging buffer to send vertex data to GPU
			VmaAllocationCreateInfo stagingAllocInfo{};
			stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

			VkBufferCreateInfo stagingBufferInfo{};
			stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			stagingBufferInfo.size = chunkVerticesSize;
			stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
			stagingBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VkBuffer stagingBuffer;
			VmaAllocation stagingAllocation;
			vmaCreateBuffer(allocator, &stagingBufferInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, nullptr);

			void* data;
			vmaMapMemory(allocator, stagingAllocation, &data);
			memcpy(data, chunkVertices.data(), (size_t)chunkVerticesSize);
			vmaUnmapMemory(allocator, stagingAllocation);

			VmaAllocationCreateInfo deviceAllocInfo = {};
			deviceAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

			VkBufferCreateInfo deviceBufferInfo{};
			deviceBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			deviceBufferInfo.size = chunkVerticesSize;
			deviceBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
			deviceBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			vmaCreateBuffer(allocator, &deviceBufferInfo, &deviceAllocInfo, &chunkBuffers.vertexBuffer, &chunkBuffers.vertexAllocation, nullptr);

			copyBuffer(stagingBuffer, chunkBuffers.vertexBuffer, chunkVerticesSize);

			vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

			//Use a staging buffer to send index data to GPU
			VkBufferCreateInfo stagingIndexBufferInfo{};
			stagingIndexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			stagingIndexBufferInfo.size = chunkIndicesSize;
			stagingIndexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
			stagingIndexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VkBuffer indexStagingBuffer;
			VmaAllocation indexStagingAllocation;
			vmaCreateBuffer(allocator, &stagingIndexBufferInfo, &stagingAllocInfo, &indexStagingBuffer, &indexStagingAllocation, nullptr);

			void* indexData;
			vmaMapMemory(allocator, indexStagingAllocation, &indexData);
			memcpy(indexData, chunkIndices.data(), (size_t)chunkIndicesSize);
			vmaUnmapMemory(allocator, indexStagingAllocation);

			VkBufferCreateInfo deviceIndexBufferInfo{};
			deviceIndexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			deviceIndexBufferInfo.size = chunkIndicesSize;
			deviceIndexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
			deviceIndexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			vmaCreateBuffer(allocator, &deviceIndexBufferInfo, &deviceAllocInfo, &chunkBuffers.indexBuffer, &chunkBuffers.indexAllocation, nullptr);

			copyBuffer(indexStagingBuffer, chunkBuffers.indexBuffer, chunkIndicesSize);

			vmaDestroyBuffer(allocator, indexStagingBuffer, indexStagingAllocation);

			chunkBuffers.indexCount = chunkIndices.size();
		}
	}

	/* Dynamically loadChunks from chunkCoords into loadedChunks */
	void loadChunks(std::vector<std::array<int, 3>> chunkCoords){
		for(const std::array<int,3>& coord : chunkCoords) {
			if(!loadedChunks.contains(coord)){
				// Add coord to loaded chunks, allocate buffers, fill with vertex and index data
				loadedChunks[coord] = {};
				// chunkBuffer is stored at loadedChunks[coord]

				std::vector<Vertex> chunkVertices{};
				std::vector<uint16_t> chunkIndices{};
				constructChunk(coord, chunkVertices, chunkIndices);

				//Skip empty chunks
				if (chunkVertices.size() == 0 || chunkIndices.size() == 0) continue;

				VkDeviceSize chunkVerticesSize = sizeof(chunkVertices[0]) * chunkVertices.size();
				VkDeviceSize chunkIndicesSize = sizeof(chunkIndices[0]) * chunkIndices.size();

				//Use a staging buffer to send vertex data to GPU
				VmaAllocationCreateInfo stagingAllocInfo{};
				stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

				VkBufferCreateInfo stagingBufferInfo{};
				stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
				stagingBufferInfo.size = chunkVerticesSize;
				stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
				stagingBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

				VkBuffer stagingBuffer;
				VmaAllocation stagingAllocation;
				vmaCreateBuffer(allocator, &stagingBufferInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, nullptr);

				void* data;
				vmaMapMemory(allocator, stagingAllocation, &data);
				memcpy(data, chunkVertices.data(), (size_t)chunkVerticesSize);
				vmaUnmapMemory(allocator, stagingAllocation);

				VmaAllocationCreateInfo deviceAllocInfo = {};
				deviceAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

				VkBufferCreateInfo deviceBufferInfo{};
				deviceBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
				deviceBufferInfo.size = chunkVerticesSize;
				deviceBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
				deviceBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

				vmaCreateBuffer(allocator, &deviceBufferInfo, &deviceAllocInfo, &loadedChunks[coord].vertexBuffer, &loadedChunks[coord].vertexAllocation, nullptr);

				copyBuffer(stagingBuffer, loadedChunks[coord].vertexBuffer, chunkVerticesSize);

				vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

				//Use a staging buffer to send index data to GPU
				VkBufferCreateInfo stagingIndexBufferInfo{};
				stagingIndexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
				stagingIndexBufferInfo.size = chunkIndicesSize;
				stagingIndexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
				stagingIndexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

				VkBuffer indexStagingBuffer;
				VmaAllocation indexStagingAllocation;
				vmaCreateBuffer(allocator, &stagingIndexBufferInfo, &stagingAllocInfo, &indexStagingBuffer, &indexStagingAllocation, nullptr);

				void* indexData;
				vmaMapMemory(allocator, indexStagingAllocation, &indexData);
				memcpy(indexData, chunkIndices.data(), (size_t)chunkIndicesSize);
				vmaUnmapMemory(allocator, indexStagingAllocation);

				VkBufferCreateInfo deviceIndexBufferInfo{};
				deviceIndexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
				deviceIndexBufferInfo.size = chunkIndicesSize;
				deviceIndexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
				deviceIndexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

				vmaCreateBuffer(allocator, &deviceIndexBufferInfo, &deviceAllocInfo, &loadedChunks[coord].indexBuffer, &loadedChunks[coord].indexAllocation, nullptr);

				copyBuffer(indexStagingBuffer, loadedChunks[coord].indexBuffer, chunkIndicesSize);

				vmaDestroyBuffer(allocator, indexStagingBuffer, indexStagingAllocation);

				loadedChunks[coord].indexCount = chunkIndices.size();

			}
		}
	}

	/* Dynamically unload chunks from loadedChunks list by chunkCoords */
	/* Should only run when device is idle to not deallocate buffers during drawing */
	void unloadChunks(std::vector<std::array<int, 3>> chunkCoords) {
		vkQueueWaitIdle(graphicsQueue);
		for(const std::array<int,3>& coord : chunkCoords){
			if(loadedChunks.contains(coord)){
				// Deallocate buffer, remove entry from loadedChunks
				if (loadedChunks[coord].indexCount != 0) {
					vmaDestroyBuffer(allocator, loadedChunks[coord].indexBuffer, loadedChunks[coord].indexAllocation);
					vmaDestroyBuffer(allocator, loadedChunks[coord].vertexBuffer, loadedChunks[coord].vertexAllocation);
				}
				loadedChunks.erase(coord);
			}
		}
	}

	// Struct that stores a vector of vertices and indices
	struct mesh {
		std::vector<Vertex> vertices;
		std::vector<uint16_t> indices;
	};

	/* Given a chunk coord, constructs the proper vertex and index arrays and writes them to the vertice and indices parameters */
	void constructChunk(std::array<int, 3> coords, std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {

		//First query our chunks to see if it exists
		if (chunks.contains(coords) == false) {
			return;
		}

		//Vertex index for this chunk's mesh
		uint16_t idx = 0;
		
		//Pre-computing these values speeds up chunk generation significantly (from around 100ms to 10ms)
		Voxel* chunk = chunks[coords];
		Voxel* chunkYP = chunks.contains(std::array<int, 3>{coords[0], coords[1] - 1, coords[2]}) ? chunks[std::array<int, 3>{coords[0], coords[1] - 1, coords[2]}] : nullptr;
		Voxel* chunkXP = chunks.contains(std::array<int, 3>{coords[0] + 1, coords[1], coords[2]}) ? chunks[std::array<int, 3>{coords[0] + 1, coords[1], coords[2]}] : nullptr;
		Voxel* chunkXN = chunks.contains(std::array<int, 3>{coords[0] - 1, coords[1], coords[2]}) ? chunks[std::array<int, 3>{coords[0] - 1, coords[1], coords[2]}] : nullptr;
		Voxel* chunkZP = chunks.contains(std::array<int, 3>{coords[0], coords[1], coords[2] + 1}) ? chunks[std::array<int, 3>{coords[0], coords[1], coords[2] + 1}] : nullptr;
		Voxel* chunkZN = chunks.contains(std::array<int, 3>{coords[0], coords[1], coords[2] - 1}) ? chunks[std::array<int, 3>{coords[0], coords[1], coords[2] - 1}] : nullptr;
		Voxel* chunkYN = chunks.contains(std::array<int, 3>{coords[0], coords[1] + 1, coords[2]}) ? chunks[std::array<int, 3>{coords[0], coords[1] + 1, coords[2]}] : nullptr;
		
		//Possibly in the future I could use greedy meshing, but there are some issues with that, which I'd need to account for
		//std::unordered_map<std::array<int, 3>, uint16_t, Array3Hash> vert_idx_map;

		for (size_t i = 0; i < CHUNK_VOL; i++) {
			if (chunk[i].isActive == true) {
				int x = i / (CHUNK_SIZE * CHUNK_SIZE);
				int y = (i / CHUNK_SIZE) % CHUNK_SIZE;
				int z = i % CHUNK_SIZE;

				//Get uv coordinates in un-normalized range
				int u = (chunk[i].blockId % (int)floor((float)texWidth / ((float)TEXTURE_DIM + 2.0f*(float)TEXTURE_PADDING))) * (TEXTURE_DIM + 2*TEXTURE_PADDING) + TEXTURE_PADDING;//(static_cast<int>(chunks[coords][i].blockId) % (TEXTURE_DIM + 2*TEXTURE_PADDING)) * 32;
				int v = (chunk[i].blockId / (int)floor((float)texWidth / ((float)TEXTURE_DIM + 2.0f * (float)TEXTURE_PADDING))) * (TEXTURE_DIM + 2 * TEXTURE_PADDING) + TEXTURE_PADDING;//(static_cast<int>(chunks[coords][i].blockId) / 32) * 32;

				glm::vec2 uv1 = glm::vec2((u / (float)texWidth) + 0.000f, (v / (float)texHeight) + 0.000f);
				glm::vec2 uv2 = glm::vec2(((u + TEXTURE_DIM) / (float)texWidth) - 0.000f, (v / (float)texHeight) + 0.000f);
				glm::vec2 uv3 = glm::vec2((u / (float)texWidth) + 0.000f, ((v + TEXTURE_DIM) / (float)texHeight) - 0.000f);
				glm::vec2 uv4 = glm::vec2(((u + TEXTURE_DIM) / (float)texWidth) - 0.000f, ((v + TEXTURE_DIM) / (float)texHeight) - 0.000f);

				//Check neighboring voxels for isActive, minding edge cases
				//3 cases to draw top face: above voxel is empty, above chunk is empty, above voxel in above chunk is empty
				//+y face (top)
				if (y == 0 && !chunkYP || y == 0 && chunkYP[voxelCoordToI(x,CHUNK_SIZE-1,z)].isActive == false || y != 0 && chunk[voxelCoordToI(x, y-1, z)].isActive == false) {
					vertices.push_back({ {x, y, z},                   {0.0f, 0.0f, 0.0f}, {uv1} });
					vertices.push_back({ {x + 1.0f, y, z},            {1.0f, 0.0f, 0.0f}, {uv2} });
					vertices.push_back({ {x, y, z + 1.0f},            {0.0f, 0.0f, 1.0f}, {uv3} });
					vertices.push_back({ {x + 1.0f, y, z + 1.0f},     {1.0f, 0.0f, 1.0f}, {uv4} });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 2) });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx + 2), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 3) });
					idx += 4;
				}

				//Handle 3 faced blocks (top-side-bottom)
				if (block3faced.contains(static_cast<int>(chunk[i].blockId))) {
					u = ((chunk[i].blockId + 1) % (int)floor((float)texWidth / ((float)TEXTURE_DIM + 2.0f * (float)TEXTURE_PADDING))) * (TEXTURE_DIM + 2 * TEXTURE_PADDING) + TEXTURE_PADDING;//(static_cast<int>(chunks[coords][i].blockId) % (TEXTURE_DIM + 2*TEXTURE_PADDING)) * 32;
					v = ((chunk[i].blockId + 1) / (int)floor((float)texWidth / ((float)TEXTURE_DIM + 2.0f * (float)TEXTURE_PADDING))) * (TEXTURE_DIM + 2 * TEXTURE_PADDING) + TEXTURE_PADDING;//(static_cast<int>(chunks[coords][i].blockId) / 32) * 32;

					uv1 = glm::vec2((u / (float)texWidth) + 0.000f, (v / (float)texHeight) + 0.000f);
					uv2 = glm::vec2(((u + TEXTURE_DIM) / (float)texWidth) - 0.000f, (v / (float)texHeight) + 0.000f);
					uv3 = glm::vec2((u / (float)texWidth) + 0.000f, ((v + TEXTURE_DIM) / (float)texHeight) - 0.000f);
					uv4 = glm::vec2(((u + TEXTURE_DIM) / (float)texWidth) - 0.000f, ((v + TEXTURE_DIM) / (float)texHeight) - 0.000f);
				}

				//repeat for +x face
				if (x == CHUNK_SIZE-1 && !chunkXP || x == CHUNK_SIZE - 1 && chunkXP[voxelCoordToI(0, y, z)].isActive == false || x != CHUNK_SIZE-1 && chunk[voxelCoordToI(x + 1, y, z)].isActive == false) {
					vertices.push_back({ {x + 1.0f, y, z + 1.0f},            {1.0f, 0.0f, 1.0f}, {uv1} });
					vertices.push_back({ {x + 1.0f, y, z},                   {1.0f, 0.0f, 0.0f}, {uv2} });
					vertices.push_back({ {x + 1.0f, y + 1.0f, z + 1.0f},     {1.0f, 1.0f, 1.0f}, {uv3} });
					vertices.push_back({ {x + 1.0f, y + 1.0f, z},            {1.0f, 1.0f, 0.0f}, {uv4} });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 2) });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx + 2), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 3) });
					idx += 4;
				}

				//repeat for -x face
				if (x == 0 && !chunkXN || x == 0 && !chunkXN[voxelCoordToI(CHUNK_SIZE-1, y, z)].isActive || x != 0 && chunk[voxelCoordToI(x - 1, y, z)].isActive == false) {
					vertices.push_back({ {x, y, z},                   {0.0f, 0.0f, 0.0f}, {uv1} });
					vertices.push_back({ {x, y, z + 1.0f},            {0.0f, 0.0f, 1.0f}, {uv2} });
					vertices.push_back({ {x, y + 1.0f, z},            {0.0f, 1.0f, 0.0f}, {uv3} });
					vertices.push_back({ {x, y + 1.0f, z + 1.0f},     {0.0f, 1.0f, 1.0f}, {uv4} });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 2) });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx + 2), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 3) });
					idx += 4;
				}

				//repeat for +z face
				if (z == CHUNK_SIZE - 1 && !chunkZP || z == CHUNK_SIZE - 1 && !chunkZP[voxelCoordToI(x, y, 0)].isActive || z != CHUNK_SIZE - 1 && chunk[voxelCoordToI(x, y, z + 1)].isActive == false) {
					vertices.push_back({{x, y + 1.0f, z + 1.0f},            {0.0f, 1.0f, 1.0f}, {uv4} });
					vertices.push_back({{x, y, z + 1.0f},                   {0.0f, 0.0f, 1.0f}, {uv2} });
					vertices.push_back({{x + 1.0f, y + 1.0f, z + 1.0f},     {1.0f, 1.0f, 1.0f}, {uv3} });
					vertices.push_back({{x + 1.0f, y, z + 1.0f},            {1.0f, 0.0f, 1.0f}, {uv1} });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 2) });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx + 2), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 3) });
					idx += 4;
				}

				//repeat for -z face
				if (z == 0 && !chunkZN || z == 0 && chunkZN[voxelCoordToI(x, y, CHUNK_SIZE-1)].isActive == false || z != 0 && chunk[voxelCoordToI(x, y, z - 1)].isActive == false) {
					vertices.push_back({ {x, y, z},                   {0.0f, 0.0f, 0.0f}, {uv1} });
					vertices.push_back({ {x, y + 1.0f, z},            {0.0f, 1.0f, 0.0f}, {uv3} });
					vertices.push_back({ {x + 1.0f, y, z},            {1.0f, 0.0f, 0.0f}, {uv2} });
					vertices.push_back({ {x + 1.0f, y + 1.0f, z},     {1.0f, 1.0f, 0.0f}, {uv4} });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 2) });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx + 2), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 3) });
					idx += 4;
				}

				// Change index to bottom texture from the texture atlas for 3faced blocks (like dirt)
				if (block3faced.contains(chunk[i].blockId)) {
					u = ((chunk[i].blockId + 2) % (int)floor((float)texWidth / ((float)TEXTURE_DIM + 2.0f * (float)TEXTURE_PADDING))) * (TEXTURE_DIM + 2 * TEXTURE_PADDING) + TEXTURE_PADDING;//(static_cast<int>(chunks[coords][i].blockId) % (TEXTURE_DIM + 2*TEXTURE_PADDING)) * 32;
					v = ((chunk[i].blockId + 2) / (int)floor((float)texWidth / ((float)TEXTURE_DIM + 2.0f * (float)TEXTURE_PADDING))) * (TEXTURE_DIM + 2 * TEXTURE_PADDING) + TEXTURE_PADDING;//(static_cast<int>(chunks[coords][i].blockId) / 32) * 32;

					uv1 = glm::vec2((u / (float)texWidth) + 0.000f, (v / (float)texHeight) + 0.000f);
					uv2 = glm::vec2(((u + TEXTURE_DIM) / (float)texWidth) - 0.000f, (v / (float)texHeight) + 0.000f);
					uv3 = glm::vec2((u / (float)texWidth) + 0.000f, ((v + TEXTURE_DIM) / (float)texHeight) - 0.000f);
					uv4 = glm::vec2(((u + TEXTURE_DIM) / (float)texWidth) - 0.000f, ((v + TEXTURE_DIM) / (float)texHeight) - 0.000f);
				}

				//repeat for -y face
				if (y == CHUNK_SIZE - 1 && !chunkYN || y == CHUNK_SIZE - 1 && chunkYN[voxelCoordToI(x, 0, z)].isActive == false || y != CHUNK_SIZE - 1 && chunk[voxelCoordToI(x, y + 1, z)].isActive == false) {
					vertices.push_back({ {x, y + 1.0f, z},                   {0.0f, 1.0f, 0.0f}, {uv1} });
					vertices.push_back({ {x + 1.0f, y + 1.0f, z + 1.0f},     {1.0f, 1.0f, 1.0f}, {uv2} });
					vertices.push_back({ {x + 1.0f, y + 1.0f, z},            {1.0f, 1.0f, 0.0f}, {uv3} });
					vertices.push_back({ {x, y + 1.0f, z + 1.0f},            {0.0f, 1.0f, 1.0f}, {uv4} });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx), static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 2) });
					indices.insert(indices.end(), { static_cast<uint16_t>(idx + 1), static_cast<uint16_t>(idx + 0), static_cast<uint16_t>(idx + 3) });
					idx += 4;
				}
			}
		}
	}

	/* Helper function to convert from coord to index */
	size_t voxelCoordToI(int x, int y, int z) {
		return (x * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + z;
	}

	void createUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

			vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createDescriptorPool() {
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = textureImageView;
			imageInfo.sampler = textureSampler;

			std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;
			descriptorWrites[0].pImageInfo = nullptr;
			descriptorWrites[0].pTexelBufferView = nullptr;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;
			descriptorWrites[1].pTexelBufferView = nullptr;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocated command buffers!");
		}


	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
		clearValues[1].depthStencil = { 1.0f, 0 };

		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

		//Draw every loaded chunk
		for (const auto& [coords, chunkMem] : loadedChunks) {
			if (chunkMem.indexCount == 0) continue;

			//Set Push Constants to model matrix offset coords (only 16 bytes!)
			float data[4] = { float(coords[0]*CHUNK_SIZE), float(coords[1]*CHUNK_SIZE), float(coords[2]*CHUNK_SIZE), 0.0f };
			uint32_t offset = 0;
			uint32_t size = 16;
			vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, offset, size, data);

			//Bind Vertex and Index Buffers
			VkBuffer vertexBuffers[] = { chunkMem.vertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(commandBuffer, chunkMem.indexBuffer, 0, VK_INDEX_TYPE_UINT16);

			vkCmdDrawIndexed(commandBuffer, chunkMem.indexCount, 1, 0, 0, 0);
		}

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	/*
	 * The main loop for this application
	 * Launches the render and update threads
	 */
	void mainLoop(){

		std::thread rt(&VoxelEngine::renderThread, this);

		updateThread();

		if (rt.joinable()) {
			rt.join();
		}

		vkDeviceWaitIdle(device);
	}

	void updateThread() {

		//auto t = std::chrono::nanoseconds(0); // If time is a factor in physics updates
		auto accumulator = std::chrono::duration<double, std::nano>(0);
		auto currentTime = std::chrono::high_resolution_clock::now();

		while (!glfwWindowShouldClose(window)) {
			bool applyState = false;

			auto newTime = std::chrono::high_resolution_clock::now();
			auto stepTime = newTime - currentTime;
			currentTime = newTime;

			accumulator += stepTime;

			// If the accumulator exceeds a fixedTimeStep, we process as many steps as needed, then apply it to the shared game state
			while(accumulator >= fixedTimeStep){
				//Update Physics
				updatePhysics();
				accumulator -= fixedTimeStep;
				//t += fixedTimeStep;
				applyState = true;
			}

			applyDrawDistance();
			applyChunkGenerator();

			//Update Shared Game State
			//TODO: Streamline update states
			if (updateState == 0 && stateUpdate0 == false && applyState) {
				std::unique_lock<std::mutex> lock(stateMutex0);
				glfwPollEvents();
				state0.playerPitch = glm::vec1(pitch);
				state0.playerYaw = glm::vec1(yaw);
				state0.playerPos = position;
				state0.loadChunks = updateLoadChunkList;
				state0.unloadChunks = updateUnloadChunkList;

				updateLoadChunkList.clear();
				updateUnloadChunkList.clear();
				stateUpdate0 = true;
				updateState = 1;
			} else if (updateState == 1 && stateUpdate1 == false && applyState) {
				std::unique_lock<std::mutex> lock(stateMutex1);
				glfwPollEvents();
				//Update Game State
				state1.playerPitch = glm::vec1(pitch);
				state1.playerYaw = glm::vec1(yaw);
				state1.playerPos = position;
				state1.loadChunks = updateLoadChunkList;
				state1.unloadChunks = updateUnloadChunkList;

				updateLoadChunkList.clear();
				updateUnloadChunkList.clear();
				stateUpdate1 = true;
				updateState = 0;
			}

			//TODO: Release the lock early

			// Sleep for the remaining time
			if(fixedTimeStep - accumulator > std::chrono::nanoseconds(0)) {
				std::this_thread::sleep_for(fixedTimeStep - accumulator);
			}
		}
	}

	bool firstDrawDistance = true;

	void applyDrawDistance() {
		if (firstDrawDistance) {
			playerChunk = getPlayerChunk();
			firstDrawDistance = false;
		}
		if(useViewDistance) {
			std::array<int, 3> newPlayerChunk = getPlayerChunk();
			if(newPlayerChunk != playerChunk) {
				//Use a set to find the load and unload vector of chunks
				std::set<std::array<int, 3>> oldChunks;
				std::set<std::array<int ,3>> newChunks;
				for (int x = -VIEW_DISTANCE; x < VIEW_DISTANCE + 1; x++) {
					for (int y = -VIEW_DISTANCE; y < VIEW_DISTANCE + 1; y++) {
						for (int z = -VIEW_DISTANCE; z < VIEW_DISTANCE + 1; z++) {
							oldChunks.insert(std::array<int, 3>{playerChunk[0]+x,playerChunk[1]+y,playerChunk[2]+z});
							newChunks.insert(std::array<int, 3>{newPlayerChunk[0]+x,newPlayerChunk[1]+y,newPlayerChunk[2]+z});
						}
					}
				}

				std::set_difference(newChunks.begin(), newChunks.end(), oldChunks.begin(), oldChunks.end(), std::back_inserter(updateLoadChunkList));
				std::set_difference(oldChunks.begin(), oldChunks.end(), newChunks.begin(), newChunks.end(), std::back_inserter(updateUnloadChunkList));

				playerChunk = getPlayerChunk();
			}
		}
	}

	std::array<int, 3> getPlayerChunk() {
		return std::array<int, 3>{static_cast<int>(floor(position.x / CHUNK_SIZE)), static_cast<int>(floor(position.y / CHUNK_SIZE)), static_cast<int>(floor(position.z / CHUNK_SIZE))};
	}

	void applyChunkGenerator() {
		if (generateChunks) {
			for (const auto& coord : updateLoadChunkList) {
				if (!chunks.contains(coord)) {
					generateChunk(coord);
				}
			}
		}
	}

	void updatePhysics(){
		if(playerController == DEFAULT){ // TODO: if default controls
			float yComponent = velocity.y;
			float ws = glfwGetKey(window, GLFW_KEY_W) - glfwGetKey(window, GLFW_KEY_S);
			float ad = glfwGetKey(window, GLFW_KEY_D) - glfwGetKey(window, GLFW_KEY_A);
			velocity = (ws * glm::normalize(forward) + ad * right) * (float)PLAYER_SPEED * (1.0f + glfwGetKey(window, GLFW_KEY_LEFT_SHIFT));
			velocity.y = yComponent + GRAVITY;

			movePlayerAndCollide();

		} else if(playerController == NOCLIP) { // NO CLIP controls
			float ws = glfwGetKey(window, GLFW_KEY_W) - glfwGetKey(window, GLFW_KEY_S);
			float ad = glfwGetKey(window, GLFW_KEY_D) - glfwGetKey(window, GLFW_KEY_A);
			velocity = (ws * glm::normalize(forward) + ad * right) * (float)PLAYER_SPEED * (1.0f + glfwGetKey(window, GLFW_KEY_LEFT_SHIFT));

			position += static_cast<float>(fixedTimeStep.count()) * velocity;
		} else {
			//managed by user
		}
	}

	struct WorldVoxel {
		std::array<int, 3> chunkPos;
		int voxelI;
		bool isActive;
	};

	/* TODO: Fully implement a collision system */
	/* I drafted out collisions, but I'd rather add features like lighting before then */
	void movePlayerAndCollide() {

	}

	/* Helper function - get chunkCoord at player position */
	std::vector<WorldVoxel> fetchVoxelsAtPlayer () {
		std::vector<WorldVoxel> voxelList;
		glm::vec3 pos = position - glm::vec3(0.5, 0, 0.5);
		for (int x = 0; x < 1; x++){
			for(int z=0; z < 1; z++) {
				for (float y=0; y < PLAYER_HEIGHT+1; y++){
					if (y == ceil(PLAYER_HEIGHT)) y = PLAYER_HEIGHT;
					voxelList.push_back(worldPosToWorldVoxel(pos + glm::vec3(x, y, z)));
				}
			}
		}

		return voxelList;
	}

	/* Helper function - convert from world position (as a vec3 float) to a chunkCoord and voxel index pair */
	WorldVoxel worldPosToWorldVoxel (glm::vec3 pos) {
		if(chunks.contains(std::array<int, 3>{static_cast<int>(floor(pos.x / CHUNK_SIZE)), static_cast<int>(floor(pos.y / CHUNK_SIZE)), static_cast<int>(floor(pos.z / CHUNK_SIZE))})){
			if(chunks[std::array<int, 3>{static_cast<int>(floor(pos.x / CHUNK_SIZE)), static_cast<int>(floor(pos.y / CHUNK_SIZE)), static_cast<int>(floor(pos.z / CHUNK_SIZE))}][voxelCoordToI(int(floor(pos.x)) % CHUNK_SIZE, int(floor(pos.y)) % CHUNK_SIZE, int(floor(pos.z)) % CHUNK_SIZE)].isActive == true) {
				return WorldVoxel{std::array<int, 3>{static_cast<int>(floor(pos.x / CHUNK_SIZE)), static_cast<int>(floor(pos.y / CHUNK_SIZE)), static_cast<int>(floor(pos.z / CHUNK_SIZE))}, (int)voxelCoordToI(int(floor(pos.x)) % CHUNK_SIZE,int(floor(pos.y)) % CHUNK_SIZE,int(floor(pos.z)) % CHUNK_SIZE), true};
			}
		}
		return WorldVoxel{std::array<int,3>{-99, -99, -99}, 0, false};
	}

	/* Helper function - convert from a chunkCoord and voxel pair to a vec3 float position. */
	/* Voxels grow from 0,0,0 to +x,+y,+z, and my coordinate system is y+ is down */
	glm::vec3 worldVoxelToWorldPos (WorldVoxel wv) {
		int x = wv.voxelI / (CHUNK_SIZE * CHUNK_SIZE);
		int y = (wv.voxelI / CHUNK_SIZE) % CHUNK_SIZE;
		int z = wv.voxelI % CHUNK_SIZE;
		return glm::vec3(wv.chunkPos[0] * CHUNK_SIZE + x, wv.chunkPos[1] * CHUNK_SIZE + y, wv.chunkPos[2] * CHUNK_SIZE + z);
	}

	void renderThread() {
		while (!glfwWindowShouldClose(window)) {
			if (renderState == 0 && stateUpdate0 == true) {
				std::unique_lock<std::mutex> lock(stateMutex0);
				unloadChunks(state0.unloadChunks);
				loadChunks(state0.loadChunks);
				state0.loadChunks.clear();
				state0.unloadChunks.clear();
				drawFrame(state0);
				stateUpdate0 = false;
				renderState = 1;
			} else if (renderState == 1 && stateUpdate1 == true) {
				std::unique_lock<std::mutex> lock(stateMutex1);
				unloadChunks(state1.unloadChunks);
				loadChunks(state1.loadChunks);
				state1.loadChunks.clear();
				state1.unloadChunks.clear();
				drawFrame(state1);
				stateUpdate1 = false;
				renderState = 0;
			}

		}
	}

	void drawFrame(GameState gs) {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		updateUniformBuffer(currentFrame, gs);

		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame]};
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame]};
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void cleanup(){
		cleanupSwapChain();

		vkDestroySampler(device, textureSampler, nullptr);
		vkDestroyImageView(device, textureImageView, nullptr);

		vkDestroyImage(device, textureImage, nullptr);
		vkFreeMemory(device, textureImageMemory, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		for (auto& [coords, chunk] : loadedChunks) {
			if (chunk.indexCount == 0) continue;
			vmaDestroyBuffer(allocator, chunk.indexBuffer, chunk.indexAllocation);
			vmaDestroyBuffer(allocator, chunk.vertexBuffer, chunk.vertexAllocation);
		}

		vmaDestroyAllocator(allocator);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

		vkDestroyRenderPass(device, renderPass, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);
		
		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

 		glfwDestroyWindow(window);

		glfwTerminate();

		for (auto& chunk : chunks) {
			delete[] chunk.second;
		}
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createColorResources();
		createDepthResources();
		createFramebuffers();
	}

	void cleanupSwapChain() {
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		vkDestroyImageView(device, colorImageView, nullptr);
		vkDestroyImage(device, colorImage, nullptr);
		vkFreeMemory(device, colorImageMemory, nullptr);

		for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
			vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
		}

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			vkDestroyImageView(device, swapChainImageViews[i], nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void updateUniformBuffer(uint32_t currentImage, GameState gs) {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo{};
		//Model matrix is obsolete here; I've implemented model matrices via push constants.
		ubo.model = glm::mat4(1.0f);
		glm::vec3 direction = glm::vec3(cos(glm::radians(gs.playerYaw)) * cos(glm::radians(gs.playerPitch)), sin(glm::radians(gs.playerPitch)), cos(glm::radians(gs.playerPitch)) * sin(glm::radians(gs.playerYaw)));
		ubo.view = glm::lookAt(gs.playerPos, gs.playerPos + direction, glm::vec3(0.0f, -1.0f, 0.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 1500.0f);
		ubo.proj[1][1] *= -1;

		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}

	void initWorld() {
		//init World geometry, set initial camera pos, set inital loaded chunks
		//Consider creating a helper function for writing to chunks (takes a function input perhaps)

		//write and load a sphere to 0,-1,0
		chunks[std::array<int, 3>{0, -1, 0}] = new Voxel[CHUNK_VOL];
		for (size_t i = 0; i < CHUNK_VOL; i++) {
			int x = i / (CHUNK_SIZE * CHUNK_SIZE);
			int y = (i / CHUNK_SIZE) % CHUNK_SIZE;
			int z = i % CHUNK_SIZE;
			if ((x - CHUNK_SIZE/2)*(x-CHUNK_SIZE/2) + (y - CHUNK_SIZE / 2) * (y - CHUNK_SIZE / 2) + (z - CHUNK_SIZE / 2) * (z - CHUNK_SIZE / 2) <= (CHUNK_SIZE/2)*(CHUNK_SIZE/2)) {
				chunks[std::array<int, 3>{0, -1, 0}][i] = Voxel(true, 5);
			}
		}
		loadedChunks[std::array<int, 3>{0, -1, 0}] = {};

		//write grass and dirt to a 5x5 ground area
		for (int i = -2; i < 3; i++) {
			for (int j = -2; j < 3; j++) {
				generateChunk(std::array<int, 3>{i, 0, j});
				loadedChunks[std::array<int, 3>{i, 0, j}] = {};
			}
		}

		velocity = glm::vec3(0.0f, 0.0f, 0.0f);
		position = glm::vec3(4.0f, -2.0f, 4.0f);
	}

	void generateChunk(std::array<int, 3> coords) {
		//logic to apply to uninitialized chunks
		chunks[coords] = new Voxel[CHUNK_VOL];
		if (coords[1] == 0) {
			for (size_t i = 0; i < CHUNK_VOL; i++) {
				int x = i / (CHUNK_SIZE * CHUNK_SIZE);
				int y = (i / CHUNK_SIZE) % CHUNK_SIZE;
				int z = i % CHUNK_SIZE;
				if (y == 0) {
					chunks[coords][i] = Voxel(true, 1);
				} else {
					chunks[coords][i] = Voxel(true, 3);
				}
			}
		} else if (coords[1] > 0) {
			for (size_t i = 0; i < CHUNK_VOL; i++) {
				chunks[coords][i] = Voxel(true, 3);
			}
		}
	}

	// --- Helper Functions ---

	// Helper function - checks if the instance supports validation layers
	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	// Helper function - creates a list of required extensions
	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	// Helper function - fills out a create info struct that filters debug messages
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	// Helper function for pickPhysicalDevice() - gets max usable msaa samples for the current physical device
	VkSampleCountFlagBits getMaxUsableSampleCount() {
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

		VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		if (counts & VK_SAMPLE_COUNT_64_BIT) return VK_SAMPLE_COUNT_64_BIT;
		if (counts & VK_SAMPLE_COUNT_32_BIT) return VK_SAMPLE_COUNT_32_BIT;
		if (counts & VK_SAMPLE_COUNT_16_BIT) return VK_SAMPLE_COUNT_16_BIT;
		if (counts & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
		if (counts & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
		if (counts & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;

		return VK_SAMPLE_COUNT_1_BIT;
	}

	// Helper function for pickPhysicalDevice() - given a physical device, returns if the device is suitable for this application
	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);
		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

		return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
	}

	// Helper function for isDeviceSuitable() - given a physical device, checks if it supports device extensions required by this application
	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	// Helper function for createLogicalDevice() - returns queue family indices for a physical device
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (presentSupport) {
				indices.presentFamily = i;
			}

			if (indices.isComplete()) {
				break;
			}

			i++;
		}

		return indices;
	}

	// Helper function for createSwapChain() - given a device, fetches capabilities and support and returns them as a struct
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	// Helper function for createSwapChain() - check if our desired format (SRGB, B8G8R8A8)  is supported
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	// Helper function for createSwapChain() - check if our desired present more (MAILBOX) is supported
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	// Helper function for createSwapChain() - returns an extent, matching the resolution of the window
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	// Helper function for createGraphicsPipeline() - reads a file by filename, returns a vector of characters
	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}

	// Helper function for createGraphicsPipeline() - loads a SPIR-V file into a shader module
	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}

	// Helper function for creating any buffer - given a set of properties, creates a buffer and stores it at the given address
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	// Helper function for createVertexAndIndexBuffers() - copies a buffer from a source to a destination on the GPU
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	// Helper function for any command buffer - allocates a command buffer and calls vkBegincommandBuffer
	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	// Helper function for any command buffer - ends a command buffer, submits it, then frees the command buffer
	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	// Helper function used in findDepthFormat() - checks if the physical device supports the desired depth format
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	// Helper function for createDepthResources() - returns the format required to create an image
	VkFormat findDepthFormat() {
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}


	void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {

		// Check if device supports linear blitting for image format
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

		//Well.. I'm using VkFilter nearest, but ill include this for practice..
		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++) {
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = { 0,0,0 };
			blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			vkCmdBlitImage(commandBuffer,
				image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				VK_FILTER_NEAREST);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		endSingleTimeCommands(commandBuffer);
	}

	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = static_cast<uint32_t>(width);
		imageInfo.extent.height = static_cast<uint32_t>(height);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples = numSamples;
		imageInfo.flags = 0;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		endSingleTimeCommands(commandBuffer);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			width,
			height,
			1
		};

		vkCmdCopyBufferToImage(
			commandBuffer,
			buffer,
			image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&region
		);

		endSingleTimeCommands(commandBuffer);
	}

	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = mipLevels;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}

		return imageView;
	}
};

int main() {
	VoxelEngine app;

	try {
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}