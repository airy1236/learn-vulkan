#pragma once

#include <iostream>
#include <vector>
#include <format>
#include <optional>
#include <set>

#include <vulkan/vulkan.h>

#include "framework/shader/shader.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

const int MAX_FRAMES_IN_FLIGHT = 2;

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

class vkInit {
public:

    void initVulkan(GLFWwindow* window, bool FramebufferResized) {
        vkWindow = window;
        framebufferResized = FramebufferResized;

        createInstance();

        setupDebugMessenger();
    
        createSurface();
        
        pickPhysicalDevice();

        createLogicalDevice();

        createSwapChain();

        createImageViews();

        createRenderPass();

        createGraphicsPipeline();

        createFramebuffers();

        createCommandPool();

        createCommandBuffers();

        createSyncObjects();
    }

    //drawFrame��������ִ������Ĳ�����
    //�ӽ�������ȡһ��ͼ��
    //��֡���帽��ִ��ָ����е���Ⱦָ��
    //������Ⱦ���ͼ�񵽽��������г��ֲ���
    
    //������Щ����ÿһ������ͨ��һ�������������õ�,��ÿ��������ʵ��ִ��ȴ��  �첽  ���еġ�
    //�������û��ڲ���ʵ�ʽ���ǰ���أ����Ҳ�����ʵ��ִ��˳��Ҳ�ǲ�ȷ���ġ�
    //��������ִ���ܰ���һ����˳�����Ծ���Ҫ����  ͬ��  ������
    //����������  ͬ��  �������¼��ķ�ʽ��դ��(fence)���ź���(semaphore)�����Ƕ��������ͬ��������

    //դ��(fence)���ź���(semaphore)�Ĳ�֮ͬ����
    //����ͨ������vkWaitForFences������ѯդ��(fence)��״̬�������ܲ�ѯ�ź���(semaphore)��״̬��
    //ͨ��ʹ��դ��(fence)����Ӧ�ó��������Ⱦ��������ͬ����
    //ʹ���ź���(semaphore)����һ��ָ������ڵĲ���������ָͬ����еĲ�������ͬ����
    //����Ӧ��ͨ��ָ������еĻ��Ʋ����ͳ��ֲ�����ʹ���ź���(semaphore)���Ӻ��ʡ�
    void drawFrame() {
        //vkWaitForFences�������������ȴ�һ��դ��(fence)�е�һ����ȫ��դ��(fence)�����ź�
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        //�ӽ������л�ȡͼ��
        uint32_t imageIndex;
        //��һ��������ʹ�õ��߼��豸����
        //�ڶ�������������Ҫ��ȡͼ��Ľ�����
        //������������ͼ���ȡ�ĳ�ʱʱ�䣬����ͨ��ʹ���޷���64λ�������ܱ�ʾ���������������ͼ���ȡ��ʱ
        //��������������ָ��ͼ����ú�֪ͨ��ͬ�����󣬿���ָ��һ���ź��������դ�����󣬻���ͬʱָ���ź�����դ���������ͬ������
        //���һ����������������õĽ�����ͼ�������
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
            imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        //�ȴ�դ�������źź���Ҫ����vkResetFences�����ֶ���դ��(fence)����Ϊδ�����źŵ�״̬
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        
        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        //�ύָ���
        //�ύ��Ϣ��ָ�����
        VkSubmitInfo submitInfo = {}; 
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1; //ָ�����п�ʼִ��ǰ��Ҫ�ȴ����ź���
        submitInfo.pWaitSemaphores = waitSemaphores; //��Ҫ�ȴ��Ĺ��߽׶�
        submitInfo.pWaitDstStageMask = waitStages;
        //ָ��ʵ�ʱ��ύִ�е�ָ������
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        //Ӧ���ύ�͸ոջ�ȡ�Ľ�����ͼ�����Ӧ��ָ������

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] }; //��ָ���ִ�н����󷢳��ź�
        //ָ����ָ���ִ�н����󷢳��źŵ��ź�������
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores; 

        //�ύָ����ͼ��ָ�����
        //vkQueueSubmit����ʹ��vkQueueSubmit�ṹ��������Ϊ����������ͬʱ�������ύ����
        //vkQueueSubmit���������һ��������һ����ѡ��դ�����󣬿�������ͬ���ύ��ָ���ִ�н�����Ҫ���еĲ���
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        //��Ⱦ����ִ�к���Ҫ����Ⱦ��ͼ�񷵻ظ����������г��ֲ���
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        //ָ����ʼ���ֲ�����Ҫ�ȴ����ź���
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        
        //ָ�����ڳ���ͼ��Ľ��������Լ���Ҫ���ֵ�ͼ���ڽ������е�����
        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        presentInfo.pResults = nullptr; // Optional //����ͨ��pResults��Ա������ȡÿ���������ĳ��ֲ����Ƿ�ɹ�����Ϣ

        //���󽻻�������ͼ����ֲ���
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


    void destroy() {
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
    }


    static vkInit& Base() {
        return singleton;
    }

    VkInstance getInstance() const {
        return instance;
    }

    VkDevice getDevice() const {
        return device;
    }

    vkInit() = default;
    ~vkInit() {}

private:

    static vkInit singleton;
    
    GLFWwindow* vkWindow;

    VkInstance instance;

    VkDebugUtilsMessengerEXT debugMessenger;
    
    VkSurfaceKHR surface;

    VkDevice device;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        
    VkQueue graphicsQueue; //ͼ�ζ���
    VkQueue presentQueue;  //���ֶ���
    VkQueue computeQueue;  //�������

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    //������Ⱦ���̶���
    VkRenderPass renderPass;
    //����ɫ����ʹ�õ�uniform������Ҫ�ڹ��ߴ���ʱʹ��VkPipelineLayout������
    VkPipelineLayout pipelineLayout;
    //���һ��VkPipeline��Ա�������洢�����Ĺ��߶���
    VkPipeline graphicsPipeline;

    //���һ��������Ϊ��Ա�������洢����֡�������
    std::vector<VkFramebuffer> swapChainFramebuffers;

    //ָ��ض���
    VkCommandPool commandPool;
    //ָ���
    std::vector<VkCommandBuffer> commandBuffers;

    //��Ҫ�����ź���
    std::vector<VkSemaphore> imageAvailableSemaphores; //����ͼ���Ѿ�����ȡ�����Կ�ʼ��Ⱦ���ź�
    std::vector<VkSemaphore> renderFinishedSemaphores; //������Ⱦ�Ѿ���������Կ�ʼ���ֵ��ź�
    std::vector<VkFence> inFlightFences; //���������źź͵ȴ��ź�
    uint32_t currentFrame = 0;

    bool framebufferResized;



    //����vulkanʵ��
    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
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
            throw std::runtime_error("failed to create instance!");
        }
    }

    //����debug��
    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    //�������ڱ��棨window surface��
    void createSurface() {
        if (glfwCreateWindowSurface(instance, vkWindow, nullptr, &surface)) {
            throw std::runtime_error("failed to create window surface!\n");
        }
    }

    //��ȡ��ѡ�������豸
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) throw std::runtime_error("failed to find out physical device!\n");
        //Ϊ�����豸�������������ڴ洢VkPhysicalDevice
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        for (auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) 
            throw std::runtime_error("failed to find a suitable physical device!\n");

    }

    //�����߼��豸
    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.emplace_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures = {};

        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else createInfo.enabledLayerCount = 0;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    //����������
    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        //ʹ�ý�����֧�ֵ���Сͼ�����+1������ͼ����ʵ����������
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
            //VK_SHARING_MODE_EXCLUSIVE��һ��ͼ��ͬһʱ��ֻ�ܱ�һ����������ӵ�У�����һ������ʹ����֮ǰ��������ʽ�ظı�ͼ������Ȩ����һģʽ�����ܱ�����ѡ�
            //VK_SHARING_MODE_CONCURRENT��ͼ������ڶ���������ʹ�ã�����Ҫ��ʽ�ظı�ͼ������Ȩ��
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

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

    void createImageViews() {
        //�����㹻������ռ����洢ͼ����ͼ
        swapChainImageViews.resize(swapChainImages.size());
        //�������н�����ͼ�񣬴���ͼ����ͼ
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            //��д�ṹ��
            VkImageViewCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; //ͼ��ʽ��ͼ�񱻿���texture1D,texture2D,texture3D 
            createInfo.format = swapChainImageFormat;    //ͼ���ʽ��ָ��ͼ�����ݵĽ��ͷ�ʽ
            //��ɫͨ��ӳ��
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            //ָ��ͼ�����;��ͼ�����һ���ֿ��Ա����ʣ�ͼ��������ȾĿ�꣬����û��ϸ�ּ���ֻ����һ��ͼ��
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
            //����ͼ����ͼ
            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }

    }

    void createRenderPass() {
        //��������
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChainImageFormat; //ָ����ɫ���帽�ŵĸ�ʽ
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; //ָ����������û�ж��ز���������Ϊ1
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; //����ΪVK_ATTACHMENT_LOAD_OP_CLEAR����ÿ����Ⱦ�µ�һ֡ǰʹ�ú�ɫ���֡����
        //ָ������Ⱦ֮ǰ�Ը����е����ݽ��еĲ���
        //VK_ATTACHMENT_LOAD_OP_LOAD�����ָ��ŵ���������
        //VK_ATTACHMENT_LOAD_OP_CLEAR��ʹ��һ������ֵ��������ŵ�����
        //VK_ATTACHMENT_LOAD_OP_DONT_CARE�������ĸ����ִ������
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        //ָ������Ⱦ֮��Ը����е����ݽ��еĲ���
        //VK_ATTACHMENT_STORE_OP_STORE����Ⱦ�����ݻᱻ�洢�������Ա�֮���ȡ
        //VK_ATTACHMENT_STORE_OP_DONT_CARE����Ⱦ�󣬲����ȡ֡���������
        //loadOp��storeOp��Ա���������û����ɫ����Ȼ�����Ч
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        //stencilLoadOp��Ա������stencilStoreOp��Ա�������ģ�建����Ч
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        //ͼ���ڴ沼������
        //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL��ͼ��������ɫ����
        //VK_IMAGE_LAYOUT_PRESENT_SRC_KHR��ͼ�����ڽ������н��г��ֲ���
        //VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL��ͼ���������Ʋ�����Ŀ��ͼ��

        //һ����Ⱦ���̿��԰�����������̡���������������һ���̴�����֡�������ݡ�
        //���磬�����ӵĺ��ڴ���Ч����������һ�εĴ������Ͻ��еġ����ǽ�������������һ����Ⱦ���̺�
        //Vulkan���Զ������һ���̶ȵ��Ż����������������Ⱦ�����εĳ�������ֻʹ����һ�������̡�
        //ÿ�������̿�������һ���������ţ���Щ���õĸ�����ͨ��VkAttachmentReference�ṹ��ָ���ģ�
        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0; //ָ��Ҫ���õĸ����ڸ��������ṹ�������е�����
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; //ָ������������ʱ���õĸ���ʹ�õĲ��ַ�ʽ

        //����������
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        //�������õ���ɫ�����������е������ᱻƬ����ɫ��ʹ�ã���ӦƬ����ɫ����ʹ�õ� layout(location = 0) out vec4 outColor���
        //����һЩ���Ա����������õĸ�������
        //pInputAttachments������ɫ����ȡ�ĸ���
        //pResolveAttachments�����ڶ��ز�������ɫ����
        //pDepthStencilAttachment��������Ⱥ�ģ�����ݵĸ���
        //pPreserveAttachments��û�б���һ������ʹ�ã�����Ҫ�������ݵĸ���

        //����������
        //��Ⱦ���̵������̻��Զ�����ͼ�񲼾ֱ任����һ�任�����������̵�������������
        //�����̵���������������֮����ڴ��ִ�е�������ϵ��
        //��Ȼ��������ֻʹ����һ�������̣���������ִ��֮ǰ��������ִ��֮��Ĳ���Ҳ�����������������̡�

        //����Ⱦ���̿�ʼ�ͽ���ʱ���Զ�����ͼ�񲼾ֱ任��������Ⱦ���̿�ʼʱ���е��Զ��任��ʱ�������ǵ����󲻷���
        //�任�����ڹ��߿�ʼʱ������ʱ���ǿ��ܻ�û�л�ȡ��������ͼ�������ַ�ʽ���Խ��������⡣
        //1.һ��������imageAvailableSemaphore�ź�����waitStagesΪVK_PIPELINE_STAGE_TOP_OF_PIPE_BIT��
        // ȷ����Ⱦ���������ǻ�ȡ������ͼ��֮ǰ���Ὺʼ��
        //2.һ����������Ⱦ���̵ȴ�VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT���߽׶Ρ�

        //��������������
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL; //ָ���������������̵�����
        dependency.dstSubpass = 0; //ָ�������������������̵�����
        //��srcSubpass��Ա����ʹ�ñ�ʾ��Ⱦ���̿�ʼǰ�������̣���dstSubpass��Աʹ�ñ�ʾ��Ⱦ���̽������������

        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; //ָ����Ҫ�ȴ��Ĺ��߽׶ν����еĲ�������
        dependency.srcAccessMask = 0; //ָ�������̽����еĲ�������
        //��Ҫ�ȴ�������������ͼ��Ķ�ȡ���ܶ�ͼ����з��ʲ�����Ҳ���ǵȴ���ɫ���������һ���߽׶�

        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; //ָ����Ҫ�ȴ��Ĺ��߽׶ν����еĲ�������
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; //ָ�������̽����еĲ�������
        //����Ϊ�ȴ���ɫ���ŵ�����׶Σ������̽��������ɫ���ŵĶ�д����
        //�������ú�ͼ�񲼾ֱ任ֱ����Ҫʱ�Ż���У������ǿ�ʼд����ɫ����ʱ


        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        //ָ����Ⱦ����ʹ�õ�������Ϣ
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

    }

    //����ͼ����Ⱦ����
    void createGraphicsPipeline() {
        Shader vertex("assets/shaders/triangle.vert", shaderc_glsl_vertex_shader);
        Shader fragment("assets/shaders/triangle.frag", shaderc_glsl_fragment_shader);

        VkShaderModule vertShaderModule = createShaderModule(vertex.data());
        VkShaderModule fragShaderModule = createShaderModule(fragment.data());

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

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };
        //��������
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{}; 
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        
        //����װ��
        {
            //VK_PRIMITIVE_TOPOLOGY_POINT_LIST����ͼԪ
            //VK_PRIMITIVE_TOPOLOGY_LINE_LIST��ÿ�������㹹��һ���߶�ͼԪ
            //VK_PRIMITIVE_TOPOLOGY_LINE_STRIP��ÿ�������㹹��һ���߶�ͼԪ������һ���߶�ͼԪ�⣬ÿ���߶�ͼԪʹ����һ���߶�ͼԪ��һ������
            //VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST��ÿ�������㹹��һ��������ͼԪ
            //VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP��ÿ�������εĵڶ����͵��������㱻��һ����������Ϊ��һ�͵ڶ�������ʹ��
        }
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; //ͼԪ���ͣ��㡢�ߡ�������
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        //�ӿڲü�
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        //��դ��
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE; //��Ⱦ��������ͼԪ��depth > 1 �� < 0��
        rasterizer.rasterizerDiscardEnable = VK_FALSE; //��ֹһ��Ƭ�������֡����
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; //ָ������ͼԪ����Ƭ�εķ�ʽ 
        {
        //VK_POLYGON_MODE_FILL����������Σ�����������ڲ�������Ƭ��
        //VK_POLYGON_MODE_LINE��ֻ�ж���εı߻����Ƭ��
        //VK_POLYGON_MODE_POINT��ֻ�ж���εĶ�������Ƭ��
            //ʹ�ó���VK_POLYGON_MODE_FILL���ģʽ����Ҫ������Ӧ��GPU���ԡ�
        }
        rasterizer.lineWidth = 1.0f; //����ָ����դ������߶ο�ȣ��߶��ڹ�դ�������б���������ؿ��
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; //����ָ��ʹ�õı����޳����ͣ�����ͨ�������ñ����޳����޳����棬�޳����棬�Լ��޳�˫��
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; //����ָ��˳ʱ��Ķ����������棬������ʱ��Ķ�����������
        rasterizer.depthBiasEnable = VK_FALSE;

        //���ز���
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        //��ɫ��ϣ�Ƭ����ɫ�����ص�Ƭ����ɫ��Ҫ��ԭ��֡�����ж�Ӧ���ص���ɫ���л��
        // (1.��Ͼ�ֵ����ֵ�������յ���ɫ 2.ʹ��λ������Ͼ�ֵ����ֵ)
        //��ÿ���󶨵�֡������е�������ɫ�������
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | 
                                              VK_COLOR_COMPONENT_G_BIT | 
                                              VK_COLOR_COMPONENT_B_BIT | 
                                              VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        //����ȫ�ֵ���ɫ�������
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        //�����Ҫʹ�õڶ��ֻ�Ϸ�ʽ(λ����)����ô����Ҫ��logicOpEnable��Ա��������Ϊ VK_TRUE��Ȼ��ʹ��logicOp��Ա����ָ��Ҫʹ�õ�λ����
        colorBlending.logicOp = VK_LOGIC_OP_COPY; 
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment; 
        //ʹ��һ��VkPipelineColorBlendAttachmentState�ṹ������ָ����ָ��ÿ��֡�������ɫ�������
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        //���߶�̬״̬
        //ֻ�зǳ����޵Ĺ���״̬�����ڲ��ؽ����ߵ�����½��ж�̬�޸ġ������ӿڴ�С���߿�ͻ�ϳ�����
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        //ָ����Ҫ��̬�޸ĵ�״̬
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        //��Ҫ����һ��VkPipelineLayout����ָ���յĹ��߲���
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }


        //Ϊ�˴���ͼ�ι��߶������Ķ���
        //1.��ɫ���׶Σ���������ɫ��ģ������ͼ�ι�����һ�ɱ�̽׶�
        //2.�̶�����״̬��������ͼ�ι��ߵĹ̶����ܽ׶�ʹ�õ�״̬��Ϣ����������װ�䣬�ӿڣ���դ������ɫ���
        //3.���߲��֣������˱���ɫ��ʹ�ã�����Ⱦʱ���Ա���̬�޸ĵ�uniform����
        //4.��Ⱦ���̣������˱�����ʹ�õĸ��Ÿ��ŵ���;
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }


        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }   

    //��֡�����������
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size()); //�����㹻�Ŀռ����洢����֡�������

        //Ϊ��������ÿһ��ͼ����ͼ���󴴽���Ӧ��֡����
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1; //����ָ�����Ÿ���
            framebufferInfo.pAttachments = attachments; //��Ⱦ���̶�����������������Ϣ��pAttachment����
            //width��height��Ա��������ָ��֡����Ĵ�С��layers��Ա��������ָ��ͼ�����
            framebufferInfo.width = swapChainExtent.width; 
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1; //ʹ�õĽ�����ͼ���ǵ���ģ����Խ�layers��Ա��������Ϊ1

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }

        //����ָ��֡������Ҫ���ݵ���Ⱦ���̶���
        //֮�����Ⱦ����������ʹ�������ָ������Ⱦ���̶�������ݵ�������Ⱦ���̶���
        //һ����˵��ʹ����ͬ��������ͬ���͸��ŵ���Ⱦ���̶���������ݵ�
    }

    //ָ���
    void createCommandPool() {
        //ָ��ض���Ĵ���
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
        //ָ�������ڱ��ύ������֮ǰ��ȡ�Ķ��к󣬱�Vulkanִ�С�ÿ��ָ��ض�������ָ������ֻ���ύ��һ���ض����͵Ķ��С����������ʹ�õ��ǻ���ָ������Ա��ύ��֧��ͼ�β����Ķ��С�
        //��������������ָ��ض��󴴽��ı�ǣ������ṩ���õ���Ϣ��Vulkan�������������һ���Ż�����
        //VK_COMMAND_POOL_CREATE_TRANSIENT_BIT��ʹ���������ָ������Ƶ��������¼�µ�ָ��(ʹ����һ��ǿ��ܻ�ı�֡���������ڴ�������)��
        //VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT��ָ������֮���໥���������ᱻһ�����á���ʹ����һ��ǣ�ָ������ᱻ����һ�����á�

        //ֻ�ڳ����ʼ��ʱ��¼ָ�ָ������Ȼ���ڳ������ѭ����ִ��ָ����Բ�ʹ���������������
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }

    }

    //ָ���
    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
        //level��Ա��������ָ�������ָ����������Ҫָ�������Ǹ���ָ������
        //VK_COMMAND_BUFFER_LEVEL_PRIMARY�����Ա��ύ�����н���ִ�У������ܱ�����ָ��������á�
        //VK_COMMAND_BUFFER_LEVEL_SECONDARY������ֱ�ӱ��ύ�����н���ִ�У������Ա���Ҫָ���������ִ��

    }

    //
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        //VkCommandBufferBeginInfo�ṹ����Ϊ������ָ��һЩ�й�ָ����ʹ��ϸ��
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; 

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
        //beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT; ����һ֡��δ������Ⱦʱ���ύ��һ֡����Ⱦָ��
        //flags��Ա��������ָ�����ǽ�Ҫ����ʹ��ָ��塣����ֵ������������Щ��
            //VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT��ָ�����ִ��һ�κ󣬾ͱ�������¼�µ�ָ�
            //VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT������һ��ֻ��һ����Ⱦ������ʹ�õĸ���ָ��塣
            //VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT����ָ���ȴ�ִ��ʱ����Ȼ�����ύ��һָ��塣


        //ʹ��VkRenderPassBeginInfo�ṹ����ָ��ʹ�õ���Ⱦ���̶���
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass; //ָ��ʹ�õ���Ⱦ���̶���
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex]; //ָ��ʹ�õ�֡�������
        //ָ��������Ⱦ������λ����һ��������������ݻᴦ��δ����״̬
        //����һ��������Ϊ������ʹ�õĸ��Ŵ�С��ȫһ��
        renderPassInfo.renderArea.offset = { 0, 0 }; 
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} }; //ʹ����ȫ��͸���ĺ�ɫ��Ϊ���ֵ
        //clearValueCount��pClearValues��Ա��������ָ��ʹ��VK_ATTACHMENT_LOAD_OP_CLEAR��Ǻ�ʹ�õ����ֵ
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;


        //���п��Լ�¼ָ�ָ���ĺ����ĺ�����������һ��vkCmdǰ׺
        //������Щ�����ķ���ֵ����void��Ҳ����˵��ָ���¼������ȫ����ǰ�����ý����κδ�����
        //1.��һ�����������ڼ�¼ָ���ָ������
        //2.�ڶ���������ʹ�õ���Ⱦ���̵���Ϣ��
        //3.���һ������������ָ����Ⱦ��������ṩ����ָ��ı�ǣ�������������������ֵ֮һ��
        //  VK_SUBPASS_CONTENTS_INLINE������Ҫִ�е�ָ�����Ҫָ����У�û�и���ָ�����Ҫִ�С�
        //  VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS�������Ը���ָ����ָ����Ҫִ�С�

        //����vkCmdBeginRenderPass�������Կ�ʼһ����Ⱦ����
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        //��ͼ�ι��ߣ�vkCmdBindPipeline�����ĵڶ�����������ָ�����߶�����ͼ�ι��߻��Ǽ������
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);


        //ʹ��vkCmdDraw�������ύ���Ʋ�����ָ��壬
        //��һ�������Ǽ�¼��Ҫִ�е�ָ���ָ����������ʣ����������ǣ�
        //vertexCount��������������û��ʹ�ö��㻺�壬����Ȼ��Ҫָ�������������������εĻ��ơ�
        //instanceCount������ʵ����Ⱦ��Ϊ1ʱ��ʾ������ʵ����Ⱦ��
        //firstVertex�����ڶ�����ɫ������gl_VertexIndex��ֵ��
        //firstInstance�����ڶ�����ɫ������gl_InstanceIndex��ֵ��
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        //����vkCmdEndRenderPass����������Ⱦ����
        vkCmdEndRenderPass(commandBuffer);

        //������¼ָ�ָ���
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    //ͬ������
    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        //�����ź���
        VkSemaphoreCreateInfo semaphoreInfo = {};
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

    //�ؽ�������
    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(vkWindow, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(vkWindow, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    void cleanupSwapChain() {
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }








    //����shaderģ��
    VkShaderModule createShaderModule(std::vector<uint32_t> code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size() * sizeof(uint32_t);
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    //vulkan�����豸��غ���
    //��������豸�Ƿ��������
    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);
        //��⽻�����������Ƿ���������
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return true;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.graphicsFamily = i;
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) indices.presentFamily = i;
            if (indices.isComplete()) break;
            ++i;
        }

        return indices;
    }


    //vulkan��������غ���
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        //��ѯsurface����
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        //��ѯsurface֧�ָ�ʽ
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        //��ѯ֧�ֵĳ��ַ�ʽ
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    //ѡ��surface��ʽ
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        {
        //ÿһ��VkSurfaceFormatKHR��Ŀ������һ��format��colorSpace��Ա������format��Ա��������ָ����ɫͨ���ʹ洢���͡�
        // ���磬���fomat��Ա������ֵΪVK_FORMAT_B8G8R8A8_UNORM��ʾ������B��G��R��A��˳��
        // ÿ����ɫͨ����8λ�޷�����������ʾ���ܹ�ÿ����ʹ��32λ��ʾ��colorSpace��Ա����������ʾSRGB��ɫ�ռ��Ƿ�֧�֣�
        // �Ƿ�ʹ��VK_COLOR_SPACE_SRGB_NONLINEAR_KHR��־��
        // ��Ҫע��VK_COLOR_SPACE_SRGB_NONLINEAR_KHR��֮ǰ��Vulkan�淶�н���VK_COLORSPACE_SRGB_NONLINEAR_KHR��
        //������ɫ�ռ䣬���SRGB��֧�֣����Ǿ�ʹ��SRGB��ʹ�������Եõ�����׼ȷ����ɫ��ʾ��
        // ֱ��ʹ��SRGB��ɫ�кܴ���ս����������ʹ��RGB��Ϊ��ɫ��ʽ����һ��ʽ����ͨ��VK_FORMAT_B8G8R8A8_UNORM��ָ����
        }

        //���Vulkan������һ����ʽ�б������Ҫ�趨�ĸ�ʽ�Ƿ����������б��У�
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    //����ģʽ
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) {
        VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR; //�Ƚ��ȳ�����ģʽ����֤һ������

        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) { //���ػ������ģʽ
                return availablePresentMode;
            }
            else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) { //ֱ������ģʽ
                bestMode = availablePresentMode;
            }
        }

        return bestMode; 
    }

    //���ý���
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(vkWindow, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

            return actualExtent;
        }
    }


    //vulkanУ������
    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr) return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        else return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) func(instance, debugMessenger, pAllocator);
    } 

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

            if (!layerFound) return false;
        }

        return true;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }

};

inline vkInit vkInit::singleton;