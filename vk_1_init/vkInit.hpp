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

    //drawFrame函数用于执行下面的操作：
    //从交换链获取一张图像
    //对帧缓冲附着执行指令缓冲中的渲染指令
    //返回渲染后的图像到交换链进行呈现操作
    
    //上面这些操作每一个都是通过一个函数调用设置的,但每个操作的实际执行却是  异步  进行的。
    //函数调用会在操作实际结束前返回，并且操作的实际执行顺序也是不确定的。
    //而操作的执行能按照一定的顺序，所以就需要进行  同步  操作。
    //有两种用于  同步  交换链事件的方式：栅栏(fence)和信号量(semaphore)。它们都可以完成同步操作。

    //栅栏(fence)和信号量(semaphore)的不同之处，
    //可以通过调用vkWaitForFences函数查询栅栏(fence)的状态，但不能查询信号量(semaphore)的状态。
    //通常使用栅栏(fence)来对应用程序本身和渲染操作进行同步。
    //使用信号量(semaphore)来对一个指令队列内的操作或多个不同指令队列的操作进行同步。
    //这里应该通过指令队列中的绘制操作和呈现操作，使用信号量(semaphore)更加合适。
    void drawFrame() {
        //vkWaitForFences函数可以用来等待一组栅栏(fence)中的一个或全部栅栏(fence)发出信号
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        //从交换链中获取图像
        uint32_t imageIndex;
        //第一个参数是使用的逻辑设备对象
        //第二个参数是我们要获取图像的交换链
        //第三个参数是图像获取的超时时间，可以通过使用无符号64位整型所能表示的最大整数来禁用图像获取超时
        //接下来两个参数指定图像可用后通知的同步对象，可以指定一个信号量对象或栅栏对象，或是同时指定信号量和栅栏对象进行同步操作
        //最后一个参数用于输出可用的交换链图像的索引
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
            imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        //等待栅栏发出信号后，需要调用vkResetFences函数手动将栅栏(fence)重置为未发出信号的状态
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        
        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        //提交指令缓冲
        //提交信息给指令队列
        VkSubmitInfo submitInfo = {}; 
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1; //指定队列开始执行前需要等待的信号量
        submitInfo.pWaitSemaphores = waitSemaphores; //需要等待的管线阶段
        submitInfo.pWaitDstStageMask = waitStages;
        //指定实际被提交执行的指令缓冲对象
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        //应该提交和刚刚获取的交换链图像相对应的指令缓冲对象

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] }; //在指令缓冲执行结束后发出信号
        //指定在指令缓冲执行结束后发出信号的信号量对象
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores; 

        //提交指令缓冲给图形指令队列
        //vkQueueSubmit函数使用vkQueueSubmit结构体数组作为参数，可以同时大批量提交数据
        //vkQueueSubmit函数的最后一个参数是一个可选的栅栏对象，可以用它同步提交的指令缓冲执行结束后要进行的操作
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        //渲染操作执行后，需要将渲染的图像返回给交换链进行呈现操作
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        //指定开始呈现操作需要等待的信号量
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        
        //指定用于呈现图像的交换链，以及需要呈现的图像在交换链中的索引
        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        presentInfo.pResults = nullptr; // Optional //可以通过pResults成员变量获取每个交换链的呈现操作是否成功的信息

        //请求交换链进行图像呈现操作
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
        
    VkQueue graphicsQueue; //图形队列
    VkQueue presentQueue;  //呈现队列
    VkQueue computeQueue;  //计算队列

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

    //创建渲染流程对象
    VkRenderPass renderPass;
    //在着色器中使用的uniform变量需要在管线创建时使用VkPipelineLayout对象定义
    VkPipelineLayout pipelineLayout;
    //添加一个VkPipeline成员变量来存储创建的管线对象
    VkPipeline graphicsPipeline;

    //添加一个向量作为成员变量来存储所有帧缓冲对象
    std::vector<VkFramebuffer> swapChainFramebuffers;

    //指令池对象
    VkCommandPool commandPool;
    //指令缓冲
    std::vector<VkCommandBuffer> commandBuffers;

    //需要两个信号量
    std::vector<VkSemaphore> imageAvailableSemaphores; //发出图像已经被获取，可以开始渲染的信号
    std::vector<VkSemaphore> renderFinishedSemaphores; //发出渲染已经结果，可以开始呈现的信号
    std::vector<VkFence> inFlightFences; //用来发出信号和等待信号
    uint32_t currentFrame = 0;

    bool framebufferResized;



    //创建vulkan实例
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

    //设置debug层
    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    //创建窗口表面（window surface）
    void createSurface() {
        if (glfwCreateWindowSurface(instance, vkWindow, nullptr, &surface)) {
            throw std::runtime_error("failed to create window surface!\n");
        }
    }

    //获取并选择物理设备
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) throw std::runtime_error("failed to find out physical device!\n");
        //为物理设备分配数组以用于存储VkPhysicalDevice
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

    //创建逻辑设备
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

    //创建交换链
    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        //使用交换链支持的最小图像个数+1数量的图像来实现三倍缓冲
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
            //VK_SHARING_MODE_EXCLUSIVE：一张图像同一时间只能被一个队列族所拥有，在另一队列族使用它之前，必须显式地改变图像所有权。这一模式下性能表现最佳。
            //VK_SHARING_MODE_CONCURRENT：图像可以在多个队列族间使用，不需要显式地改变图像所有权。
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
        //分配足够的数组空间来存储图像视图
        swapChainImageViews.resize(swapChainImages.size());
        //遍历所有交换链图像，创建图像视图
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            //填写结构体
            VkImageViewCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; //图像方式，图像被看作texture1D,texture2D,texture3D 
            createInfo.format = swapChainImageFormat;    //图像格式，指定图像数据的解释方式
            //颜色通道映射
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            //指定图像的用途和图像的哪一部分可以被访问：图像被用作渲染目标，并且没有细分级别，只存在一个图层
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
            //创建图像视图
            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }

    }

    void createRenderPass() {
        //附件描述
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChainImageFormat; //指定颜色缓冲附着的格式
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; //指定采样数，没有多重采样，所以为1
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; //设置为VK_ATTACHMENT_LOAD_OP_CLEAR，在每次渲染新的一帧前使用黑色清除帧缓冲
        //指定在渲染之前对附着中的数据进行的操作
        //VK_ATTACHMENT_LOAD_OP_LOAD：保持附着的现有内容
        //VK_ATTACHMENT_LOAD_OP_CLEAR：使用一个常量值来清除附着的内容
        //VK_ATTACHMENT_LOAD_OP_DONT_CARE：不关心附着现存的内容
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        //指定在渲染之后对附着中的数据进行的操作
        //VK_ATTACHMENT_STORE_OP_STORE：渲染的内容会被存储起来，以便之后读取
        //VK_ATTACHMENT_STORE_OP_DONT_CARE：渲染后，不会读取帧缓冲的内容
        //loadOp和storeOp成员变量的设置会对颜色和深度缓冲起效
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        //stencilLoadOp成员变量和stencilStoreOp成员变量会对模板缓冲起效
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        //图形内存布局设置
        //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL：图像被用作颜色附着
        //VK_IMAGE_LAYOUT_PRESENT_SRC_KHR：图像被用在交换链中进行呈现操作
        //VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL：图像被用作复制操作的目的图像

        //一个渲染流程可以包含多个子流程。子流程依赖于上一流程处理后的帧缓冲内容。
        //比如，许多叠加的后期处理效果就是在上一次的处理结果上进行的。我们将多个子流程组成一个渲染流程后，
        //Vulkan可以对其进行一定程度的优化。对于我们这个渲染三角形的程序，我们只使用了一个子流程。
        //每个子流程可以引用一个或多个附着，这些引用的附着是通过VkAttachmentReference结构体指定的：
        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0; //指定要引用的附着在附着描述结构体数组中的索引
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; //指定进行子流程时引用的附着使用的布局方式

        //描述子流程
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        //这里设置的颜色附着在数组中的索引会被片段着色器使用，对应片段着色器中使用的 layout(location = 0) out vec4 outColor语句
        //其它一些可以被子流程引用的附着类型
        //pInputAttachments：被着色器读取的附着
        //pResolveAttachments：用于多重采样的颜色附着
        //pDepthStencilAttachment：用于深度和模板数据的附着
        //pPreserveAttachments：没有被这一子流程使用，但需要保留数据的附着

        //子流程依赖
        //渲染流程的子流程会自动进行图像布局变换。这一变换过程由子流程的依赖所决定。
        //子流程的依赖包括子流程之间的内存和执行的依赖关系。
        //虽然我们现在只使用了一个子流程，但子流程执行之前和子流程执行之后的操作也被算作隐含的子流程。

        //在渲染流程开始和结束时会自动进行图像布局变换，但在渲染流程开始时进行的自动变换的时机和我们的需求不符，
        //变换发生在管线开始时，但那时我们可能还没有获取到交换链图像。有两种方式可以解决这个问题。
        //1.一个是设置imageAvailableSemaphore信号量的waitStages为VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT，
        // 确保渲染流程在我们获取交换链图像之前不会开始。
        //2.一个是设置渲染流程等待VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT管线阶段。

        //配置子流程依赖
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL; //指定被依赖的子流程的索引
        dependency.dstSubpass = 0; //指定依赖被依赖的子流程的索引
        //对srcSubpass成员变量使用表示渲染流程开始前的子流程，对dstSubpass成员使用表示渲染流程结束后的子流程

        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; //指定需要等待的管线阶段将进行的操作类型
        dependency.srcAccessMask = 0; //指定子流程将进行的操作类型
        //需要等待交换链结束对图像的读取才能对图像进行访问操作，也就是等待颜色附着输出这一管线阶段

        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; //指定需要等待的管线阶段将进行的操作类型
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; //指定子流程将进行的操作类型
        //设置为等待颜色附着的输出阶段，子流程将会进行颜色附着的读写操作
        //这样设置后，图像布局变换直到必要时才会进行：当我们开始写入颜色数据时


        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        //指定渲染流程使用的依赖信息
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

    }

    //创建图形渲染管线
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
        //顶点数据
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{}; 
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        
        //输入装配
        {
            //VK_PRIMITIVE_TOPOLOGY_POINT_LIST：点图元
            //VK_PRIMITIVE_TOPOLOGY_LINE_LIST：每两个顶点构成一个线段图元
            //VK_PRIMITIVE_TOPOLOGY_LINE_STRIP：每两个顶点构成一个线段图元，除第一个线段图元外，每个线段图元使用上一个线段图元的一个顶点
            //VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST：每三个顶点构成一个三角形图元
            //VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP：每个三角形的第二个和第三个顶点被下一个三角形作为第一和第二个顶点使用
        }
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; //图元类型，点、线、三角形
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        //视口裁剪
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        //光栅化
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE; //渲染看不见的图元（depth > 1 或 < 0）
        rasterizer.rasterizerDiscardEnable = VK_FALSE; //禁止一切片段输出到帧缓冲
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; //指定几何图元生成片段的方式 
        {
        //VK_POLYGON_MODE_FILL：整个多边形，包括多边形内部都产生片段
        //VK_POLYGON_MODE_LINE：只有多边形的边会产生片段
        //VK_POLYGON_MODE_POINT：只有多边形的顶点会产生片段
            //使用除了VK_POLYGON_MODE_FILL外的模式，需要启用相应的GPU特性。
        }
        rasterizer.lineWidth = 1.0f; //用于指定光栅化后的线段宽度，线段在光栅化过程中被赋予的像素宽度
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; //用于指定使用的表面剔除类型，可以通过它禁用表面剔除，剔除背面，剔除正面，以及剔除双面
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; //用于指定顺时针的顶点序是正面，还是逆时针的顶点序是正面
        rasterizer.depthBiasEnable = VK_FALSE;

        //多重采样
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        //颜色混合，片段着色器返回的片段颜色需要和原来帧缓冲中对应像素的颜色进行混合
        // (1.混合旧值和新值产生最终的颜色 2.使用位运算组合旧值和新值)
        //对每个绑定的帧缓冲进行单独的颜色混合配置
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | 
                                              VK_COLOR_COMPONENT_G_BIT | 
                                              VK_COLOR_COMPONENT_B_BIT | 
                                              VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        //进行全局的颜色混合配置
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        //如果想要使用第二种混合方式(位运算)，那么就需要将logicOpEnable成员变量设置为 VK_TRUE，然后使用logicOp成员变量指定要使用的位运算
        colorBlending.logicOp = VK_LOGIC_OP_COPY; 
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment; 
        //使用一个VkPipelineColorBlendAttachmentState结构体数组指针来指定每个帧缓冲的颜色混合设置
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        //管线动态状态
        //只有非常有限的管线状态可以在不重建管线的情况下进行动态修改。包括视口大小，线宽和混合常量。
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        //指定需要动态修改的状态
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        //需要创建一个VkPipelineLayout对象，指定空的管线布局
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }


        //为了创建图形管线而创建的对象：
        //1.着色器阶段：定义了着色器模块用于图形管线哪一可编程阶段
        //2.固定功能状态：定义了图形管线的固定功能阶段使用的状态信息，比如输入装配，视口，光栅化，颜色混合
        //3.管线布局：定义了被着色器使用，在渲染时可以被动态修改的uniform变量
        //4.渲染流程：定义了被管线使用的附着附着的用途
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

    //对帧缓冲进行配置
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size()); //分配足够的空间来存储所有帧缓冲对象

        //为交换链的每一个图像视图对象创建对应的帧缓冲
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1; //用于指定附着个数
            framebufferInfo.pAttachments = attachments; //渲染流程对象用于描述附着信息的pAttachment数组
            //width和height成员变量用于指定帧缓冲的大小，layers成员变量用于指定图像层数
            framebufferInfo.width = swapChainExtent.width; 
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1; //使用的交换链图像都是单层的，所以将layers成员变量设置为1

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }

        //首先指定帧缓冲需要兼容的渲染流程对象
        //之后的渲染操作，可以使用与这个指定的渲染流程对象相兼容的其它渲染流程对象
        //一般来说，使用相同数量，相同类型附着的渲染流程对象是相兼容的
    }

    //指令池
    void createCommandPool() {
        //指令池对象的创建
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
        //指令缓冲对象在被提交给我们之前获取的队列后，被Vulkan执行。每个指令池对象分配的指令缓冲对象只能提交给一个特定类型的队列。在这里，我们使用的是绘制指令，它可以被提交给支持图形操作的队列。
        //有下面两种用于指令池对象创建的标记，可以提供有用的信息给Vulkan的驱动程序进行一定优化处理：
        //VK_COMMAND_POOL_CREATE_TRANSIENT_BIT：使用它分配的指令缓冲对象被频繁用来记录新的指令(使用这一标记可能会改变帧缓冲对象的内存分配策略)。
        //VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT：指令缓冲对象之间相互独立，不会被一起重置。不使用这一标记，指令缓冲对象会被放在一起重置。

        //只在程序初始化时记录指令到指令缓冲对象，然后在程序的主循环中执行指令，所以不使用上面这两个标记
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }

    }

    //指令缓冲
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
        //level成员变量用于指定分配的指令缓冲对象是主要指令缓冲对象还是辅助指令缓冲对象：
        //VK_COMMAND_BUFFER_LEVEL_PRIMARY：可以被提交到队列进行执行，但不能被其它指令缓冲对象调用。
        //VK_COMMAND_BUFFER_LEVEL_SECONDARY：不能直接被提交到队列进行执行，但可以被主要指令缓冲对象调用执行

    }

    //
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        //VkCommandBufferBeginInfo结构体作为参数来指定一些有关指令缓冲的使用细节
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; 

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
        //beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT; 在上一帧还未结束渲染时，提交下一帧的渲染指令
        //flags成员变量用于指定我们将要怎样使用指令缓冲。它的值可以是下面这些：
            //VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT：指令缓冲在执行一次后，就被用来记录新的指令。
            //VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT：这是一个只在一个渲染流程内使用的辅助指令缓冲。
            //VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT：在指令缓冲等待执行时，仍然可以提交这一指令缓冲。


        //使用VkRenderPassBeginInfo结构体来指定使用的渲染流程对象
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass; //指定使用的渲染流程对象
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex]; //指定使用的帧缓冲对象
        //指定用于渲染的区域，位于这一区域外的像素数据会处于未定义状态
        //将这一区域设置为和我们使用的附着大小完全一样
        renderPassInfo.renderArea.offset = { 0, 0 }; 
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} }; //使用完全不透明的黑色作为清除值
        //clearValueCount和pClearValues成员变量用于指定使用VK_ATTACHMENT_LOAD_OP_CLEAR标记后，使用的清除值
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;


        //所有可以记录指令到指令缓冲的函数的函数名都带有一个vkCmd前缀
        //并且这些函数的返回值都是void，也就是说在指令记录操作完全结束前，不用进行任何错误处理
        //1.第一个参数是用于记录指令的指令缓冲对象。
        //2.第二个参数是使用的渲染流程的信息。
        //3.最后一个参数是用来指定渲染流程如何提供绘制指令的标记，它可以是下面这两个值之一：
        //  VK_SUBPASS_CONTENTS_INLINE：所有要执行的指令都在主要指令缓冲中，没有辅助指令缓冲需要执行。
        //  VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS：有来自辅助指令缓冲的指令需要执行。

        //调用vkCmdBeginRenderPass函数可以开始一个渲染流程
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        //绑定图形管线，vkCmdBindPipeline函数的第二个参数用于指定管线对象是图形管线还是计算管线
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


        //使用vkCmdDraw函数来提交绘制操作到指令缓冲，
        //第一个参数是记录有要执行的指令的指令缓冲对象，它的剩余参数依次是：
        //vertexCount：尽管这里我们没有使用顶点缓冲，但仍然需要指定三个顶点用于三角形的绘制。
        //instanceCount：用于实例渲染，为1时表示不进行实例渲染。
        //firstVertex：用于定义着色器变量gl_VertexIndex的值。
        //firstInstance：用于定义着色器变量gl_InstanceIndex的值。
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        //调用vkCmdEndRenderPass函数结束渲染流程
        vkCmdEndRenderPass(commandBuffer);

        //结束记录指令到指令缓冲
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    //同步对象
    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        //创建信号量
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

    //重建交换链
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








    //创建shader模块
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

    //vulkan物理设备相关函数
    //检查物理设备是否符合需求
    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);
        //检测交换链的能力是否满足需求
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


    //vulkan交换链相关函数
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        //查询surface属性
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        //查询surface支持格式
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        //查询支持的呈现方式
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    //选择surface格式
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        {
        //每一个VkSurfaceFormatKHR条目包含了一个format和colorSpace成员变量。format成员变量用于指定颜色通道和存储类型。
        // 比如，如果fomat成员变量的值为VK_FORMAT_B8G8R8A8_UNORM表示我们以B，G，R和A的顺序，
        // 每个颜色通道用8位无符号整型数表示，总共每像素使用32位表示。colorSpace成员变量用来表示SRGB颜色空间是否被支持，
        // 是否使用VK_COLOR_SPACE_SRGB_NONLINEAR_KHR标志。
        // 需要注意VK_COLOR_SPACE_SRGB_NONLINEAR_KHR在之前的Vulkan规范中叫做VK_COLORSPACE_SRGB_NONLINEAR_KHR。
        //对于颜色空间，如果SRGB被支持，我们就使用SRGB，使用它可以得到更加准确的颜色表示。
        // 直接使用SRGB颜色有很大挑战，所以我们使用RGB作为颜色格式，这一格式可以通过VK_FORMAT_B8G8R8A8_UNORM宏指定。
        }

        //如果Vulkan返回了一个格式列表，检查想要设定的格式是否存在于这个列表中：
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    //呈现模式
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) {
        VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR; //先进先出呈现模式，保证一定可用

        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) { //三重缓冲呈现模式
                return availablePresentMode;
            }
            else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) { //直出呈现模式
                bestMode = availablePresentMode;
            }
        }

        return bestMode; 
    }

    //设置交换
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


    //vulkan校验层相关
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