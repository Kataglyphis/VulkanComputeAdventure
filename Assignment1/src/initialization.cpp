#include <iostream>
#include <fstream>
#include <cstring>
#include <functional>
#include <optional>

#define VK_ENABLE_BETA_EXTENSIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "initialization.h"
#include "utils.h"

#include <optional>

// here you create the instance and physical / logical device and maybe compute/transfer queues
// also check if device is suitable etc

struct DeviceSelectionCache {
    uint32_t vendorID;
    uint32_t deviceID;
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char *> validationLayers = {
#ifndef NDEBUG
    "VK_LAYER_KHRONOS_validation"
#endif
};
const std::vector<const char *> instanceExtensions = {
#ifndef NDEBUG
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
};

const std::vector<const char *> extensionNames = {
    #ifndef NDEBUG
    
    #endif
} ;

void AppResources::destroy()
{
    this->device.destroyQueryPool(this->queryPool);
    //this->device.freeCommandBuffers(this->computeCommandPool, 1U, &this->computeCommandBuffer);
    //this->device.freeCommandBuffers(this->transferCommandPool, 1U, &this->transferCommandBuffer);
    this->device.destroyCommandPool(this->computeCommandPool);
    this->device.destroyCommandPool(this->transferCommandPool);
        
    this->device.destroy();

#ifndef NDEBUG
    this->instance.destroyDebugUtilsMessengerEXT(this->dbgUtilsMgr);
#endif
    this->instance.destroy();
}
void initApp(AppResources &app)
{
    createInstance(app.instance, app.dbgUtilsMgr, "Assignment1, Task 1", "Idkwhattowrite");

    selectPhysicalDevice(app.instance, app.pDevice);
    // printDeviceCapabilities(app.pDevice);
    std::tie(app.cQ, app.tQ) = getComputeAndTransferQueues(app.pDevice);
    createLogicalDevice(app.instance, app.pDevice, app.device);
    
    app.device.getQueue(app.cQ, 0U, &app.computeQueue);
    app.device.getQueue(app.tQ, 0U, &app.transferQueue);

    createCommandPool(app.device, app.computeCommandPool, app.cQ);
    createCommandPool(app.device, app.transferCommandPool, app.tQ);

    createTimestampQueryPool(app.device, app.queryPool, 2);
}

/* 
    This is the function in which errors will go through to be displayed.
*/
VKAPI_ATTR VkBool32 VKAPI_CALL
debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                            VkDebugUtilsMessengerCallbackDataEXT const *pCallbackData,
                            void * /*pUserData*/)
{
    if (enableValidationLayers)
    {
        if (pCallbackData->messageIdNumber == 648835635)
        {
            // UNASSIGNED-khronos-Validation-debug-build-warning-message
            return VK_FALSE;
        }
        if (pCallbackData->messageIdNumber == 767975156)
        {
            // UNASSIGNED-BestPractices-vkCreateInstance-specialuse-extension
            return VK_FALSE;
        }
    }

    std::cerr << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity)) << ": "
              << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageTypes)) << ":\n";
    std::cerr << "\t"
              << "messageIDName   = <" << pCallbackData->pMessageIdName << ">\n";
    std::cerr << "\t"
              << "messageIdNumber = " << pCallbackData->messageIdNumber << "\n";
    std::cerr << "\t"
              << "message         = <" << pCallbackData->pMessage << ">\n";
    if (0 < pCallbackData->queueLabelCount)
    {
        std::cerr << "\t"
                  << "Queue Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->queueLabelCount; i++)
        {
            std::cerr << "\t\t"
                      << "labelName = <" << pCallbackData->pQueueLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->cmdBufLabelCount)
    {
        std::cerr << "\t"
                  << "CommandBuffer Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->cmdBufLabelCount; i++)
        {
            std::cerr << "\t\t"
                      << "labelName = <" << pCallbackData->pCmdBufLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->objectCount)
    {
        std::cerr << "\t"
                  << "Objects:\n";
        for (uint8_t i = 0; i < pCallbackData->objectCount; i++)
        {
            std::cerr << "\t\t"
                      << "Object " << i << "\n";
            std::cerr << "\t\t\t"
                      << "objectType   = "
                      << vk::to_string(static_cast<vk::ObjectType>(pCallbackData->pObjects[i].objectType)) << "\n";
            std::cerr << "\t\t\t"
                      << "objectHandle = " << pCallbackData->pObjects[i].objectHandle << "\n";
            if (pCallbackData->pObjects[i].pObjectName)
            {
                std::cerr << "\t\t\t"
                          << "objectName   = <" << pCallbackData->pObjects[i].pObjectName << ">\n";
            }
        }
    }
    return VK_TRUE;
}

/*
    This function fills the structure with flags indicating 
    which error messages should go through
*/
vk::DebugUtilsMessengerCreateInfoEXT makeDebugUtilsMessengerCreateInfoEXT()
{

    using SEVERITY = vk::DebugUtilsMessageSeverityFlagBitsEXT; // for readability
    using MESSAGE = vk::DebugUtilsMessageTypeFlagBitsEXT;
    return {{},
            SEVERITY::eWarning | SEVERITY::eError,
            MESSAGE::eGeneral | MESSAGE::ePerformance | MESSAGE::eValidation,
            &debugUtilsMessengerCallback};
}

/*
    The dynamic loader allows us to access many extensions
    Required before creating instance for loading the extension VK_EXT_DEBUG_UTILS_EXTENSION_NAME
*/
void initDynamicLoader()
{
    static vk::DynamicLoader dl;
    static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
}


void createInstance(vk::Instance &instance, vk::DebugUtilsMessengerEXT &debugUtilsMessenger,
                    std::string appName, std::string engineName)
{
    initDynamicLoader();
    vk::ApplicationInfo applicationInfo(appName.c_str(), 1, engineName.c_str(), 1, VK_API_VERSION_1_2);

    // ### initialize the InstanceCreateInfo ###
    vk::InstanceCreateInfo instanceCreateInfo( //flags, pAppInfo, layerCount, layerNames, extcount, extNames
        {}, &applicationInfo,
        static_cast<uint32_t>(validationLayers.size()), validationLayers.data(),
        static_cast<uint32_t>(instanceExtensions.size()), instanceExtensions.data());

    // ### DebugInfo: use of StructureChain instead of pNext ###
    // DebugUtils is used to catch errors from the instance
    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo = makeDebugUtilsMessengerCreateInfoEXT();
    // the StructureChain fills the pNext member of the struct in a typesafe way
    // this is only possible with vulkan-hpp, in plain vulkan there is no typechecking
    vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> chain =
        {instanceCreateInfo, debugCreateInfo};

    if (!enableValidationLayers) // for Release mode
        chain.unlink<vk::DebugUtilsMessengerCreateInfoEXT>();

    // create an Instance
    instance = vk::createInstance(chain.get<vk::InstanceCreateInfo>());

    // update the dispatcher to use instance related extensions 
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance); 

    if (enableValidationLayers)
        debugUtilsMessenger = instance.createDebugUtilsMessengerEXT(makeDebugUtilsMessengerCreateInfoEXT());
}


std::tuple<uint32_t, uint32_t> getComputeAndTransferQueues(vk::PhysicalDevice &pDevice)
{
    uint32_t tq = -1;
    std::optional<uint32_t> otq;
    uint32_t cq = -1;
    std::optional<uint32_t> ocq;

    using Chain = vk::StructureChain<vk::QueueFamilyProperties2, vk::QueueFamilyCheckpointPropertiesNV>;
    using QFB = vk::QueueFlagBits;
    auto queueFamilyProperties2 = pDevice.getQueueFamilyProperties2<Chain, std::allocator<Chain>, vk::DispatchLoaderDynamic>();

    for (uint32_t j = 0; j < queueFamilyProperties2.size(); j++)
    {
        vk::QueueFamilyProperties const &properties =
            queueFamilyProperties2[static_cast<size_t>(j)].get<vk::QueueFamilyProperties2>().queueFamilyProperties;

        if (properties.queueFlags & QFB::eCompute)
        {
            if (!(properties.queueFlags & QFB::eGraphics ||
                  properties.queueFlags & QFB::eProtected))
                ocq = j; // when a queue supports only compute and not graphics we want to use that
            cq = j;
        }

        if (properties.queueFlags & QFB::eTransfer)
        {
            if (!(properties.queueFlags & QFB::eCompute ||
                  properties.queueFlags & QFB::eGraphics ||
                  properties.queueFlags & QFB::eProtected))
                otq = j; // when a queue supports only transfer, we want to use this one
            tq = j;
        }
    }

    if (otq.has_value())
        tq = otq.value();
    if (ocq.has_value())
        cq = ocq.value();
    return std::tuple<uint32_t, uint32_t>(cq, tq);
}
void selectPhysicalDevice(vk::Instance &instance, vk::PhysicalDevice &pDevice)
{
    // takes the first one
    std::vector<vk::PhysicalDevice> physDs = instance.enumeratePhysicalDevices();

    const static char* cache_name = "device_selection_cache";
    const static char* recreation_message = "To select a new device, delete the file \"device_selection_cache\" in your working directory before executing the framework.";

    std::ifstream ifile(cache_name, std::ios::binary);
    if (ifile.is_open()) {
        DeviceSelectionCache cache;
        ifile.read(reinterpret_cast<char*>(&cache), sizeof(cache));
        ifile.close();
        for (auto physD : physDs) {
            auto props = physD.getProperties2().properties;
            if (props.vendorID == cache.vendorID && props.deviceID == cache.deviceID) {
                std::cout << "Selecting previously selected device: \"" << props.deviceName << "\"" << std::endl;
                std::cout << recreation_message << std::endl;
                pDevice = physD;
                return;
            }
        }
        std::cout << "Previously selected device was not found." << std::endl;
    } else {
        std::cout << "No previous device selection found." << std::endl;
    }
    
    std::cout << "Select one of the available devices:" << std::endl;
    
    for (int i = 0; i < physDs.size(); i++) {
        auto props = physDs[i].getProperties2().properties;
        std::cout << i << ")\t" << props.deviceName.data() << std::endl;
    }
    
    uint32_t i;
    while (true) {
        std::cout << "Enter device number: ";
        std::cin >> i;
        if (i < physDs.size()) break;
    }
    
    auto props = physDs[i].getProperties2().properties;
    DeviceSelectionCache cache;
    cache.vendorID = props.vendorID;
    cache.deviceID = props.deviceID;

    std::ofstream ofile(cache_name, std::ios::out | std::ios::binary);
    ofile.write(reinterpret_cast<const char*>(&cache), sizeof(cache));
    ofile.close();
    std::cout << "Selected device: \"" << props.deviceName.data() << "\"" << std::endl
        << "This device will be automatically selected in the future." << std::endl
        << recreation_message << std::endl;

    pDevice = physDs[i];
}

/*
    The logical device holds the queues and will be used in almost every call from now on
*/
void createLogicalDevice(vk::Instance &instance, vk::PhysicalDevice &pDevice, vk::Device &device)
{

    //first get the queues
    uint32_t cQ, tQ;
    std::tie(cQ, tQ) = getComputeAndTransferQueues(pDevice);
    std::vector<vk::DeviceQueueCreateInfo> queuesInfo;
    // flags, queueFamily, queueCount, queuePriority
    float prio = 1.f;
    vk::DeviceQueueCreateInfo computeInfo({}, cQ, 1U, &prio);
    vk::DeviceQueueCreateInfo transferInfo({}, tQ, 1U, &prio);

    queuesInfo.push_back(computeInfo);
    queuesInfo.push_back(transferInfo);
    // {}, queueCreateInfoCount, pQueueCreateInfos, enabledLayerCount, ppEnabledLayerNames, enabledExtensionCount, ppEnabledExtensionNames, pEnabledFeatures

    std::vector<const char *> extensionNames_(extensionNames);
    
    auto deviceExtensionProperties = pDevice.enumerateDeviceExtensionProperties();
    bool enable_portability_subset = false;;
    for (auto ext : deviceExtensionProperties) {
        if (strcmp(ext.extensionName.data(), VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0) {
            enable_portability_subset = true;
        }
    }
    
    if (enable_portability_subset) {
        extensionNames_.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
    
    vk::DeviceCreateInfo dci({}, 2U, queuesInfo.data(), 
            CAST(validationLayers), validationLayers.data(),
            CAST(extensionNames_), extensionNames_.data() ); // no extension

    device = pDevice.createDevice(dci);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
    
    setObjectName(device, device, "This is my lovely device !");
}
void createCommandPool(vk::Device &device, vk::CommandPool &commandPool, uint32_t queueIndex)
{
    vk::CommandPoolCreateInfo cpi(vk::CommandPoolCreateFlags(), queueIndex);
    commandPool = device.createCommandPool(cpi);
}

void destroyInstance(vk::Instance &instance, vk::DebugUtilsMessengerEXT &debugUtilsMessenger)
{
#ifndef NDEBUG
    instance.destroyDebugUtilsMessengerEXT(debugUtilsMessenger);
#endif
    instance.destroy();
}
void destroyLogicalDevice(vk::Device &device)
{
    device.destroy();
}

void destroyCommandPool(vk::Device &device, vk::CommandPool &commandPool)
{
    device.destroyCommandPool(commandPool);
    commandPool = vk::CommandPool();
}

void showAvailableQueues(vk::PhysicalDevice &pDevice, bool diagExt)
{

    using Chain = vk::StructureChain<vk::QueueFamilyProperties2, vk::QueueFamilyCheckpointPropertiesNV>;
    auto queueFamilyProperties2 = pDevice.getQueueFamilyProperties2<Chain, std::allocator<Chain>, vk::DispatchLoaderDynamic>();

    for (size_t j = 0; j < queueFamilyProperties2.size(); j++)
    {
        std::cout << "\t"
                  << "QueueFamily " << j << "\n";
        vk::QueueFamilyProperties const &properties =
            queueFamilyProperties2[j].get<vk::QueueFamilyProperties2>().queueFamilyProperties;
        std::cout << "\t\t"
                  << "QueueFamilyProperties:\n";
        std::cout << "\t\t\t"
                  << "queueFlags                  = " << vk::to_string(properties.queueFlags) << "\n";
        std::cout << "\t\t\t"
                  << "queueCount                  = " << properties.queueCount << "\n";
        std::cout << "\t\t\t"
                  << "timestampValidBits          = " << properties.timestampValidBits << "\n";
        std::cout << "\t\t\t"
                  << "minImageTransferGranularity = " << properties.minImageTransferGranularity.width << " x "
                  << properties.minImageTransferGranularity.height << " x "
                  << properties.minImageTransferGranularity.depth << "\n";
        std::cout << "\n";

        if (diagExt)
        {
            vk::QueueFamilyCheckpointPropertiesNV const &checkpointProperties =
                queueFamilyProperties2[j].get<vk::QueueFamilyCheckpointPropertiesNV>();
            std::cout << "\t\t"
                      << "CheckPointPropertiesNV:\n";
            std::cout << "\t\t\t"
                      << "checkpointExecutionStageMask  = "
                      << vk::to_string(checkpointProperties.checkpointExecutionStageMask) << "\n";
            std::cout << "\n";
        }
    }
}

void createTimestampQueryPool(vk::Device &device, vk::QueryPool &queryPool, uint32_t queryCount)
{
    vk::QueryPoolCreateInfo createInfo({}, vk::QueryType::eTimestamp, queryCount);
    queryPool = device.createQueryPool(createInfo);
}

void destroyQueryPool(vk::Device &device, vk::QueryPool &queryPool)
{
    device.destroyQueryPool(queryPool);
    queryPool = vk::QueryPool();
}

void printDeviceCapabilities(vk::PhysicalDevice &pDevice)
{
    //vk::PhysicalDeviceFeatures features = physicalDevice.getFeatures();
    std::vector<vk::ExtensionProperties> ext = pDevice.enumerateDeviceExtensionProperties();
    std::vector<vk::LayerProperties> layers = pDevice.enumerateDeviceLayerProperties();
    vk::PhysicalDeviceMemoryProperties memoryProperties = pDevice.getMemoryProperties();
    vk::PhysicalDeviceProperties properties = pDevice.getProperties();
    vk::PhysicalDeviceType dt = properties.deviceType;

    std::cout << "====================" << std::endl
              << "Device Name: " << properties.deviceName << std::endl
              << "Device ID: " << properties.deviceID << std::endl
              << "Device Type: " << vk::to_string(properties.deviceType) << std::endl
              << "Driver Version: " << properties.driverVersion << std::endl
              << "API Version: " << properties.apiVersion << std::endl
              << "====================" << std::endl
              << std::endl;

    bool budgetExt = false;
    bool diagExt = false;
    std::cout << "This device supports the following extensions (" << ext.size() << "): " << std::endl;
    for (vk::ExtensionProperties e : ext)
    {
        std::cout << std::string(e.extensionName.data()) << std::endl;
        if (std::string(e.extensionName.data()) == VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)
            budgetExt = true;
        if (std::string(e.extensionName.data()) == VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME)
            diagExt = true;
    }

    std::cout << "This device supports the following memory types (" << memoryProperties.memoryTypeCount << "): " << std::endl;
    uint32_t c = 0U;
    for (vk::MemoryType e : memoryProperties.memoryTypes)
    {
        if (c > memoryProperties.memoryTypeCount)
            break;

        std::cout << e.heapIndex << "\t ";
        std::cout << vk::to_string(e.propertyFlags) << std::endl;
        c++;
    }
    std::cout << "====================" << std::endl
              << std::endl;

    if (budgetExt)
    {
        std::cout << "This device has the following heaps (" << memoryProperties.memoryHeapCount << "): " << std::endl;
        c = 0U;
        for (vk::MemoryHeap e : memoryProperties.memoryHeaps)
        {
            if (c > memoryProperties.memoryHeapCount)
                break;

            std::cout << "Size: " << formatSize(e.size) << "\t ";
            std::cout << vk::to_string(e.flags) << std::endl;
            c++;
        }
    }

    std::cout << "====================" << std::endl
              << std::endl
              << "This device has the following layers (" << layers.size() << "): " << std::endl;
    for (vk::LayerProperties l : layers)
        std::cout << std::string(l.layerName.data()) << "\t : " << std::string(l.description.data()) << std::endl;
    std::cout << "====================" << std::endl
              << std::endl;

    showAvailableQueues(pDevice, diagExt);
}
