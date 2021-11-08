#ifndef UTILS
#define UTILS
#include <vector>
#include <cstring>

#include <vulkan/vulkan.hpp>

#define CAST(a) static_cast<uint32_t>(a.size())
struct Buffer
{
    vk::Buffer buf;
    vk::DeviceMemory mem;
};

typedef uint32_t uint;

template<typename T, typename V>
T ceilDiv(T x, V y) {
    return x / y + (x % y != 0);
}

std::vector<char> readFile(const std::string &filename);
std::string formatSize(uint64_t size);
uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, vk::PhysicalDevice &pdevice);
void createBuffer(vk::PhysicalDevice &pDevice, vk::Device &device,
                  const vk::DeviceSize &size, vk::BufferUsageFlags usage,
                  vk::MemoryPropertyFlags properties, std::string name, vk::Buffer &buffer, vk::DeviceMemory &bufferMemory);
void createBuffer(vk::PhysicalDevice &pDevice, vk::Device &device,
                  const vk::DeviceSize &size, vk::BufferUsageFlags usage,
                  vk::MemoryPropertyFlags properties, std::string name, Buffer &buffer);
void destroyBuffer(vk::Device &device, Buffer &buffer);
void copyBuffer(vk::Device &device, vk::Queue &q, vk::CommandPool &commandPool,
                const vk::Buffer &srcBuffer, vk::Buffer &dstBuffer, vk::DeviceSize byteSize);

vk::CommandBuffer beginSingleTimeCommands(vk::Device &device, vk::CommandPool &commandPool);
void endSingleTimeCommands(vk::Device &device, vk::Queue &q,
                           vk::CommandPool &commandPool, vk::CommandBuffer &commandBuffer);

Buffer addHostCoherentBuffer(vk::PhysicalDevice &pDevice, vk::Device &device, vk::DeviceSize size, std::string name);
Buffer addDeviceOnlyBuffer(vk::PhysicalDevice &pDevice, vk::Device &device, vk::DeviceSize size, std::string name);

template <typename T>
void fillDeviceBuffer(vk::Device &device, vk::DeviceMemory &mem, const std::vector<T> &input)
{
    void *data = device.mapMemory(mem, 0, input.size() * sizeof(T), vk::MemoryMapFlags());
    memcpy(data, input.data(), static_cast<size_t>(input.size() * sizeof(T)));
    device.unmapMemory(mem);
}

template <typename T>
void fillHostBuffer(vk::Device &device, vk::DeviceMemory &mem, std::vector<T> &output)
{
    // copy memory from mem to output
    void *data = device.mapMemory(mem, 0, output.size() * sizeof(T), vk::MemoryMapFlags());
    memcpy(output.data(), data, static_cast<size_t>(output.size() * sizeof(T)));
    device.unmapMemory(mem);
}

template <typename T>
void fillDeviceWithStagingBuffer(vk::PhysicalDevice &pDevice, vk::Device &device,
                           vk::CommandPool &commandPool, vk::Queue &q,
                           Buffer &b, const std::vector<T> &data)
{
    // Buffer b requires the eTransferSrc bit
    // data (host) -> staging (device) -> Buffer b (device)
    vk::Buffer staging;
    vk::DeviceMemory mem;
    vk::DeviceSize byteSize = data.size() * sizeof(T);

    createBuffer(pDevice, device, byteSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible, "staging",
                 staging, mem);
    // V host -> staging V
    fillDeviceBuffer<T>(device, mem, data);
    // V staging -> buffer V
    copyBuffer(device, q, commandPool, staging, b.buf, byteSize);
    device.destroyBuffer(staging);
    device.freeMemory(mem);
}

template <typename T>
void fillHostWithStagingBuffer(vk::PhysicalDevice &pDevice, vk::Device &device,
                           vk::CommandPool &commandPool, vk::Queue &q,
                           const Buffer &b, std::vector<T> &data)
{
    // Buffer b requires the eTransferDst bit
    // Buffer b (device) -> staging (device) -> data (host)
    vk::Buffer staging;
    vk::DeviceMemory mem;
    vk::DeviceSize byteSize = data.size() * sizeof(T);

    createBuffer(pDevice, device, byteSize, vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible, "staging",
                 staging, mem);
    // V buffer -> staging V
    copyBuffer(device, q, commandPool, b.buf, staging, byteSize);
    // V staging -> host V
    fillHostBuffer<T>(device, mem, data);
    
    device.destroyBuffer(staging);
    device.freeMemory(mem);
}

template <typename T>
void setObjectName(vk::Device &device, T handle, std::string name)
{
#ifndef NDEBUG
    vk::DebugUtilsObjectNameInfoEXT infoEXT(handle.objectType, uint64_t(static_cast<typename T::CType>(handle)), name.c_str());
    device.setDebugUtilsObjectNameEXT(infoEXT);
#endif
}

#endif