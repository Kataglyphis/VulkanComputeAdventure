#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include "utils.h"

std::vector<char> readFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        std::string error = "failed to open file: " + filename;
        throw std::runtime_error(error);
    }
    size_t fileSize = (size_t)file.tellg();

    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    std::cout << "read " << buffer.size() << " bytes of data in file " << filename << std::endl;
    return buffer;
}

std::string formatSize(uint64_t size)
{
    std::ostringstream oss;
    if (size < 1024)
    {
        oss << size << " B";
    }
    else if (size < 1024 * 1024)
    {
        oss << size / 1024.f << " KB";
    }
    else if (size < 1024 * 1024 * 1024)
    {
        oss << size / (1024.0f * 1024.0f) << " MB";
    }
    else
    {
        oss << size / (1024.0f * 1024.0f * 1024.0f) << " GB";
    }
    return oss.str();
}

uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, vk::PhysicalDevice &pdevice)
{
    vk::PhysicalDeviceMemoryProperties memProperties = pdevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ( (typeFilter & (1 << i)) && 
             (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
             
            return i;
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void createBuffer(vk::PhysicalDevice &pDevice, vk::Device &device,
                  const vk::DeviceSize &size, vk::BufferUsageFlags usage,
                  vk::MemoryPropertyFlags properties, std::string name, vk::Buffer &buffer, vk::DeviceMemory &bufferMemory)
{
    vk::BufferCreateInfo in_buffer_info({}, size, usage);
    buffer = device.createBuffer(in_buffer_info);

    vk::MemoryRequirements mem_req = device.getBufferMemoryRequirements(buffer);
    vk::MemoryAllocateInfo alloc_info(mem_req.size, findMemoryType(mem_req.memoryTypeBits,
        properties, pDevice));
    bufferMemory = device.allocateMemory(alloc_info);
    device.bindBufferMemory(buffer, bufferMemory, 0U);
}

void copyBuffer(vk::Device &device, vk::Queue &q, vk::CommandPool &commandPool,
                const vk::Buffer &srcBuffer, vk::Buffer &dstBuffer, vk::DeviceSize byteSize)
{
    vk::CommandBuffer command_buffer = beginSingleTimeCommands(device, commandPool);
    vk::BufferCopy copy_region(0ULL, 0ULL, byteSize);
    command_buffer.copyBuffer(srcBuffer, dstBuffer, 1, &copy_region);
    endSingleTimeCommands(device, q, commandPool, command_buffer);
}

vk::CommandBuffer beginSingleTimeCommands(vk::Device &device, vk::CommandPool &commandPool)
{
    vk::CommandBufferAllocateInfo allocInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1);

    vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    commandBuffer.begin(beginInfo);

    return commandBuffer;
}

void endSingleTimeCommands(vk::Device &device, vk::Queue &q,
                           vk::CommandPool &commandPool, vk::CommandBuffer &commandBuffer)
{
    commandBuffer.end();
    vk::SubmitInfo submitInfo(0U, nullptr, nullptr, 1U, &commandBuffer);
    q.submit({submitInfo}, nullptr);
    q.waitIdle();
    device.freeCommandBuffers(commandPool, 1, &commandBuffer);
}
