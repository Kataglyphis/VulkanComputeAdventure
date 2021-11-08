#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include <algorithm>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"
#include "A1task1.h"

void defaultVectors(std::vector<int> &in1, std::vector<int> &in2, size_t size)
{
    // === prepare data ===
    in1 = std::vector<int>(size, 0u);
    for (size_t i = 0; i < in1.size(); i++)
        in1[i] = static_cast<int>(i);
    in2 = std::vector<int>(in1);
    std::reverse(in2.begin(), in2.end());
}
/* requires to have called prepare() because we need the buffers to be correctly created*/
void A1_Task1::defaultValues()
{
    std::vector<int> inputVec, inputVec2;
    defaultVectors(inputVec, inputVec2, this->workloadSize);
    // === fill buffers ===
    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inBuffer1, inputVec);
    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inBuffer2, inputVec2);
}

void A1_Task1::checkDefaultValues()
{
    // ### gather the output data after having called compute() ###
    std::vector<unsigned int> result(this->workloadSize, 1u);

    fillHostWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, outBuffer, result);

    std::vector<int> inputVec, inputVec2;
    defaultVectors(inputVec, inputVec2, this->workloadSize);
    std::vector<int> outputVec(this->workloadSize, 0u);
    std::transform(inputVec.begin(), inputVec.end(), inputVec2.begin(), outputVec.begin(), std::plus<int>());

    if (std::equal(result.begin(), result.end(), outputVec.begin()))
        std::cout << "All is good it seems" << std::endl;
    else
        std::cout << " Oh no! We found errors!" << std::endl;
}

void A1_Task1::prepare(unsigned int size)
{
    this->workloadSize = size;
    
    // ### Fill the descriptorLayoutBindings  ###
    Cmn::addStorage(task.bindings, 0); //                           |||
    Cmn::addStorage(task.bindings, 1); //                           |||
    Cmn::addStorage(task.bindings, 2); //                           |||
    // ### Push Constant ###
    vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushStruct));
    // ### Create Pipeline Layout ###
    Cmn::createDescriptorSetLayout(app.device, task.bindings, task.descriptorSetLayout);
    Cmn::createDescriptorPool(app.device, task.bindings, task.descriptorPool);
    Cmn::allocateDescriptorSet(app.device, task.descriptorSet, task.descriptorPool, task.descriptorSetLayout);
    vk::PipelineLayoutCreateInfo pipInfo(vk::PipelineLayoutCreateFlags(), 1U, &task.descriptorSetLayout, 1U, &pcr);
    task.pipelineLayout = app.device.createPipelineLayout(pipInfo);
    // ### create buffers ###
    createBuffer(app.pDevice, app.device, workloadSize * sizeof(int), vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        "inBuffer1", inBuffer1.buf, inBuffer1.mem);

    createBuffer(app.pDevice, app.device, workloadSize * sizeof(int), vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        "inBuffer2", inBuffer2.buf, inBuffer2.mem);

    createBuffer(app.pDevice, app.device, workloadSize * sizeof(int), vk::BufferUsageFlagBits::eTransferSrc |
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        "outBuffer", outBuffer.buf, outBuffer.mem);

    // ### Fills inBuffer1 and inBuffer2 ###

    this->defaultValues();

    // === prepare data ===
    std::vector<int> inputVec(workloadSize, 0u);
    for (size_t i = 0; i < inputVec.size(); i++)
        inputVec[i] = static_cast<int>(i);
    std::vector<int> inputVec2 = inputVec;
    std::reverse(inputVec2.begin(), inputVec2.end());
    // inputVec and inputVec2 are correctly filled

    // ### Create  structures ###
    // ### DescriptorSet is created but not filled yet ###
    // ### Bind buffers to descriptor set ### (calls update several times)

    Cmn::bindBuffers(app.device, inBuffer1.buf, task.descriptorSet, 0);
    Cmn::bindBuffers(app.device, inBuffer2.buf, task.descriptorSet, 1);
    Cmn::bindBuffers(app.device, outBuffer.buf, task.descriptorSet, 2);
    // ### Preparation work done! ###
}

void A1_Task1::compute(uint32_t dx, uint32_t dy, uint32_t dz, std::string file)
{
    uint32_t groupCount = (workloadSize + dx - 1) / dx; 
    PushStruct push{ workloadSize }; // todo: fill
    // ### Create ShaderModule ###
 
    std::string compute = "shaders/" + file + ".comp.spv";
    vkDestroyShaderModule(app.device, task.cShader, nullptr);
    Cmn::createShader(app.device, task.cShader, compute);
    // ### Specialization constants
    // constantID, offset, sizeof(type)

    // ### Create Pipeline ###
    std::array<vk::SpecializationMapEntry, 1> spec_entries =
        std::array<vk::SpecializationMapEntry, 1>{ { {0U, 0U, sizeof(int)}}};

    std::array<int, 1> spec_values = { int(dx) };
    vk::SpecializationInfo spec_info = vk::SpecializationInfo(
        CAST(spec_entries), spec_entries.data(),
        CAST(spec_values) * sizeof(int), spec_values.data());

    app.device.destroyPipeline(task.pipeline);
    Cmn::createPipeline(app.device, task.pipeline, task.pipelineLayout, spec_info, task.cShader);

    // ### finally do the compute ###
    this->dispatchWork(groupCount, 1U, 1U, push);
}

void A1_Task1::dispatchWork(uint32_t dx, uint32_t dy, uint32_t dz, PushStruct &pushConstant)
{
    /* ### Create Command Buffer ### */
    vk::CommandBufferAllocateInfo alloc_info(app.computeCommandPool,
        vk::CommandBufferLevel::ePrimary, 1U);

    vk::CommandBuffer cb = app.device.allocateCommandBuffers(alloc_info)[0];
    /* ### Call Begin and register commands ### */
    vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cb.begin(begin_info);
    cb.resetQueryPool(app.queryPool, 0, 2);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 0);
    cb.pushConstants(task.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushStruct), &pushConstant);
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, task.pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, task.pipelineLayout, 0U, 1U, &task.descriptorSet, 0U, nullptr);
    cb.dispatch(dx, dy, dz);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 1);
    /* ### End of Command Buffer, enqueue it and use a Fence ### */
    cb.end();
    /* ### Collect data from the query Pool ### */
    vk::SubmitInfo submit_info = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb);
    vk::Fence fence = app.device.createFence(vk::FenceCreateInfo());
    app.computeQueue.submit({ submit_info }, fence);
    vk::Result haveIWaited = app.device.waitForFences({ fence }, true, uint64_t(-1));
    app.device.destroyFence(fence);
    /* Uncomment this once you've finished this function:
    ###*/
    uint64_t timestamps[2];
    vk::Result result = app.device.getQueryPoolResults(app.queryPool, 0, 2, sizeof(timestamps), &timestamps, sizeof(timestamps[0]), vk::QueryResultFlagBits::e64);
    assert(result == vk::Result::eSuccess);
    uint64_t timediff = timestamps[1] - timestamps[0];
    vk::PhysicalDeviceProperties properties = app.pDevice.getProperties();
    uint64_t nanoseconds = properties.limits.timestampPeriod * timediff;

    mstime = nanoseconds / 1000000.f;
    
    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);
}