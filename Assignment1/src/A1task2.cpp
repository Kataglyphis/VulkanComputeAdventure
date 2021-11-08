#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"
#include "A1task2.h"

/* requires to have called prepare() because we need the buffers to be correctly created*/
std::vector<int> A1_Task2::incArray()
{
    // === prepare data ===
    std::vector<int> inputVec(workloadSize, 0u);
    for (size_t i = 0; i < inputVec.size(); i++)
    {
        inputVec[i] = i;
    }
    return inputVec;
}
void A1_Task2::defaultValues()
{
    std::vector<int> inputVec = incArray();
    // === fill buffer ===  
    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool,
        app.transferQueue, inBuffer, inputVec);

}

void A1_Task2::checkDefaultValues()
{
    // ### gather the output data after having called compute() ###
    std::vector<int> result(workloadSize, 1u);

    fillHostWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, outBuffer, result);
    std::vector<int> input = incArray();

    std::vector<int> rotate = rotateCPU(input, workloadSize_w, workloadSize_h);
    int errors = 0;
    for(int i = 0 ; i < rotate.size(); i++)
        if(rotate[i] != result[i])
            errors++;
    if(errors>0)
        std::cout<<std::endl<<"=== There were " << errors << " error(s). ===" << std::endl;

/*
    std::cout << std::endl
              << std::endl
              << "========================" << std::endl
              << std::endl;
    // ### CHECK RESULT IS VALID ###
    for (unsigned int i = 0; i < workloadSize; i++)
    {
        if (i % workloadSize_h == 0 && i != 0)
            std::cout << std::endl;
        std::cout << result[i] << " ";
    }*/
}

void A1_Task2::prepare(unsigned int size_w, unsigned int size_h)
{
    this->workloadSize_w = size_w;
    this->workloadSize_h = size_h;
    this->workloadSize = size_h * size_w;

    // ||| this fills the descriptorLayoutBindings  |||
    Cmn::addStorage(task.bindings, 0);
    Cmn::addStorage(task.bindings, 1);
    

    Cmn::createDescriptorSetLayout(app.device, task.bindings, task.descriptorSetLayout);
    Cmn::createDescriptorPool(app.device, task.bindings, task.descriptorPool);
    Cmn::allocateDescriptorSet(app.device, task.descriptorSet, task.descriptorPool, task.descriptorSetLayout);
    // ### DescriptorSet is created but not filled yet ###

    // ### create buffers, get their index in the task.buffers[] array ###
    using BFlag = vk::BufferUsageFlagBits;
    auto makeDLocalBuffer = [this](vk::BufferUsageFlags usage, vk::DeviceSize size, std::string name) -> Buffer
    {// this lambda just fills the createBuffer with a more human readable set of parameters
        Buffer b;
        createBuffer(app.pDevice, app.device, size, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, name, b.buf, b.mem);
        return b;
    };

    this->inBuffer = makeDLocalBuffer(BFlag::eTransferDst | BFlag::eStorageBuffer, workloadSize * sizeof(unsigned int), "buffer_in");
    this->outBuffer = makeDLocalBuffer(BFlag::eTransferSrc | BFlag::eStorageBuffer, workloadSize * sizeof(unsigned int), "buffer_out");
    
    // ### fill buffer with default values
    this->defaultValues();

    // ### Bind buffers to descriptor set ### (calls update several times)
    Cmn::bindBuffers(app.device, inBuffer.buf, task.descriptorSet, 0);
    Cmn::bindBuffers(app.device, outBuffer.buf, task.descriptorSet, 1);

    // ### Create Pipeline Layout ###
    vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushStruct));
    vk::PipelineLayoutCreateInfo pipInfo(vk::PipelineLayoutCreateFlags(), 1U, &task.descriptorSetLayout, 1U, &pcr);
    task.pipelineLayout = app.device.createPipelineLayout(pipInfo);
}

void A1_Task2::compute(uint32_t dx, uint32_t dy, uint32_t dz, std::string file)
{
    int numGroupsX, numGroupsY;
    numGroupsX = (workloadSize_w + dx - 1) / dx;
    numGroupsY = (workloadSize_h + dy - 1) / dy;
    PushStruct push{ workloadSize_w, workloadSize_h };
    
    // ### Create ShaderModule ###
    // Same as in task1
    std::string compute = "shaders/" + file + ".comp.spv";
    vkDestroyShaderModule(app.device, task.cShader, nullptr);
    Cmn::createShader(app.device, task.cShader, compute);

    // ### Create Pipeline ###
    // create the specialization entries for a 2D array
    std::array<vk::SpecializationMapEntry, 2> spec_entires =
        std::array<vk::SpecializationMapEntry, 2> {
        vk::SpecializationMapEntry{ 0U, 0U, sizeof(int) },
            vk::SpecializationMapEntry{ 1U, sizeof(int), sizeof(int) }};

    std::array<int, 2> spec_values = { int(dx) , int(dy) };

    vk::SpecializationInfo spec_info = vk::SpecializationInfo(
        CAST(spec_entires), spec_entires.data(),
        CAST(spec_values) * sizeof(int), spec_values.data());
    
    // in case a pipeline was already created, destroy it
    app.device.destroyPipeline(task.pipeline);
    Cmn::createPipeline(app.device, task.pipeline, task.pipelineLayout, spec_info, task.cShader);

    // ### finally do the compute ###
    this->dispatchWork(numGroupsX, numGroupsY, 1U, push);

    // ### gather the output data ###
    std::vector<unsigned int> result(workloadSize, 1u);

    fillHostWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, outBuffer, result);

    // ### END ###
}

void A1_Task2::dispatchWork(uint32_t dx, uint32_t dy, uint32_t dz, PushStruct &pushConstant)
{
    /* ### You can copy/paste the function of A1_Task1 here ### */
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

    vk::SubmitInfo submit_info = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb);
    vk::Fence fence = app.device.createFence(vk::FenceCreateInfo());
    app.computeQueue.submit({ submit_info }, fence);
    vk::Result haveIWaited = app.device.waitForFences({ fence }, true, uint64_t(-1));
    app.device.destroyFence(fence);

    uint64_t timestamps[2];
    vk::Result result = app.device.getQueryPoolResults(app.queryPool, 0, 2, sizeof(timestamps), &timestamps, sizeof(timestamps[0]), vk::QueryResultFlagBits::e64);
    assert(result == vk::Result::eSuccess);
    uint64_t timediff = timestamps[1] - timestamps[0];
    vk::PhysicalDeviceProperties properties = app.pDevice.getProperties();
    uint64_t nanoseconds = properties.limits.timestampPeriod * timediff;

    mstime = nanoseconds / 1000000.f;
    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);
}