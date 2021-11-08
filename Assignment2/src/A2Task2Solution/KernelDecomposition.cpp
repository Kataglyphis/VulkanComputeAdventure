#include "KernelDecomposition.h"

#include "host_timer.h"

A2Task2SolutionKernelDecomposition::A2Task2SolutionKernelDecomposition(
    AppResources &app, uint workGroupSize):
    app(app), workGroupSize(workGroupSize) {}

void A2Task2SolutionKernelDecomposition::prepare(const std::vector<uint> &input) {
    workSize = input.size();

    // Descriptor & Pipeline Layout
    Cmn::addStorage(bindings, 0);
    Cmn::addStorage(bindings, 1);
    Cmn::createDescriptorSetLayout(app.device, bindings, descriptorSetLayout);
    vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushStruct));
    vk::PipelineLayoutCreateInfo pipInfo(vk::PipelineLayoutCreateFlags(), 1U, &descriptorSetLayout, 1U, &pcr);
    pipelineLayout = app.device.createPipelineLayout(pipInfo);

    // Specialization constant for workgroup size
    std::array<vk::SpecializationMapEntry, 2> specEntries = std::array<vk::SpecializationMapEntry, 2>{ 
        {{0U, 0U, sizeof(workGroupSize)}},
    }; 
    std::array<uint32_t, 2> specValues = {workGroupSize}; //for workgroup sizes
    vk::SpecializationInfo specInfo = vk::SpecializationInfo(CAST(specEntries), specEntries.data(),
                                    CAST(specValues) * sizeof(int), specValues.data());

    // Local PPS Pipeline
    Cmn::createShader(app.device, cShaderLocalPPS, "./shaders/A2Task2KernelDecomposition.comp.spv");
    Cmn::createPipeline(app.device, pipelineLocalPPS, pipelineLayout, specInfo, cShaderLocalPPS);
    
    // Local PPS Offset Pipeline
    Cmn::createShader(app.device, cShaderLocalPPSOffset, "./shaders/A2Task2KernelDecompositionOffset.comp.spv");
    Cmn::createPipeline(app.device, pipelineLocalPPSOffset, pipelineLayout, specInfo, cShaderLocalPPSOffset);

    // ### create buffers, get their index in the task.buffers[] array ###
    using BFlag = vk::BufferUsageFlagBits;
    auto makeDLocalBuffer = [ this ](vk::BufferUsageFlags usage, vk::DeviceSize size, std::string name) -> Buffer
    {
        Buffer b;
        createBuffer(app.pDevice, app.device, size, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, name, b.buf, b.mem);
        return b;
    };

    inoutBuffers.push_back(makeDLocalBuffer(BFlag::eTransferDst | BFlag::eTransferSrc | BFlag::eStorageBuffer, input.size() * sizeof(uint32_t), "buffer_inout_0"));

    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inoutBuffers[0], input);

    // TO DO create additional buffers (by pushing into inoutBuffers) and descriptors (by pushing into descriptorSets)
    // You need to create an appropriately-sized DescriptorPool first
    Cmn::createDescriptorPool(app.device, bindings, descriptorPool, 2);
    for (int i = 0; i < 1; i++)
        Cmn::allocateDescriptorSet(app.device, descriptorSets[i], descriptorPool, descriptorSetLayout);
    Cmn::bindBuffers(app.device, inoutBuffers[0].buf, descriptorSets[0], 0);

}

void A2Task2SolutionKernelDecomposition::compute() {
    vk::CommandBufferAllocateInfo allocInfo(
        app.computeCommandPool, vk::CommandBufferLevel::ePrimary, 1U);
    vk::CommandBuffer cb = app.device.allocateCommandBuffers( allocInfo )[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cb.begin(beginInfo);
    cb.resetQueryPool(app.queryPool, 0, 2);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 0);
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipelineLocalPPS);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 1);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0U, 1U, &descriptorSets[0], 0U, nullptr);


    unsigned int curr_vec_size = workSize;
    // how often do we have to perform the group sums
    // unsigned int reduction_depth = std::ceil(log(curr_vec_size) / log(workGroupSize));

    uint resulting_kernel_size = std::min(workGroupSize, curr_vec_size);
    unsigned int current_wrk_grp_count = (curr_vec_size + resulting_kernel_size - 1) / (resulting_kernel_size);
    PushStruct push{ curr_vec_size , resulting_kernel_size };
    cb.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushStruct), &push);
    cb.dispatch(current_wrk_grp_count, 1, 1);

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags(),
        { vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead) },
        {},
        {}
    );

    curr_vec_size = current_wrk_grp_count / 2;

    cb.end();

    vk::SubmitInfo submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb);

    HostTimer timer;

    app.computeQueue.submit({submitInfo});
    app.device.waitIdle();

    mstime = timer.elapsed() * 1000;

    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);
}

std::vector<uint> A2Task2SolutionKernelDecomposition::result() const {
    std::vector<uint> result(workSize, 0);
    fillHostWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, inoutBuffers[0], result);
    return result;
}


void A2Task2SolutionKernelDecomposition::cleanup() {
    app.device.destroyDescriptorPool(descriptorPool);

    app.device.destroyPipeline(pipelineLocalPPSOffset);
    app.device.destroyShaderModule(cShaderLocalPPSOffset);

    app.device.destroyPipeline(pipelineLocalPPS);
    app.device.destroyShaderModule(cShaderLocalPPS);

    app.device.destroyPipelineLayout(pipelineLayout);
    app.device.destroyDescriptorSetLayout(descriptorSetLayout);
    bindings.clear();

    auto Bclean = [&](Buffer &b){
        app.device.destroyBuffer(b.buf);
        app.device.freeMemory(b.mem);};

    for (auto inoutBuffer : inoutBuffers) {
        Bclean(inoutBuffer);
    }

    inoutBuffers.clear();
}