#include "KernelDecomposition.h"

#include "host_timer.h"

A2Task1SolutionKernelDecomposition::A2Task1SolutionKernelDecomposition(AppResources &app, uint workGroupSize, std::string shaderFileName) :
    app(app), workGroupSize(workGroupSize), shaderFileName(shaderFileName) {}

void A2Task1SolutionKernelDecomposition::prepare(const std::vector<uint> &input)
{
    mpInput = &input;

    Cmn::addStorage(bindings, 0);
    Cmn::addStorage(bindings, 1);
    Cmn::createDescriptorSetLayout(app.device, bindings, descriptorSetLayout);
    vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstant));
    vk::PipelineLayoutCreateInfo pipInfo(vk::PipelineLayoutCreateFlags(), 1U, &descriptorSetLayout, 1U, &pcr);
    pipelineLayout = app.device.createPipelineLayout(pipInfo);

    // Specialization constant for workgroup size
    std::array<vk::SpecializationMapEntry, 1> specEntries = std::array<vk::SpecializationMapEntry, 1>{ 
        {{0U, 0U, sizeof(workGroupSize)}},
    }; 
    std::array<uint32_t, 1> specValues = {workGroupSize}; //for workgroup sizes
    vk::SpecializationInfo specInfo = vk::SpecializationInfo(CAST(specEntries), specEntries.data(),
                                    CAST(specValues) * sizeof(int), specValues.data());

    Cmn::createShader(app.device, shaderModule, shaderFileName);
    Cmn::createPipeline(app.device, pipeline, pipelineLayout, specInfo, shaderModule);

    for (int i = 0; i < 2; i++) {
        createBuffer(app.pDevice, app.device, mpInput->size() * sizeof((*mpInput)[0]),
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal, "buffer_" + std::to_string(i), buffers[i].buf, buffers[i].mem);
    }

    fillDeviceWithStagingBuffer(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, buffers[0], input);
    
    Cmn::createDescriptorPool(app.device, bindings, descriptorPool, 2);
    for (int i = 0; i < 2; i++)
        Cmn::allocateDescriptorSet(app.device, descriptorSets[i], descriptorPool, descriptorSetLayout);
    Cmn::bindBuffers(app.device, buffers[0].buf, descriptorSets[0], 0);
    Cmn::bindBuffers(app.device, buffers[1].buf, descriptorSets[0], 1);
    Cmn::bindBuffers(app.device, buffers[1].buf, descriptorSets[1], 0);
    Cmn::bindBuffers(app.device, buffers[0].buf, descriptorSets[1], 1);
}

void A2Task1SolutionKernelDecomposition::compute()
{
    vk::CommandBufferAllocateInfo allocInfo(
        app.computeCommandPool, vk::CommandBufferLevel::ePrimary, 1U);
    vk::CommandBuffer cb = app.device.allocateCommandBuffers( allocInfo )[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cb.begin(beginInfo);

    cb.resetQueryPool(app.queryPool, 0, 2);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 0);
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cb.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, app.queryPool, 1);

    // we only need half the threads than we have entries in the vector in the beginning
    // for the first reduction

    unsigned int curr_vec_size = mpInput->size() / 2;
    unsigned int reduction_depth = std::ceil(log(curr_vec_size) / log(workGroupSize));

    for (int i = 0; i < reduction_depth; i++) {

        // ping pong the buffer
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0U, 1U, &descriptorSets[activeBuffer], 0U, nullptr);

        uint resulting_kernel_size = std::min(workGroupSize, curr_vec_size);
        unsigned int current_wrk_grp_count = (curr_vec_size + resulting_kernel_size - 1) / (resulting_kernel_size);

        PushConstant push{ curr_vec_size , resulting_kernel_size };
        cb.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstant), &push);

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

        // ping pong game :)
        activeBuffer++;
        activeBuffer = activeBuffer % 2;

    }

    cb.end();

    vk::SubmitInfo submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &cb);

    HostTimer timer;

    app.computeQueue.submit({submitInfo});
    app.device.waitIdle();

    mstime = timer.elapsed() * 1000;

    app.device.freeCommandBuffers(app.computeCommandPool, 1U, &cb);
}

uint A2Task1SolutionKernelDecomposition::result() const
{
    std::vector<uint> result(1, 0);
    fillHostWithStagingBuffer<uint>(app.pDevice, app.device, app.transferCommandPool, app.transferQueue, buffers[activeBuffer], result);
    return result[0];
}

void A2Task1SolutionKernelDecomposition::cleanup()
{
    app.device.destroyDescriptorPool(descriptorPool);

    app.device.destroyPipeline(pipeline);
    app.device.destroyShaderModule(shaderModule);

    app.device.destroyPipelineLayout(pipelineLayout);
    app.device.destroyDescriptorSetLayout(descriptorSetLayout);
    bindings.clear();

    for (int i = 0; i < 2; i++)
        destroyBuffer(app.device, buffers[i]);
}