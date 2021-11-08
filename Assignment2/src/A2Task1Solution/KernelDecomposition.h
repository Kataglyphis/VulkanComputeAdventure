#pragma once

#include "A2Task1.h"
#include <algorithm>
#include <math.h>

class A2Task1SolutionKernelDecomposition : public A2Task1Solution{
public:
    A2Task1SolutionKernelDecomposition(AppResources &app, uint workGroupSize, std::string shaderFileName);

    void prepare(const std::vector<uint> &input) override;
    void compute() override;
    uint result() const override;
    void cleanup() override;

private:
    struct PushConstant
    {
        uint size;
        uint kernel_size;

    };

    AppResources &app;
    uint workGroupSize;
    std::string shaderFileName;

    const std::vector<uint>* mpInput;

    Buffer buffers[2];

    // Descriptor & Pipeline Layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;

    // Local PPS Pipeline
    vk::ShaderModule shaderModule;
    vk::Pipeline pipeline;

    // Descriptor Pool
    vk::DescriptorPool descriptorPool;

    // Per-dispatch data
    vk::DescriptorSet descriptorSets[2];

    uint activeBuffer = 0;
};
