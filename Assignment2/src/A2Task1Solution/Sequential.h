#pragma once

#include "A2Task1.h"

class A2Task1SolutionSequential : public A2Task1Solution{
public:
    A2Task1SolutionSequential(AppResources &app, uint workGroupSize);

    void prepare(const std::vector<uint> &input) override;
    void compute() override;
    uint result() const override;
    void cleanup() override;

private:
    struct PushConstant
    {
        uint size;
        uint offset;
    };

    AppResources &app;
    uint workGroupSize;

    const std::vector<uint>* mpInput;

    Buffer inoutBuffer;

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
    vk::DescriptorSet descriptorSet;
};
