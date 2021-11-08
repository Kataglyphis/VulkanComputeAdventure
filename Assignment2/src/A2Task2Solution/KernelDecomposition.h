#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "exercise_template.h"

#include "A2Task2.h"

struct A2Task2SolutionKernelDecomposition : A2Task2Solution {
public:
    A2Task2SolutionKernelDecomposition(AppResources &app, uint workGroupSize);

    void prepare(const std::vector<uint> &input) override;
    void compute() override;
    std::vector<uint> result() const override;
    void cleanup() override;

private:
    struct PushStruct
    {
        uint32_t size;
        uint32_t kernel_size;
    };

    AppResources &app;
    uint workGroupSize;
    std::string localPPSShaderFileName;

    uint workSize;

    std::vector<Buffer> inoutBuffers;

    // Descriptor & Pipeline Layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;

    // Local PPS Pipeline
    vk::ShaderModule cShaderLocalPPS;
    vk::Pipeline pipelineLocalPPS;

    // Local PPS Offset Pipeline
    vk::ShaderModule cShaderLocalPPSOffset;
    vk::Pipeline pipelineLocalPPSOffset;

    // Descriptor Pool
    vk::DescriptorPool descriptorPool;

    // TO DO extend with any additional members you may need
    vk::DescriptorSet descriptorSets[1];
};
  