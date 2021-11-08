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

struct A2Task2SolutioNaive : A2Task2Solution {
public:
    A2Task2SolutioNaive(AppResources &app, uint workGroupSize);

    void prepare(const std::vector<uint> &input) override;
    void compute() override;
    std::vector<uint> result() const override;
    void cleanup() override;

private:
    struct PushStruct
    {
        uint size;
        uint offset;
    };

    AppResources &app;
    uint workGroupSize;

    uint workSize;

    Buffer buffers[2];

    // Descriptor & Pipeline Layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;

    vk::ShaderModule cShader;
    vk::Pipeline pipeline;

    // Descriptor Pool
    vk::DescriptorPool descriptorPool;

    // Descriptors
    vk::DescriptorSet descriptorSets[2];

    uint activeBuffer = 0;
};
  