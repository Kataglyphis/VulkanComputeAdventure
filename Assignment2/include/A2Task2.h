#pragma once

#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "exercise_template.h"

class A2Task2Solution {
public:
    float mstime;

    virtual void prepare(const std::vector<uint> &input) = 0;
    virtual void compute() = 0;
    virtual std::vector<uint> result() const = 0;
    virtual void cleanup() = 0;
};

class A2Task2 {
public:
    A2Task2(uint problemSize);
    A2Task2(std::vector<uint> input);

    bool evaluateSolution(A2Task2Solution& solution);
    size_t size() const {
        return input.size();
    }

private:
    void computeReference();

    std::vector<uint> input;
    std::vector<uint> reference;
};