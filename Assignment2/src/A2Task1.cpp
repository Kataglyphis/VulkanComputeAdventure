#include "A2Task1.h"

#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "exercise_template.h"
#include "host_timer.h"


A2Task1::A2Task1(uint problemSize) : input(problemSize, 0) {
    for (auto i = 0; i < problemSize; i++)
        input[i] = i % 97;
    computeReference();
}

A2Task1::A2Task1(std::vector<uint> input) : input(input) {
    computeReference();
}

bool A2Task1::evaluateSolution(A2Task1Solution& solution) {
    solution.prepare(input);
    solution.compute();
    auto result = solution.result();
    if (reference != result) {
        std::cout << "error: expected " << reference << ", but got " << result << std::endl;
        return false;
    }
    return true;
}

void A2Task1::computeReference() {
    reference = 0;
    for (auto e : input)
        reference += e;
}