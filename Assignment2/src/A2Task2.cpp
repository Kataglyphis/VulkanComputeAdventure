#include "A2Task2.h"

A2Task2::A2Task2(uint problemSize) : input(problemSize, 0) {
    for (auto i = 0; i < problemSize; i++)
        input[i] = i % 97;
    computeReference();
}

A2Task2::A2Task2(std::vector<uint> input) : input(input) {
    computeReference();
}

void A2Task2::computeReference() {
    reference.reserve(input.size());
    uint acc = 0;
    for (auto i = 0; i < input.size(); i++) {
        acc += input[i];
        reference.push_back(acc);
    }
}

bool A2Task2::evaluateSolution(A2Task2Solution& solution) {
    solution.prepare(input);
    solution.compute();
    auto result = solution.result();

    if (result.size() != reference.size()) {
        std::cout << "error: result and reference vector size don't match!";
        return false;
    } 

    for (uint i = 0; i < reference.size(); i++) {
        if (result[i] != reference[i]) {
            std::cout << "error: result and reference don't match at index " << i << "!" << std::endl;
            std::cout << "\tresult:    " << result[i] << std::endl;
            std::cout << "\treference: " << reference[i] << std::endl;
            return false;
        }
    }

    return true;
}
