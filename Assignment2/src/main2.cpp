#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "A2task1.h"
#include "A2task2.h"
#include "A2Task1Solution/Sequential.h"
#include "A2Task1Solution/Interleaved.h"
#include "A2Task1Solution/KernelDecomposition.h"
#include "A2Task2Solution/Naive.h"
#include "A2Task2Solution/KernelDecomposition.h"
#include "renderdoc.h"
#include <CSVWriter.h>

int main()
{
    try
    {
        AppResources app;
        

        initApp(app);

        renderdoc::initialize();
        renderdoc::startCapture();
        
        // You can freely modify this function for debugging & evaluation purposes.
        // For example, you could add some more loops to evaluate your code at different work group sizes and output a csv with the timings.

        {
            size_t size = 128*1024*1024;
            A2Task1 a2Task1(size);

            auto evaluateTask1Solution = [&](A2Task1Solution* solution, std::string name) {
                std::cout << "[Task1] evaluating " << name << std::endl;
                
                uint N = 10;

                bool pass = true;
                float mstime = 0.f;
                for (int i = 0; i < N; i++) {
                    pass &= a2Task1.evaluateSolution(*solution);
                    solution->cleanup();
                    mstime += solution->mstime / N;

                    if (!pass) break;
                }

                if (pass) {
                    std::cout << "Execution time: " << mstime<< " ms, "
                              << "Throughput: " << size / mstime / 1000000 << " GE/s" << std::endl;
                    std::cout << "TEST PASSED" << std::endl;
                } else {
                    std::cout << "TEST FAILED" << std::endl;
                }

            };

            /*A2Task1SolutionInterleaved interleavedSolution(app, 128);
            evaluateTask1Solution(&interleavedSolution, "Interleaved");

            A2Task1SolutionSequential sequentialSolution(app, 128);
            evaluateTask1Solution(&sequentialSolution, "Sequential");

            A2Task1SolutionKernelDecomposition kernelDecompositionSolution(app, 128, "shaders/A2Task1KernelDecomposition.comp.spv");
            evaluateTask1Solution(&kernelDecompositionSolution, "KernelDecomposition");

            A2Task1SolutionKernelDecomposition kernelDecompositionUnrollSolution(app, 128, "shaders/A2Task1KernelDecompositionUnroll.comp.spv");
            evaluateTask1Solution(&kernelDecompositionUnrollSolution, "KernelDecomposition Unroll");

            A2Task1SolutionKernelDecomposition kernelDecompositionAtomicSolution(app, 128, "shaders/A2Task1KernelDecompositionAtomic.comp.spv");
            evaluateTask1Solution(&kernelDecompositionAtomicSolution, "KernelDecomposition Atomic");*/
        }

        {
            size_t size = 128*1024*1024;
           
            // This is used for testing local kernel decomposition without extension to arbitrary arrays.
            // Must be power of two and <= 1024!
            size_t sizeLocal = 128;

            A2Task2 a2Task2(size);
            A2Task2 a2Task2Local(sizeLocal);

            auto evaluateTask2Solution = [&](A2Task2 *task, A2Task2Solution* solution, std::string name) {
                std::cout << "[Task2] evaluating " << name << std::endl;
                
                uint N = 10;

                bool pass = true;
                float mstime = 0.f;
                for (int i = 0; i < N; i++) {
                    pass &= task->evaluateSolution(*solution);
                    solution->cleanup();
                    mstime += solution->mstime / N;

                    if (!pass) break;
                }
                
                if (pass) {
                    std::cout << "Execution time: " << mstime<< " ms, "
                              << "Throughput: " << task->size() / mstime / 1000000 << " GE/s" << std::endl;
                    std::cout << "TEST PASSED" << std::endl;
                } else {
                    std::cout << "TEST FAILED" << std::endl;
                }
            };

            /*A2Task2SolutioNaive naiveSolution(app, 128);
            evaluateTask2Solution(&a2Task2, &naiveSolution, "Naive");*/

            A2Task2SolutionKernelDecomposition kernelDecompositionSolutionLocal(app, sizeLocal);
            evaluateTask2Solution(&a2Task2Local, &kernelDecompositionSolutionLocal, "Kernel Decomposition Local");

            /*A2Task2SolutionKernelDecomposition kernelDecompositionSolution(app, 128);
            evaluateTask2Solution(&a2Task2, &kernelDecompositionSolution, "Kernel Decomposition");*/
        }

        renderdoc::endCapture();

        app.destroy();
    }
    catch (vk::SystemError &err)
    {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        exit(-1);
    }
    catch (std::exception &err)
    {
        std::cout << "std::exception: " << err.what() << std::endl;
        exit(-1);
    }
    catch (...)
    {
        std::cout << "unknown error\n";
        exit(-1);
    }
    return EXIT_SUCCESS;
}