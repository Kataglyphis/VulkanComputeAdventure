#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "A1task1.h"
#include "A1task2.h"
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

        CSVWriter csv;

        const unsigned int num_iterations_task1 = 50;

        csv.newRow() << "dx" << "dy" << "dz" << "vecsize" << "timing";

        std::array<unsigned int, 4> workload_sizes_w_task1 = { 2048, 1024, 512, 256 };
        std::array<unsigned int, 6> dx_s_task1 = { 1024, 512, 256, 128, 64, 32};

        A1_Task1 task(app);
        task.prepare(workload_sizes_w_task1[0] * workload_sizes_w_task1[0] + 31);
        for (size_t workload_size_w_index = 0; workload_size_w_index < workload_sizes_w_task1.size(); workload_size_w_index++) {
            
            for (size_t dx_index = 0; dx_index < dx_s_task1.size(); dx_index++) {

                float mstime = 0;

                for (size_t iteration_index = 0; iteration_index < num_iterations_task1; iteration_index++) {

                    task.compute(dx_s_task1[dx_index], 1, 1, "vectorAdd");
                    task.checkDefaultValues();
                    mstime += task.mstime;

                }

                mstime /= num_iterations_task1;
                //std::cout << "Naive version " << "took: " << mstime << " ms" << std::endl;

                csv.newRow() << dx_s_task1[dx_index] << 1 << 1 <<
                    workload_sizes_w_task1[workload_size_w_index] * workload_sizes_w_task1[workload_size_w_index] << mstime;

            }
        }

        std::cout << csv << std::endl;
        csv.writeToFile("vectorAddition.csv" ,false);


        CSVWriter csv_task2;

        csv_task2.newRow() << "dx" << "dy" << "dz" << "vecsize" << "timing naive" << "timing optimized";

        unsigned int num_iterations_task2 = 20;
        std::array<unsigned int, 3> workload_sizes_w_task2 = { 6400,4000,3200 };
        std::array<unsigned int, 3> workload_sizes_h = { 6400,4000,3200 };
        std::array<unsigned int, 3> dx_s_task2 = { 32,16,8 };
        std::array<unsigned int, 3> dy_s = { 32,16,8 };

        for (size_t workload_size_w_index = 0; workload_size_w_index < workload_sizes_w_task2.size(); workload_size_w_index++) {
            for (size_t workload_size_h_index = 0; workload_size_h_index < workload_sizes_h.size(); workload_size_h_index++) {
                for (size_t dx_index = 0; dx_index < dx_s_task2.size(); dx_index++) {
                    for (size_t dy_index = 0; dy_index < dy_s.size(); dy_index++) {

                        float mstime_naive = 0;
                        float mstime_optimized = 0;

                        A1_Task2 A1task2(app);
                        A1task2.prepare(workload_sizes_w_task2[workload_size_w_index], workload_sizes_h[workload_size_h_index]);

                        for (size_t iteration_index = 0; iteration_index < num_iterations_task2; iteration_index++) {

                            A1task2.compute(dx_s_task2[dx_index], dy_s[dy_index], 1, "matrixRotNaive");
                            A1task2.checkDefaultValues();

                            mstime_naive += A1task2.mstime;
                            //std::cout << "Naive version " << "took: " << mstime_naive << " ms" << std::endl;

                            A1task2.compute(dx_s_task2[dx_index], dy_s[dy_index], 1, "matrixRotOpti");
                            A1task2.checkDefaultValues();

                            mstime_optimized += A1task2.mstime;
                            //std::cout << "Optimized version " << "took: " << A1task2.mstime << " ms" << std::endl;

                        }

                        A1task2.cleanup();

                        mstime_naive /= num_iterations_task2;
                        mstime_optimized /= num_iterations_task2;

                        csv_task2.newRow() << dx_s_task2[dx_index] << dy_s[dy_index] << 1 <<
                            workload_sizes_w_task2[workload_size_w_index] * workload_sizes_h[workload_size_h_index] << mstime_naive << mstime_optimized;

                    }
                }
            }
        }

        csv_task2.writeToFile("matrixRot.csv", false);
        app.destroy();
    
        renderdoc::endCapture();

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