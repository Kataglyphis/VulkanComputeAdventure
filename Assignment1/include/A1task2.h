#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"

struct A1_Task2 {
    struct PushStruct
    {
        uint32_t size_x;
        uint32_t size_y;
    };
    AppResources &app;
    TaskResources task;
    Buffer inBuffer, outBuffer;
    uint32_t workloadSize_w, workloadSize_h, workloadSize;
    float mstime = -1.f;

    A1_Task2(AppResources &app):app(app){}

    void cleanup() { 
        task.destroy( app.device ); 
        auto Bclean = [&](Buffer &b){
            app.device.destroyBuffer(b.buf);
            app.device.freeMemory(b.mem);}; 
        Bclean(inBuffer);
        Bclean(outBuffer);
    }
    void compute( uint32_t dx, uint32_t dy, uint32_t dz, std::string function="matrixRotNaive");
    
    void dispatchWork(uint32_t dx, uint32_t dy, uint32_t dz, PushStruct &pushConstant);
    void createSpecialization(vk::SpecializationInfo &specInfo, int workgroupSize);
    void prepare(unsigned int size_x, unsigned int size_y);
    void checkDefaultValues();
    void defaultValues();
    std::vector<int> incArray();

};

template<typename T>
std::vector<T> rotateCPU(const std::vector<T> &input, unsigned int w, unsigned int h){
    std::vector<T> vec(input.size(),0);
    for(unsigned int x = 0; x < w; x++)
	{
		for(unsigned int y = 0; y < h; y++)
		{
			vec[ x * h + (h - y - 1) ] = input[ y * w + x ];
		}
	}
    return vec;
}