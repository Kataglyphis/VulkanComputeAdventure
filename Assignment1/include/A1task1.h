#include <iostream>
#include <cstdlib>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <vector>
#include "initialization.h"
#include "utils.h"
#include "task_common.h"

struct A1_Task1 {
    struct PushStruct
    {
        uint32_t size;
    };
    AppResources &app;
    TaskResources task;
    Buffer inBuffer1, inBuffer2, outBuffer;
    uint32_t workloadSize;

    float mstime = -1.f;

    A1_Task1(AppResources &app):app(app){}

    void cleanup() { 
        task.destroy( app.device ); 
        auto Bclean = [&](Buffer &b){
            app.device.destroyBuffer(b.buf);
            app.device.freeMemory(b.mem);}; 
        Bclean(inBuffer1);
        Bclean(inBuffer2);
        Bclean(outBuffer);
    }

    void compute( uint32_t dx, uint32_t dy, uint32_t dz, std::string file="vectorAdd");
    void defaultValues();
    void checkDefaultValues();
    void dispatchWork(uint32_t dx, uint32_t dy, uint32_t dz, PushStruct &pushConstant);
    void prepare(unsigned int size);

};
  