#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

#define GROUP_SIZE 8  // 定義 Group Size，根據硬體最佳值調整

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int imageSize = imageHeight * imageWidth;
    int filterSize = filterWidth * filterWidth;

    // Step 1: Create OpenCL command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);
    if (status != CL_SUCCESS) {
        printf("Failed to create command queue. Error: %d\n", status);
        exit(1);
    }

    // Step 2: Create OpenCL buffers
    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                        sizeof(float) * imageSize, inputImage, &status);
    if (status != CL_SUCCESS) {
        printf("Failed to create input buffer. Error: %d\n", status);
        exit(1);
    }

    // 將 Filter 設定為 Constant Memory
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, 
                                        sizeof(float) * filterSize, filter, &status);
    if (status != CL_SUCCESS) {
        printf("Failed to create filter buffer. Error: %d\n", status);
        exit(1);
    }

    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, 
                                        sizeof(float) * imageSize, NULL, &status);
    if (status != CL_SUCCESS) {
        printf("Failed to create output buffer. Error: %d\n", status);
        exit(1);
    }

    // Step 3: Create and build the program
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    if (status != CL_SUCCESS) {
        printf("Failed to create kernel. Error: %d\n", status);
        exit(1);
    }

    // Step 4: Set kernel arguments
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterBuffer); // Filter 作為 Constant Memory
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuffer);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &imageHeight);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &imageWidth);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);
    
    // 計算 Local Memory 的大小
    int halo = filterWidth / 2;
    size_t localMemWidth = GROUP_SIZE + 2 * halo;
    size_t localMemSize = localMemWidth * localMemWidth * sizeof(float);
    status |= clSetKernelArg(kernel, 6, localMemSize, NULL); // 新增 Local Memory 參數

    if (status != CL_SUCCESS) {
        printf("Failed to set kernel arguments. Error: %d\n", status);
        exit(1);
    }

    // Step 5: Define global and local work size
    size_t localWorkSize[2] = {GROUP_SIZE, GROUP_SIZE};
    size_t globalWorkSize[2] = {
        ((imageHeight + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE,  // 向上取整
        ((imageWidth + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE
    };

    // Step 6: Enqueue kernel execution
    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Failed to enqueue kernel. Error: %d\n", status);
        exit(1);
    }

    // Step 7: Read the output buffer back to the host
    status = clEnqueueReadBuffer(command_queue, outputBuffer, CL_TRUE, 0, 
                                 sizeof(float) * imageSize, outputImage, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Failed to read output buffer. Error: %d\n", status);
        exit(1);
    }

    // Step 8: Clean up
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
}
