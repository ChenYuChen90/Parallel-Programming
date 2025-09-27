#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"

#define GROUP_SIZE 8

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, NULL);

    // 創建緩衝區
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, filterSize, filter, NULL);
    cl_mem d_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageSize, inputImage, NULL);
    cl_mem d_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, NULL);

    // 創建內核
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);
    clSetKernelArg(kernel, 0, sizeof(int), &filterWidth);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_inputImage);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_outputImage);

    // 計算局部記憶體大小
    int halfFilter = filterWidth / 2;
    size_t localMemWidth = GROUP_SIZE + 2 * halfFilter;
    size_t localMemSize = localMemWidth * localMemWidth * sizeof(float);

    clSetKernelArg(kernel, 4, localMemSize, NULL); // 動態局部記憶體

    // 設定工作組大小和全局工作大小
    size_t local_work_size[] = {GROUP_SIZE, GROUP_SIZE};
    size_t global_work_size[] = {
        ((imageWidth + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE,
        ((imageHeight + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE
    };

    // 觸發內核
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    // 讀取結果
    clEnqueueReadBuffer(command_queue, d_outputImage, CL_TRUE, 0, imageSize, outputImage, 0, NULL, NULL);

    // 釋放資源
    clReleaseKernel(kernel);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_inputImage);
    clReleaseMemObject(d_outputImage);
    clReleaseCommandQueue(command_queue);
}
