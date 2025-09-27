#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int maxIteration) {
    float z_re = c_re, z_im = c_im;
    float z_re2 = z_re * z_re, z_im2 = z_im * z_im; // 預先計算平方
    int i;
    for (i = 0; i < maxIteration; ++i) {
        if (z_re2 + z_im2 > 4.f) break;
        z_im = 2.f * z_re * z_im + c_im;
        z_re = z_re2 - z_im2 + c_re;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;
    }
    return i;
}

__device__ bool earlyExitCheck(float x, float y) {
    double zx = sqrt((x - 0.25) * (x - 0.25) + y * y);
    if (x < zx - 2.0 * zx * zx + 0.25 || (x + 1.0) * (x + 1.0) + y * y < 0.0625) {
        return true;
    }
    return false;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int* d_img, size_t pitch, int resX, int resY, int maxIterations, int groupSize) {
    int startX = blockIdx.x * blockDim.x + threadIdx.x;
    int startY = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < groupSize; ++i) {
        for (int j = 0; j < groupSize; ++j) {
            int thisX = startX * groupSize + i;
            int thisY = startY * groupSize + j;
            if (thisX < resX && thisY < resY) {
                float x = lowerX + thisX * stepX;
                float y = lowerY + thisY * stepY;
                if (earlyExitCheck(x, y)) {
                    int* row = (int*)((char*)d_img + thisY * pitch);
                    row[thisX] = maxIterations;
                } else {
                    int* row = (int*)((char*)d_img + thisY * pitch);
                    row[thisX] = mandel(x, y, maxIterations);
                }
            }
        }
    }
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    size_t pitch;

    // Allocate host memory using cudaHostAlloc
    int* h_img;
    cudaHostAlloc((void**)&h_img, resX * resY * sizeof(int), cudaHostAllocDefault);

    // Allocate device memory using cudaMallocPitch
    int* d_img;
    cudaMallocPitch((void**)&d_img, &pitch, resX * sizeof(int), resY);

    // Define block and grid sizes, and group size
    int groupSize = 4; // You can try different sizes for performance optimization
    dim3 blockSize(16, 16);
    //dim3 gridSize((resX + blockSize.x * groupSize - 1) / (blockSize.x * groupSize), (resY + blockSize.y * groupSize - 1) / (blockSize.y * groupSize));
    dim3 gridSize(resX / blockSize.x / 4, resY / blockSize.y);
    
    // Calculate step sizes
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // Launch kernel
    mandelKernel<<<gridSize, blockSize>>>(lowerX, lowerY, stepX, stepY, d_img, pitch, resX, resY, maxIterations, groupSize);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy2D(h_img, resX * sizeof(int), d_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);

    // Copy results to the output image array using memcpy
    memcpy(img, h_img, resX * resY * sizeof(int));

    // Free device and host memory
    cudaFree(d_img);
    cudaFreeHost(h_img);
}
