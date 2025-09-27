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
    double zx = sqrt((x - 0.25) * (x - 0.25) + y * y); // Replace hypot with sqrt to make it device-compatible
    if (x < zx - 2.0 * zx * zx + 0.25 || (x + 1.0) * (x + 1.0) + y * y < 0.0625) {
        return true;
    }
    return false;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int* d_img, size_t pitch, int resX, int resY, int maxIterations) {
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX < resX && thisY < resY) {
        float x = lowerX + thisX * stepX;
        float y = lowerY + thisY * stepY;
        int* row = (int*)((char*)d_img + thisY * pitch);

        if (earlyExitCheck(x, y)) {
            row[thisX] = maxIterations;
        } else {
            row[thisX] = mandel(x, y, maxIterations);
        }
    }
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    size_t pitch;

    // Allocate intermediate host memory and register it using cudaHostRegister
    int* h_img;
    cudaHostAlloc((void**)&h_img, resX * resY * sizeof(int), cudaHostAllocDefault);

    // Allocate device memory using cudaMallocPitch
    int* d_img;
    cudaMallocPitch((void**)&d_img, &pitch, resX * sizeof(int), resY);

    // Define block and grid sizes
    int blockSizeDimX = 16; // Block size in X
    int blockSizeDimY = 16; // Block size in Y

    dim3 block(blockSizeDimX, blockSizeDimY);
    dim3 grid((resX + block.x - 1) / block.x, (resY + block.y - 1) / block.y);

    // Calculate step sizes
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // Launch kernel
    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, d_img, pitch, resX, resY, maxIterations);
    cudaDeviceSynchronize();

    // Copy results back to host using cudaMemcpy2D
    cudaMemcpy2D(h_img, resX * sizeof(int), d_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);

    // Copy intermediate results to the output image array using memcpy
    memcpy(img, h_img, resX * resY * sizeof(int));

    // Free device and intermediate host memory
    cudaFree(d_img);
    cudaFreeHost(h_img);
}
