#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int maxIteration) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < maxIteration; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f) break;
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
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

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int* d_img, size_t pitch, int resX, int resY, int maxIterations) {
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX < resX && thisY < resY) {
        float x = lowerX + thisX * stepX;
        float y = lowerY + thisY * stepY;

        // Early exit check
        int* row = (int*)((char*)d_img + thisY * pitch); // 使用 pitch 計算行偏移
        if (earlyExitCheck(x, y)) {
            row[thisX] = maxIterations; // 提早結束
        } else {
            row[thisX] = mandel(x, y, maxIterations); // 正常計算
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

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((resX + blockSize.x - 1) / blockSize.x, (resY + blockSize.y - 1) / blockSize.y);

    // Calculate step sizes
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // Launch kernel
    mandelKernel<<<gridSize, blockSize>>>(lowerX, lowerY, stepX, stepY, d_img, pitch, resX, resY, maxIterations);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy2D(h_img, resX * sizeof(int), d_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);

    // Copy results to the output image array
    memcpy(img, h_img, resX * resY * sizeof(int));

    // Free device and host memory
    cudaFree(d_img);
    cudaFreeHost(h_img);
}
