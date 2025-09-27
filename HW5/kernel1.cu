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

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int* d_img, int resX, int resY, int maxIterations) {
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX < resX && thisY < resY) {
        float x = lowerX + thisX * stepX;
        float y = lowerY + thisY * stepY;

        // Early exit check
        if (earlyExitCheck(x, y)) {
            d_img[thisY * resX + thisX] = maxIterations; // 提早結束時設定為最大迭代次數
        } else {
            d_img[thisY * resX + thisX] = mandel(x, y, maxIterations); // 否則正常計算
        }
    }
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    int imgSize = resX * resY * sizeof(int);

    // Allocate host memory
    int* h_img = (int*)malloc(imgSize);
    if (h_img == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Allocate device memory
    int* d_img;
    cudaMalloc((void**)&d_img, imgSize);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((resX + blockSize.x - 1) / blockSize.x, (resY + blockSize.y - 1) / blockSize.y);

    // Calculate step sizes
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // Launch kernel
    mandelKernel<<<gridSize, blockSize>>>(lowerX, lowerY, stepX, stepY, d_img, resX, resY, maxIterations);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_img, d_img, imgSize, cudaMemcpyDeviceToHost);

    // Copy results to the output image array
    memcpy(img, h_img, resX * resY * sizeof(int));

    // Free device and host memory
    cudaFree(d_img);
    free(h_img);
}
