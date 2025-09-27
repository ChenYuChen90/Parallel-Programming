#define GROUP_SIZE 8

__kernel void convolution(const int filterWidth,
                          __constant const float *restrict filter,
                          __global const float *restrict inputImage,
                          __global float *restrict outputImage,
                          __local float *restrict localMem) {
    // 計算半濾波器尺寸
    int halfFilter = filterWidth / 2;

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);

    int imageWidth = get_global_size(0);
    int imageHeight = get_global_size(1);

    // 定義局部記憶體的寬度
    int localMemWidth = GROUP_SIZE + 2 * halfFilter;

    // 加載輸入圖像到局部記憶體，包含 halo 區域
    for(int dy = -halfFilter; dy <= halfFilter; dy++) {
        for(int dx = -halfFilter; dx <= halfFilter; dx++) {
            int img_x = clamp(global_x + dx, 0, imageWidth - 1);
            int img_y = clamp(global_y + dy, 0, imageHeight - 1);
            int local_mem_x = local_x + halfFilter + dx;
            int local_mem_y = local_y + halfFilter + dy;
            int local_index = local_mem_y * localMemWidth + local_mem_x;
            int img_index = img_y * imageWidth + img_x;
            localMem[local_index] = inputImage[img_index];
        }
    }

    // 確保所有工作項完成載入
    barrier(CLK_LOCAL_MEM_FENCE);

    // 開始卷積計算
    float sum = 0.0f;
    for(int fy = -halfFilter; fy <= halfFilter; fy++) {
        for(int fx = -halfFilter; fx <= halfFilter; fx++) {
            float filter_val = filter[(fy + halfFilter) * filterWidth + (fx + halfFilter)];
            int local_mem_x = local_x + halfFilter + fx;
            int local_mem_y = local_y + halfFilter + fy;
            int local_index = local_mem_y * localMemWidth + local_mem_x;
            float image_val = localMem[local_index];
            sum += filter_val * image_val;
        }
    }

    // 將結果寫回全局記憶體
    outputImage[global_y * imageWidth + global_x] = sum;
}
