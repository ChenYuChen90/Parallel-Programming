__kernel void convolution(__global const float *inputImage, 
                          __constant const float *filter,
                          __global float *outputImage,
                          const int imageHeight, const int imageWidth, 
                          const int filterWidth,
                          __local float *localImage)
{
    // 計算 Halo 大小
    int halo = filterWidth / 2;

    // 獲取全局和本地的索引
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    // 獲取工作群組的尺寸
    int groupSizeX = get_local_size(0);
    int groupSizeY = get_local_size(1);

    // 計算 Local Memory 的寬度（包括 Halo）
    int localWidth = groupSizeY + 2 * halo;
    int localHeight = groupSizeX + 2 * halo;

    // 計算 Local Memory 的索引
    int localIndex = (localRow + halo) * localWidth + (localCol + halo);

    // 將中心區域載入 Local Memory
    if (globalRow < imageHeight && globalCol < imageWidth) {
        localImage[localIndex] = inputImage[globalRow * imageWidth + globalCol];
    } else {
        localImage[localIndex] = 0.0f; // 超出影像邊界時填充 0
    }

    // 載入 Halo 區域 (上、下、左、右)
    // 上方
    if (localRow < halo) {
        int haloRow = globalRow - halo;
        int haloCol = globalCol;
        if (haloRow >= 0 && haloCol < imageWidth) {
            localImage[(localRow) * localWidth + (localCol + halo)] = inputImage[haloRow * imageWidth + haloCol];
        } else {
            localImage[(localRow) * localWidth + (localCol + halo)] = 0.0f;
        }
    }

    // 下方
    if (localRow >= groupSizeX - halo) {
        int haloRow = globalRow + halo;
        int haloCol = globalCol;
        if (haloRow < imageHeight && haloCol < imageWidth) {
            localImage[(localRow + 2 * halo) * localWidth + (localCol + halo)] = inputImage[haloRow * imageWidth + haloCol];
        } else {
            localImage[(localRow + 2 * halo) * localWidth + (localCol + halo)] = 0.0f;
        }
    }

    // 左方
    if (localCol < halo) {
        int haloRow = globalRow;
        int haloCol = globalCol - halo;
        if (haloRow < imageHeight && haloCol >= 0) {
            localImage[(localRow + halo) * localWidth + (localCol)] = inputImage[haloRow * imageWidth + haloCol];
        } else {
            localImage[(localRow + halo) * localWidth + (localCol)] = 0.0f;
        }
    }

    // 右方
    if (localCol >= groupSizeY - halo) {
        int haloRow = globalRow;
        int haloCol = globalCol + halo;
        if (haloRow < imageHeighbt && haloCol < imageWidth) {
            localImage[(localRow + halo) * localWidth + (localCol + 2 * halo)] = inputImage[haloRow * imageWidth + haloCol];
        } else {
            localImage[(localRow + halo) * localWidth + (localCol + 2 * halo)] = 0.0f;
        }
    }

    // 載入角落區域（左上、右上、左下、右下）
    if (localRow < halo && localCol < halo) { // 左上
        int haloRow = globalRow - halo;
        int haloCol = globalCol - halo;
        if (haloRow >= 0 && haloCol >= 0) {
            localImage[(localRow) * localWidth + (localCol)] = inputImage[haloRow * imageWidth + haloCol];
        } else {
            localImage[(localRow) * localWidth + (localCol)] = 0.0f;
        }
    }

    if (localRow < halo && localCol >= groupSizeY - halo) { // 右上
        int haloRow = globalRow - halo;
        int haloCol = globalCol + halo;
        if (haloRow >= 0 && haloCol < imageWidth) {
            localImage[(localRow) * localWidth + (localCol + 2 * halo)] = inputImage[haloRow * imageWidth + haloCol];
        } else {
            localImage[(localRow) * localWidth + (localCol + 2 * halo)] = 0.0f;
        }
    }

    if (localRow >= groupSizeX - halo && localCol < halo) { // 左下
        int haloRow = globalRow + halo;
        int haloCol = globalCol - halo;
        if (haloRow < imageHeight && haloCol >= 0) {
            localImage[(localRow + 2 * halo) * localWidth + (localCol)] = inputImage[haloRow * imageWidth + haloCol];
        } else {
            localImage[(localRow + 2 * halo) * localWidth + (localCol)] = 0.0f;
        }
    }

    if (localRow >= groupSizeX - halo && localCol >= groupSizeY - halo) { // 右下
        int haloRow = globalRow + halo;
        int haloCol = globalCol + halo;
        if (haloRow < imageHeight && haloCol < imageWidth) {
            localImage[(localRow + 2 * halo) * localWidth + (localCol + 2 * halo)] = inputImage[haloRow * imageWidth + haloCol];
        } else {
            localImage[(localRow + 2 * halo) * localWidth + (localCol + 2 * halo)] = 0.0f;
        }
    }

    // 確保所有工作項目都已經載入 Local Memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // 檢查是否在影像範圍內
    if (globalRow >= imageHeight || globalCol >= imageWidth) {
        return;
    }

    // 執行卷積運算
    float sum = 0.0f;
    for (int k = -halo; k <= halo; k++) {
        for (int l = -halo; l <= halo; l++) {
            float pixel = localImage[(localRow + halo + k) * localWidth + (localCol + halo + l)];
            float filterValue = filter[(k + halo) * filterWidth + (l + halo)];
            sum += pixel * filterValue; // 使用 mad 指令可視情況進行替換
        }
    }

    // 將結果寫回輸出影像
    outputImage[globalRow * imageWidth + globalCol] = sum;
}
