#include <stdio.h>
#include <stdlib.h>
#include <thread>

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

static inline int mandel(float c_re, float c_im)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < 256; ++i){
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

  return i;
}

void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow,
    int output[])
{   
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    for (int i = 0; i < width; ++i)
    {
        float x = x0 + i * dx;
        float y = y0 + startRow * dy;

        int index = (startRow * width + i);
        output[index] = mandel(x, y);
    }
}

void workerThreadStart(WorkerArgs *const args)
{
    for (unsigned start_row = args->threadId; start_row < args->height; start_row += args->numThreads ) {
        mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                        args->width, args->height,
                        start_row,
                        args->output);
    }
}

void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS] = {};

    WorkerArgs baseArgs;
    baseArgs.x0 = x0;
    baseArgs.y0 = y0;
    baseArgs.x1 = x1;
    baseArgs.y1 = y1;
    baseArgs.width = width;
    baseArgs.height = height;
    baseArgs.maxIterations = maxIterations;
    baseArgs.numThreads = numThreads;
    baseArgs.output = output;

    for (int i = 0; i < numThreads; i++)
    {
        args[i] = baseArgs;
        args[i].threadId = i;
    }

    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
