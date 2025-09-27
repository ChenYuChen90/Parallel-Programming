#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long long int local_tosses = tosses / world_size;
    long long int local_count = 0;

    unsigned int seed = time(NULL) * world_rank;
    for (long long int i = 0; i < local_tosses; i++)
    {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0)
        {
            local_count++;
        }
    }

    long long int *global_count = NULL;
    if (world_rank == 0) {
        // Only rank 0 allocates the window memory
        global_count = (long long int*)malloc(sizeof(long long int));  // Fix here
        *global_count = 0;
    }

    MPI_Win win;
    // Create the window for one-sided communication
    MPI_Win_create(global_count, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Synchronize the window to ensure safe access
    MPI_Win_fence(0, win);

    // Use MPI_Accumulate to add the local counts to the global count
    MPI_Accumulate(&local_count, 1, MPI_LONG_LONG, 0, 0, 1, MPI_LONG_LONG, MPI_SUM, win);

    // Synchronize again after the accumulate to ensure all operations are completed
    MPI_Win_fence(0, win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * (double)(*global_count) / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    MPI_Win_free(&win);

    if (world_rank == 0) {
        free(global_count);  // Free here
    }

    MPI_Finalize();
    return 0;
}