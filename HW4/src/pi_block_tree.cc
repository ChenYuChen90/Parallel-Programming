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

    // TODO: binary tree redunction
    int step = 1;
    while (step < world_size) {
        if (world_rank % (2 * step) == 0) {
            // Receive data from the right child
            if (world_rank + step < world_size) {
                long long int recv_count;
                MPI_Recv(&recv_count, 1, MPI_LONG_LONG, world_rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_count += recv_count;  // Accumulate the result
            }
        } else {
            // Send data to the parent (left child)
            MPI_Send(&local_count, 1, MPI_LONG_LONG, world_rank - step, 0, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 * local_count / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
