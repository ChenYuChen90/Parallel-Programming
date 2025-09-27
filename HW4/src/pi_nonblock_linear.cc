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

    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Send(&local_count, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        //MPI_Request requests[];

        long long int global_count = local_count;
        long long int *received_counts = (long long int *)malloc((world_size - 1) * sizeof(long long int));
        MPI_Request *requests = (MPI_Request *)malloc((world_size - 1) * sizeof(MPI_Request));
        MPI_Status *statuses = (MPI_Status *)malloc((world_size - 1) * sizeof(MPI_Status));

        for (int i = 1; i < world_size; i++)
        {
            MPI_Irecv(&received_counts[i - 1], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }
        
        MPI_Waitall(world_size - 1, requests, statuses);

        for (int i = 0; i < world_size - 1; i++)
        {
            global_count += received_counts[i];
        }

        // 计算 PI 值
        pi_result = 4.0 * (double)global_count / (double)tosses;

        free(received_counts);
        free(requests);
        free(statuses);
    }

    if (world_rank == 0)
    {
        // TODO: PI result

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
