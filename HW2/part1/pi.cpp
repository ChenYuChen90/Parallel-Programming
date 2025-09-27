#include <iostream>
#include <random>
#include <omp.h>
#include <pthread.h>

using namespace std;

void *toss(void *__restrict partial_num_in_circle) {
    long long toss_num = *((long long *) partial_num_in_circle);
    long long num_in_circle = 0;

    thread_local unsigned int seed = static_cast<unsigned int>(pthread_self());

    for (long long i = 0; i < toss_num; i++) {
        double x = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        double y = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        if (x * x + y * y <= 1) {
            num_in_circle++;
        }
    }

    *((long long *) partial_num_in_circle) = num_in_circle;
    return nullptr;
}

int main(int argc, char *__restrict argv[]) {
    int thread_num = atoi(argv[1]);
    long long total_toss = atoll(argv[2]);

    pthread_t threads[thread_num];
    long long *partial_num_in_circle[thread_num];
    long long num_in_circle = 0;
    long long thread_toss = (total_toss / thread_num);
    int remainder = total_toss % thread_num;

    for (int i = 0; i < thread_num; i++) {
        partial_num_in_circle[i] = new long long;
        *partial_num_in_circle[i] = thread_toss;
        if (remainder) {
            remainder--;
            (*partial_num_in_circle[i])++;
        }
        pthread_create(&threads[i], nullptr, toss, (void *) partial_num_in_circle[i]);
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(threads[i], nullptr);
        num_in_circle += *partial_num_in_circle[i];
        delete partial_num_in_circle[i];
    }

    printf("%.7lf\n", 4 * (num_in_circle / static_cast<double>(total_toss)));
    return 0;
}
