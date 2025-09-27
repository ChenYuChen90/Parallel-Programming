#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

	/*
		For PP students: Implement the page rank algorithm here.  You
		are expected to parallelize the algorithm using openMP.  Your
		solution may need to allocate (and free) temporary arrays.

		Basic page rank pseudocode is provided below to get you started:

		// initialization: see example code above
		score_old[vi] = 1/numNodes;
	*/
	int numNodes = num_nodes(g);
	double equal_prob = 1.0 / numNodes;
	#pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }
	/*
		while (!converged) {

		// compute score_new[vi] for all nodes vi:
		score_new[vi] = sum over all nodes vj reachable from incoming edges
							{ score_old[vj] / number of edges leaving vj  }
		score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

		score_new[vi] += sum over all nodes v in graph with no outgoing edges
							{ damping * score_old[v] / numNodes }

		// compute how much per-node scores have changed
		// quit once algorithm has converged

		global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
		converged = (global_diff < convergence)
		}

	*/
    auto *const temp_space = (double *)malloc(numNodes * sizeof(double));

	double *score_new, *score_old, *score_temp;
	score_new = solution;
	score_old = temp_space;
	score_temp = (double *)malloc(numNodes * sizeof(double));

	double global_diff = convergence;
	double no_outgoing_sum;

    while (global_diff >= convergence) {
		global_diff = 0.0;
		no_outgoing_sum = 0.0;

		std::swap(score_old, score_new);

		// 計算無出邊節點的貢獻
		#pragma omp parallel for reduction(+:no_outgoing_sum)
		for (int v = 0; v < numNodes; v++) {
			if (outgoing_size(g, v) == 0) {
				no_outgoing_sum += damping * score_old[v] / numNodes;
			}
			score_temp[v] = 0.0;
		}

		// 計算每個節點的分數總和
		#pragma omp parallel for
		for (int vi = 0; vi < numNodes; vi++) {
			const Vertex *start = incoming_begin(g, vi);
			const Vertex *end = incoming_end(g, vi);
			double local_sum = 0.0;

			for (const Vertex *vj = start; vj != end; vj++) {
				int out_size = outgoing_size(g, *vj);
				if (out_size > 0) {
					local_sum += score_old[*vj] / out_size;
				}
			}
			score_temp[vi] = local_sum; // 將局部累加結果存入 score_temp
		}

		// 更新 score_new 和計算全局差異
		#pragma omp parallel for reduction(+:global_diff)
		for (int vi = 0; vi < numNodes; vi++) {
			score_new[vi] = (damping * score_temp[vi]) + (1.0 - damping) / numNodes + no_outgoing_sum;

			double abs_diff = fabs(score_new[vi] - score_old[vi]);
			global_diff += abs_diff;
		}
    }

    // 将最终结果复制到solution中
    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = score_new[i];
    }

    free(temp_space);
}
