import numpy as np
import time
import matplotlib.pyplot as plt
from floyd_warshall import floyd_warshall
from floyd_warshall_parallel import floyd_warshall_parallel

if __name__ == "__main__":
    # Load the example graph
    adj_matrix = np.load("data/example_graph.npy")

    # Non-parallel execution
    start = time.time()
    floyd_warshall(adj_matrix)
    non_parallel_time = time.time() - start

    # Parallel execution with varying workers
    workers_list = [1, 2, 4, 8]
    parallel_times = []
    for workers in workers_list:
        start = time.time()
        floyd_warshall_parallel(adj_matrix, workers=workers)
        parallel_times.append(time.time() - start)

    # Calculate speedup
    speedup = [non_parallel_time / t for t in parallel_times]

    # Plot speedup graph
    plt.plot(workers_list, speedup, marker='o')
    plt.xlabel("Number of Workers")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Number of Workers")
    plt.grid(True)
    plt.savefig("results/speedup_plot.png")
    plt.show()

