import numpy as np
import time
import matplotlib.pyplot as plt
from floyd_warshall import floyd_warshall
from floyd_warshall_parallel import floyd_warshall_parallel

def generate_random_matrix(n=500, p=0.5, weight_range=(1, 100)):
    matrix = np.random.choice(
        [np.inf, np.random.randint(weight_range[0], weight_range[1] + 1)],
        size=(n, n),
        p=[1 - p, p],
    )
    np.fill_diagonal(matrix, 0)
    return matrix

if __name__ == "__main__":
    adj_matrix = generate_random_matrix()
    np.save("data/example_graph.npy", adj_matrix)

    non_parallel_time = time.time()
    floyd_warshall(adj_matrix)
    non_parallel_time = time.time() - non_parallel_time

    workers_list = [1, 2, 4, 8]
    parallel_times = []

    for workers in workers_list:
        start = time.time()
        floyd_warshall_parallel(adj_matrix, workers=workers)
        parallel_times.append(time.time() - start)

    speedup = [non_parallel_time / t for t in parallel_times]

    plt.plot(workers_list, speedup, marker='o')
    plt.xlabel("Number of Workers")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Number of Workers")
    plt.grid(True)
    plt.savefig("results/speedup_plot.png")
    plt.show()
