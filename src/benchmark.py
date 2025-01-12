import os
import numpy as np
import time
import subprocess
import matplotlib.pyplot as plt

from floyd_warshall import floyd_warshall


def generate_random_matrix(n=500, p=0.5, weight_range=(1, 100)):
    """
    Generate a random adjacency matrix for a graph.
    """
    np.random.seed(42)
    matrix = np.random.choice(
        [np.inf, np.random.randint(weight_range[0], weight_range[1] + 1)],
        size=(n, n),
        p=[1 - p, p],
    )
    np.fill_diagonal(matrix, 0)
    return matrix


def run_cuda_version():
    """
    Run the CUDA version of the algorithm and measure its time.
    """
    start = time.time()
    subprocess.run(["./floyd_warshall_cuda"], stdout=subprocess.DEVNULL)
    return time.time() - start


def plot_execution_times(sizes, non_parallel_times, cuda_times):
    """
    Plot a line graph comparing execution times for different graph sizes.
    """
    if not os.path.exists("results"):
        os.makedirs("results")

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, non_parallel_times, marker="o", label="Non-parallel")
    plt.plot(sizes, cuda_times, marker="o", label="CUDA")
    plt.xlabel("Graph Size (n)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time vs Graph Size")
    plt.legend()
    plt.grid()
    plt.savefig("results/execution_time_plot.png")
    print("Execution time plot saved as 'results/execution_time_plot.png'.")


def plot_speedup(sizes, non_parallel_times, cuda_times):
    """
    Plot a speedup graph showing the efficiency of the CUDA version.
    """
    speedups = [np / cp for np, cp in zip(non_parallel_times, cuda_times)]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, speedups, marker="o", color="orange")
    plt.xlabel("Graph Size (n)")
    plt.ylabel("Speedup")
    plt.title("Speedup of CUDA over Non-Parallel")
    plt.axhline(1, color="red", linestyle="--", label="No Speedup")
    plt.legend()
    plt.grid()
    plt.savefig("results/speedup_plot.png")
    print("Speedup plot saved as 'results/speedup_plot.png'.")


if __name__ == "__main__":
    # Graph sizes to test
    sizes = [100, 200, 300, 400, 500]
    non_parallel_times = []
    cuda_times = []

    for n in sizes:
        print(f"Testing graph size: {n}")
        adj_matrix = generate_random_matrix(n)

        # Non-parallel version
        print("Running non-parallel implementation...")
        start = time.time()
        floyd_warshall(adj_matrix)
        non_parallel_time = time.time() - start
        non_parallel_times.append(non_parallel_time)
        print(f"Non-parallel time: {non_parallel_time:.2f} seconds")

        # CUDA version
        print("Running CUDA implementation...")
        cuda_time = run_cuda_version()
        cuda_times.append(cuda_time)
        print(f"CUDA time: {cuda_time:.2f} seconds")

    # Plot execution times
    plot_execution_times(sizes, non_parallel_times, cuda_times)

    # Plot speedup
    plot_speedup(sizes, non_parallel_times, cuda_times)
