import numpy as np
import time
import subprocess

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
    subprocess.run(["./floyd_warshall_cuda"])
    return time.time() - start

if __name__ == "__main__":
    # Generate random graph
    n = 500
    adj_matrix = generate_random_matrix(n)

    # Non-parallel version
    print("Running non-parallel implementation...")
    start = time.time()
    floyd_warshall(adj_matrix)
    non_parallel_time = time.time() - start
    print(f"Non-parallel time: {non_parallel_time:.2f} seconds")

    # CUDA version
    print("Running CUDA implementation...")
    cuda_time = run_cuda_version()
    print(f"CUDA time: {cuda_time:.2f} seconds")

    # Speedup
    speedup = non_parallel_time / cuda_time
    print(f"Speedup: {speedup:.2f}x")
