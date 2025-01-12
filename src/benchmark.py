import numpy as np
import time
import matplotlib.pyplot as plt
from floyd_warshall import floyd_warshall
from floyd_warshall_parallel import floyd_warshall_parallel

print("Loading adjacency matrix...")
try:
    adj_matrix = np.load("data/example_graph.npy")
    print("Adjacency matrix loaded successfully!")
except FileNotFoundError:
    print("Error: Adjacency matrix file not found. Please check 'data/example_graph.npy'.")
    exit()

print("Running non-parallel implementation...")
start = time.time()
floyd_warshall(adj_matrix)
non_parallel_time = time.time() - start
print(f"Non-parallel time: {non_parallel_time:.2f} seconds")

print("Running parallel implementation...")
workers_list = [1, 2, 4, 8]
parallel_times = []
for workers in workers_list:
    print(f"Running with {workers} workers...")
    start = time.time()
    floyd_warshall_parallel(adj_matrix, workers=workers)
    elapsed_time = time.time() - start
    parallel_times.append(elapsed_time)
    print(f"Time with {workers} workers: {elapsed_time:.2f} seconds")

print("Calculating speedup...")
speedup = [non_parallel_time / t for t in parallel_times]

print("Plotting results...")
plt.plot(workers_list, speedup, marker='o')
plt.xlabel("Number of Workers")
plt.ylabel("Speedup")
plt.title("Speedup vs Number of Workers")
plt.grid(True)
plt.savefig("results/speedup_plot.png")
print("Plot saved as 'results/speedup_plot.png'.")
