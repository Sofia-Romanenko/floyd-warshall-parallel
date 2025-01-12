from multiprocessing import Pool
import numpy as np

def update_row(args):
    """
    Updates a single row of the distance matrix for a given k.
    """
    i, k, dist = args
    n = len(dist)
    for j in range(n):
        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist[i]

def floyd_warshall_parallel_optimized(matrix, workers=4):
    """
    Parallel implementation of the Floyd-Warshall algorithm.
    """
    n = len(matrix)
    dist = matrix.copy()
    for k in range(n):
        dist_k = dist[k].copy()  # Extract the k-th row once
        with Pool(workers) as pool:
            updated_rows = pool.map(update_row, [(i, k, dist) for i in range(n)])
        dist = np.array(updated_rows)
    return dist
