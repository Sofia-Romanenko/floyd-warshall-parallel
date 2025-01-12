from multiprocessing import Pool
import numpy as np

def update_row(args):
    i, k, dist = args
    n = len(dist)
    for j in range(n):
        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist[i]

def floyd_warshall_parallel(matrix, workers=4):
    n = len(matrix)
    dist = matrix.copy()
    for k in range(n):
        with Pool(workers) as pool:
            rows = pool.map(update_row, [(i, k, dist) for i in range(n)])
        dist = np.array(rows)
    return dist
