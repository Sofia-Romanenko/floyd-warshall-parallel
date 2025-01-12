import numpy as np

def floyd_warshall(matrix):
    """
    Non-parallel implementation of the Floyd-Warshall algorithm.
    """
    n = len(matrix)
    dist = matrix.copy()
    for k in range(n):
        print(f"Processing intermediate node {k + 1}/{n}...")
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist
