def floyd_warshall(matrix):
    n = len(matrix)
    dist = matrix.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist
