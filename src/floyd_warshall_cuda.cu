#include <iostream>
#include <cuda.h>
#include <limits>

#define INF 1e9

__global__ void floyd_warshall_kernel(int *dist, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        int ik = dist[i * n + k];
        int kj = dist[k * n + j];
        if (ik < INF && kj < INF) {
            atomicMin(&dist[i * n + j], ik + kj);
        }
    }
}

void floyd_warshall_cuda(int *h_dist, int n) {
    int *d_dist;
    size_t size = n * n * sizeof(int);

    // Copy data to GPU
    cudaMalloc(&d_dist, size);
    cudaMemcpy(d_dist, h_dist, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + 15) / 16, (n + 15) / 16);

    for (int k = 0; k < n; ++k) {
        floyd_warshall_kernel<<<numBlocks, threadsPerBlock>>>(d_dist, n, k);
        cudaDeviceSynchronize();
    }

    // Copy result back to CPU
    cudaMemcpy(h_dist, d_dist, size, cudaMemcpyDeviceToHost);
    cudaFree(d_dist);
}

int main() {
    int n = 500;
    int *dist = new int[n * n];

    // Initialize graph (example: random graph)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dist[i * n + j] = (i == j) ? 0 : (rand() % 100 + 1);
        }
    }

    floyd_warshall_cuda(dist, n);

    delete[] dist;
    return 0;
}
