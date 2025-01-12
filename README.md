# Floyd-Warshall Algorithm: Parallel and Non-Parallel Implementations

## Project Overview
This project contains implementations of the Floyd-Warshall algorithm:
1. **Non-parallel version**, executed sequentially.
2. **Parallel version**, using Python's `multiprocessing` library to speed up computations.

The input graph is represented as a 500-node adjacency matrix with a 0.5 probability of connection between nodes. The graph is saved in the file `data/example_graph.npy`.

---

## Project Features
- **Parallelization**: The parallel version distributes the computation of adjacency matrix rows across multiple processes.
- **Data**: The input graph is included in the repository under the `data/` folder.
- **Benchmarking**: The provided code compares the performance of both implementations and generates a speedup graph.

---

## How to Use the Repository
### 1. Clone the Repository
Clone the repository to your cloud environment (or work directly in GitHub Codespaces/Google Colab):
```bash
git clone https://github.com/<your-username>/floyd-warshall-parallel.git
cd floyd-warshall-parallel
