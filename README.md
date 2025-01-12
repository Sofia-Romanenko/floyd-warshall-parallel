# Floyd-Warshall Algorithm: Parallel and Non-Parallel Implementations

## Project Overview
This repository contains implementations of the Floyd-Warshall algorithm:
1. **Non-parallel version**: A sequential implementation for baseline comparison.
2. **Parallel version**: A faster implementation using Python's `multiprocessing` library.

The input graph is dynamically generated as an adjacency matrix with random weights.

---

## Features
- **Parallelization**: The parallel implementation processes rows of the adjacency matrix in parallel.
- **Benchmarking**: Compares the performance of both implementations and generates a speedup graph.

---

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/floyd-warshall-parallel.git
cd floyd-warshall-parallel
```

### 2. Install Dependencies
Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Benchmark
Execute the benchmark to test both implementations and generate the speedup graph:

```bash
python src/benchmark.py
```

---

## Results
- Input Data: The script dynamically generates a random graph with customizable size and density.
- Performance:
  - The parallel implementation is designed to scale with the number of workers.
  - The generated speedup graph shows the performance improvement.
-Visualization: The graph is saved as results/speedup_plot.png.
