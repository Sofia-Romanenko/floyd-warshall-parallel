# Floyd-Warshall Algorithm: Non-Parallel vs CUDA Implementation

## Project Overview
This repository compares the performance of:
1. **Non-parallel implementation**: Written in Python.
2. **CUDA implementation**: Written in C++ using CUDA for GPU acceleration.

The input graph is generated dynamically as an adjacency matrix with random weights.

---

## Features
- **Non-Parallel Version**: Simple Python implementation for baseline performance.
- **CUDA Version**: Optimized GPU implementation for high-speed computation.
- **Benchmark**: Compares the execution time and calculates speedup.

---

## How to Run

### 1. Compile the CUDA version
```bash
nvcc src/floyd_warshall_cuda.cu -o floyd_warshall_cuda
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
- The benchmark script will display the execution time for both implementations and the calculated speedup
