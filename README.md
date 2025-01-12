# Floyd-Warshall Algorithm: Non-Parallel vs CUDA Implementation

## Project Overview
This repository implements and compares two versions of the Floyd-Warshall algorithm:
1. **Non-parallel implementation**: A sequential Python implementation.
2. **CUDA implementation**: A parallelized version written in C++ using NVIDIA CUDA to leverage GPU acceleration.

---

## Why CUDA for Parallelization?

CUDA was chosen for parallelizing the Floyd-Warshall algorithm because it offers the following advantages:

1. **High Computational Efficiency**:
   - CUDA is specifically designed for highly parallel tasks and excels at matrix operations, which form the core of the Floyd-Warshall algorithm.

2. **Scalability for Large Graphs**:
   - The algorithm's computational complexity is \(O(n^3)\). CUDA can handle these intensive computations effectively for large graphs (e.g., \(n = 500\) or more).

3. **Better Resource Utilization**:
   - Unlike multi-threading or distributed systems, CUDA utilizes thousands of GPU cores simultaneously, maximizing the use of hardware.

4. **Reduction in Overheads**:
   - While distributed systems (e.g., MPI or Spark) require communication between nodes, CUDA operates on a single device, minimizing data transfer overheads.

5. **Wide Ecosystem and Performance**:
   - NVIDIA GPUs are widely available and optimized for scientific computations, offering significant speedup compared to CPU-based solutions.

### **Comparison with Other Tools**

| **Tool**              | **Advantages**                                                                                     | **Limitations**                                                                              |
|-----------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **OpenMP**            | Simple setup, utilizes multiple CPU cores effectively.                                            | Limited by the number of cores on a single machine.                                         |
| **MPI**               | Scales across clusters, suitable for distributed environments.                                     | Requires inter-node communication, which adds latency.                                      |
| **Apache Spark**      | Excellent for very large datasets in distributed systems.                                         | High overhead for smaller-scale tasks like Floyd-Warshall.                                 |
| **CUDA**              | Leverages thousands of GPU cores for matrix-heavy computations, providing the best performance.   | Requires a GPU and knowledge of CUDA programming.                                           |
| **PyTorch Distributed**| Useful for tasks involving neural networks or multiple GPUs.                                      | Overkill for simple algorithms like Floyd-Warshall.                                        |
| **Numba**             | Simple integration with Python, JIT-compilation for CPU parallelism.                              | Limited to single-machine CPU computations.                                                |

**Why Not Other Tools?**
- **OpenMP and Numba**: Great for CPU-bound tasks, but GPUs outperform CPUs for matrix-based operations.
- **MPI**: Adds unnecessary overhead for single-machine execution.
- **Spark**: Ideal for big data applications but inefficient for algorithms with small datasets.
- **PyTorch Distributed**: Primarily suited for deep learning tasks, not optimal for simple algorithms.

CUDA was selected because it offers the **best performance for matrix-heavy tasks** like the Floyd-Warshall algorithm on modern hardware.

---

## Features
- **Non-Parallel Version**: A simple Python implementation for baseline performance.
- **CUDA Version**: Optimized C++ implementation with GPU acceleration.
- **Speedup Visualization**: The project includes a speedup plot that compares the non-parallel and CUDA versions.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Sofia-Romanenko/floyd-warshall-parallel.git
%cd floyd-warshall-parallel
```


### 2. Compile the CUDA Version
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
The benchmark script will:

- Run the non-parallel version.
- Run the CUDA version.
- Plot the speedup graph, saved as results/speedup_plot.png.
