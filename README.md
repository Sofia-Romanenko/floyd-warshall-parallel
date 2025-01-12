# Floyd-Warshall Algorithm: Comparing Non-Parallel and CUDA Implementations

## Project Overview

This repository implements the Floyd-Warshall algorithm using two approaches:
1. **Non-parallel implementation**: A sequential version written in Python.
2. **CUDA implementation**: A GPU-accelerated version written in C++ using NVIDIA CUDA.

The project compares the performance of these implementations in terms of execution time and speedup, visualized through graphs.

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

## Purpose of the Project

The goal is to analyze the performance difference between a CPU-based sequential algorithm and a GPU-based parallel implementation. The Floyd-Warshall algorithm, with its \(O(n^3)\) complexity, is a computationally intensive task well-suited for demonstrating the advantages of parallelization using CUDA.

### Key Objectives:
1. **Performance Comparison**:
   - Measure execution time for both implementations on graphs of varying sizes.
   - Evaluate the scalability of each approach as the graph size increases.

2. **Speedup Visualization**:
   - Calculate the speedup (\( \text{Speedup} = \frac{\text{Time(Non-parallel)}}{\text{Time(CUDA)}} \)) to quantify the benefits of GPU acceleration.

---

## Features

- **Non-Parallel Implementation**:
  - A simple sequential Python implementation for baseline performance.

- **CUDA Implementation**:
  - A parallel version leveraging thousands of GPU cores for efficient computation.

- **Graph Visualization**:
  - Execution Time Plot: Shows the time taken by each implementation for different graph sizes.
  - Speedup Plot: Illustrates the relative improvement in performance of the CUDA implementation.

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

### **Graph 1: Execution Time**
The execution time plot compares the time taken by the non-parallel and CUDA implementations for varying graph sizes (\(n\)).

- **X-axis**: Graph size (\(n\)).
- **Y-axis**: Execution time (in seconds).
- **Insights**:
  - The CUDA implementation is consistently faster, especially as the graph size increases.

### **Graph 2: Speedup**
The speedup plot highlights how much faster the CUDA implementation is compared to the non-parallel version.

- **X-axis**: Graph size (\(n\)).
- **Y-axis**: Speedup (\( \text{Non-parallel Time} / \text{CUDA Time} \)).
- **Insights**:
  - The speedup increases as the graph size grows, demonstrating the scalability of the GPU-based solution.

Both graphs are saved in the `results/` folder:
- `execution_time_plot.png`
- `speedup_plot.png`

---
