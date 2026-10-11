# HIGGS Parallel Inference
## Sequential vs OpenMP vs MPI vs CUDA

**Parallel and Grid Computing — Experimental Project**

This project evaluates the performance of one neural-network inference workload implemented with four approaches: sequential CPU execution, shared-memory parallelism with OpenMP, distributed-memory processing with MPI, and GPU execution with CUDA.

The workload uses the HIGGS high-energy-physics dataset and a trained multilayer perceptron (MLP). The project emphasizes correctness checks, repeatable measurements, and a careful explanation of the differences between shared-memory, distributed-memory, and GPU execution.

> **Important measurement note:** The CPU implementations run on Ubuntu virtual machines, MPI uses a VM cluster, and CUDA runs on the Windows host's NVIDIA GeForce RTX 2050. Current timing boundaries are not perfectly identical across implementations. Cross-platform times should not be presented as rigorous apples-to-apples speedups.

## Table of Contents

- [1. Objective](#1-objective)
- [2. Problem Definition and Mathematical Formulation](#2-problem-definition-and-mathematical-formulation)
- [3. Experimental Environment](#3-experimental-environment)
- [4. Repository Structure](#4-repository-structure)
- [5. System Architecture and Workflow](#5-system-architecture-and-workflow)
- [6. Sequential CPU Implementation](#6-sequential-cpu-implementation)
- [7. OpenMP Implementation](#7-openmp-implementation)
- [8. MPI Implementation](#8-mpi-implementation)
- [9. CUDA Implementation](#9-cuda-implementation)
- [10. Benchmark Methodology](#10-benchmark-methodology)
- [11. Actual Experimental Results](#11-actual-experimental-results)
- [12. Speedup and Parallel Efficiency](#12-speedup-and-parallel-efficiency)
- [13. Graphs and Result Files](#13-graphs-and-result-files)
- [14. Architectural Comparison](#14-architectural-comparison)
- [15. Correctness Verification and Interpretation](#15-correctness-verification-and-interpretation)
- [16. Compilation and Execution Guide](#16-compilation-and-execution-guide)
- [17. Limitations and Future Work](#17-limitations-and-future-work)
- [18. Key Observations](#18-key-observations)
- [19. Conclusion](#19-conclusion)
- [Technologies Used](#technologies-used)
- [Experiment Summary](#experiment-summary)
- [Author](#author)

---

## 1. Objective

The primary objectives are:

- Implement the same neural-network inference computation using sequential C++, OpenMP, MPI, and CUDA.
- Verify that the implementations process the same inputs and produce matching prediction counts, accuracy values, and average predicted probabilities.
- Measure inference time across multiple dataset sizes and thread/process configurations.
- Investigate speedup, parallel efficiency, scaling, synchronization overhead, and the impact of concurrency.
- Compare shared-memory CPU parallelism, distributed-memory message passing, and GPU execution while documenting hardware and timing differences.
- Preserve raw results and scripts so the summary tables and plots can be regenerated.

## 2. Problem Definition and Mathematical Formulation

### 2.1 Dataset and input

The project uses the [UCI HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs). Each example contains a binary class label and 28 numerical input features. The original dataset contains approximately 11 million examples.

The large dataset files are kept outside GitHub. This repository contains dataset acquisition instructions and benchmark records, not the full multi-gigabyte dataset.

### 2.2 Neural-network inference

The exported model standardizes the input features and evaluates a fully connected neural network:

~~~text
28 input features
      |
StandardScaler transformation
      |
Dense layer: 28 -> 32
      |
ReLU activation
      |
Dense layer: 32 -> 16
      |
ReLU activation
      |
Dense layer: 16 -> 1
      |
Sigmoid probability
      |
Binary prediction: probability >= 0.5
~~~

For feature x_j, standardization is:

~~~text
z_j = (x_j - mean_j) / scale_j
~~~

The ReLU activation is max(0, x). The output layer uses the sigmoid function:

~~~text
sigmoid(x) = 1 / (1 + exp(-x))
~~~

The predicted class is 1 when the model output is at least 0.5, and 0 otherwise. Each implementation accumulates correct predictions and the sum of predicted probabilities while timing inference.

### 2.3 Model training and model file

The model used for the initial benchmarks was trained with scikit-learn using hidden layers of 32 and 16 neurons, Adam optimization, batch size 256, a maximum of 20 iterations, and random seed 42. The trained scaler parameters and neural-network weights are exported to a text file named **model_weights.txt**.

The model file is intentionally not committed at present. The large HIGGS data files are also excluded from version control. Obtain these assets locally and ensure the model file is available in the working directory expected by each program.

**Important evaluation caveat:** The initial model was trained using the first 100,000 rows, while benchmark subsets were also created from the beginning of the dataset. Therefore, the reported prediction accuracy is a pipeline/correctness observation, not an unbiased held-out generalization estimate. A final machine-learning accuracy claim should be based on a clearly separated test set that was not used for training.

## 3. Experimental Environment

### 3.1 CPU and MPI environment

- Operating system: Ubuntu 24.04.4 LTS virtual machines.
- C++ compiler: GCC 13.3.0.
- OpenMP: GCC OpenMP runtime, enabled with the -fopenmp compiler flag.
- MPI: Open MPI 4.1.6.
- Input parsing: zlib, used to read compressed CSV data.
- VM allocation: approximately 2 vCPUs per VM; exact host CPU model was not recorded.
- MPI execution: processes distributed across the Ubuntu VM cluster.

### 3.2 CUDA environment

- Host operating system: Windows.
- GPU: NVIDIA GeForce RTX 2050, 4 GB VRAM.
- Reported GPU compute capability: 8.6.
- CUDA Toolkit: 13.4.
- CUDA compilation target: compute_86 / sm_86.
- Build environment: Visual Studio x64 Native Tools / Developer Command Prompt.

The Ubuntu virtual machines do not directly access the physical RTX 2050 through VMware, so CUDA runs on the Windows host. This environmental difference matters when interpreting comparisons.

## 4. Repository Structure

~~~text
HIGGS-Parallel-Inference/
├── README.md
├── .gitignore
├── src/
│   ├── ai_sequential.cpp       # Sequential C++ inference
│   ├── ai_openmp.cpp           # OpenMP shared-memory inference
│   ├── ai_mpi.cpp              # MPI distributed-process inference
│   └── ai_cuda.cu              # CUDA GPU inference
├── scripts/
│   └── generate_graphs.py      # Generate derived summaries and plots
├── model/
│   └── README.md               # Model artifact and format notes
├── data/
│   └── README.md               # Dataset acquisition notes
├── results/
│   ├── benchmark_results.csv   # Raw recorded measurements
│   └── derived_summary.csv     # Grouped statistics, generated locally
├── graphs/
│   ├── cpu_execution_time_vs_rows.png
│   ├── cuda_kernel_time_vs_rows.png
│   ├── openmp_speedup_100k.png
│   ├── openmp_efficiency_100k.png
│   ├── mpi_speedup_100k.png
│   └── mpi_efficiency_100k.png
├── report/
│   └── README.md
├── presentation/
│   └── README.md
└── legacy/
    └── README.md
~~~

Generated graph images and derived_summary.csv should be committed after running the generator. If these files are not yet visible in GitHub, they are local outputs and still need to be added and pushed.

## 5. System Architecture and Workflow

The four implementations perform the same logical computation but use different execution models.

~~~text
                 HIGGS CSV + model_weights.txt
                              |
                  Parse rows and features
                              |
                   Standardized model input
                              |
         +--------------------+---------------------+
         |                    |                     |
   Sequential CPU          OpenMP                 MPI
   one execution       shared-memory        multiple processes
       context          CPU threads          across VM cluster
         |                    |                     |
         +--------------------+---------------------+
                              |
                   Correct predictions and
                    mean output probability

                   CUDA path on Windows host
                              |
                      Host input parsing
                              |
                 Copy inputs and weights to GPU
                              |
                   CUDA inference kernel
                              |
                     Copy summary results back
~~~

### Logical algorithm

1. Load the exported model and requested dataset subset.
2. Parse the label and 28 feature values for each valid row.
3. Standardize features using the stored scaler mean and scale.
4. Run the 28 → 32 → 16 → 1 neural network.
5. Convert each probability to a binary prediction using the 0.5 threshold.
6. Accumulate correct predictions and the sum of probabilities.
7. Report processed rows, accuracy, average probability, and elapsed time.

The sequential version performs inference serially. OpenMP distributes row inference across threads. MPI distributes rows among processes and reduces summary values. CUDA assigns rows to GPU threads in a kernel.

## 6. Sequential CPU Implementation

Source file: **src/ai_sequential.cpp**

The sequential implementation serves as the single-process CPU baseline. It loads the model and dataset, then evaluates rows one by one. CSV parsing occurs before the measured inference loop.

Compile on Ubuntu:

~~~bash
g++ -O2 src/ai_sequential.cpp -o ai_sequential -lz
~~~

Run from a directory containing model_weights.txt and the compressed dataset:

~~~bash
./ai_sequential HIGGS_1M.csv.gz 1000000
~~~

Change the requested row count to match an available dataset subset. The input filename must refer to a gzip-compressed CSV for this implementation.

## 7. OpenMP Implementation

Source file: **src/ai_openmp.cpp**

OpenMP uses shared-memory CPU threads to process different rows concurrently. Each thread calculates local correct-prediction and probability totals, which are aggregated after the parallel loop. The OMP_NUM_THREADS environment variable can be used to control thread count.

Compile:

~~~bash
g++ -O2 -fopenmp src/ai_openmp.cpp -o ai_openmp -lz
~~~

Run using two threads:

~~~bash
OMP_NUM_THREADS=2 ./ai_openmp HIGGS_1M.csv.gz 1000000
~~~

Configurations tested include 1, 2, 4, and 8 threads. More threads do not necessarily improve performance when the VM has only two virtual CPUs; oversubscription and runtime overhead can outweigh the additional concurrency.

## 8. MPI Implementation

Source file: **src/ai_mpi.cpp**

MPI divides the dataset across processes. The root rank reads the dataset and distributes row chunks; each process computes local inference summaries. MPI reduction operations combine correct-prediction counts, probability sums, and the maximum process inference time.

Compile on Ubuntu with Open MPI:

~~~bash
mpic++ -O2 src/ai_mpi.cpp -o ai_mpi -lz
~~~

Run on a configured MPI environment with a valid host file:

~~~bash
mpirun -np 4 --hostfile hostfile ./ai_mpi HIGGS_1M.csv.gz 1000000
~~~

The hostfile must list reachable cluster hosts using the names or addresses configured on the VMs. SSH access, MPI installation, executable availability, and the model file must be configured on the participating machines. Adapt the host file and launch command to the actual cluster configuration.

**Timing caveat:** The recorded MPI timer begins after the input-distribution phase, so current MPI values exclude initial data-distribution time. They are not complete end-to-end application runtimes.

## 9. CUDA Implementation

Source file: **src/ai_cuda.cu**

CUDA runs the inference kernel on the physical RTX 2050. Each GPU thread processes one input row. The implementation performs standardization, both hidden layers with ReLU, the sigmoid output, and prediction aggregation. It performs a warm-up kernel before the timed run.

Compile from the Windows Developer Command Prompt where nvcc and the Visual Studio C++ compiler are available:

~~~bat
nvcc -O2 -std=c++17 -arch=compute_86 -code=sm_86 src\ai_cuda.cu -o ai_cuda.exe
~~~

Run with a plain, decompressed CSV and the exported model file:

~~~bat
ai_cuda.exe C:\path\to\HIGGS_1M.csv 1000000 C:\path\to\model_weights.txt
~~~

Replace the paths with the actual local file locations. Unlike the CPU programs, this CUDA program expects plain CSV input rather than a gzip-compressed file.

**Timing scope:** The CUDA inference timer covers the measured kernel launch through device synchronization. CSV parsing, model loading, host-to-device input/model transfers, and result copy-back are outside this interval. Report this as kernel-focused inference time rather than end-to-end application time.

## 10. Benchmark Methodology

- Raw measurements are stored in **results/benchmark_results.csv**.
- Recorded columns include implementation, dataset size, thread/process count, run index, elapsed time, correct predictions, accuracy, average probability, timing scope, and environment.
- For configurations with repeated runs, the graph generator calculates the arithmetic mean and sample standard deviation.
- Dataset parsing/loading occurs before the timed inference region for the CPU implementations.
- MPI timings currently exclude initial data distribution.
- CUDA timings exclude file parsing, model loading, data transfer, and result copy-back.
- CPU and CUDA experiments ran in different physical/virtual environments.
- Large HIGGS input files, binaries, and the model artifact are intentionally excluded from Git.
- Only comparisons with compatible timing boundaries should be used to claim speedup. Cross-platform CPU-to-GPU numbers are informative context, but are not an apples-to-apples end-to-end benchmark.

## 11. Actual Experimental Results

### 11.1 Dataset size: 100,000 rows

All implementations reported 72,287 correct predictions, accuracy 0.722870, and average probability 0.531431 for this subset.

| Implementation | Configuration | Inference time (s) |
|---|---:|---:|
| Sequential | 1 CPU process | 0.201418 |
| OpenMP | 1 thread | 0.198609 |
| OpenMP | 2 threads | 0.112685 |
| OpenMP | 4 threads | 0.095396 |
| OpenMP | 8 threads | 0.115471 |
| MPI | 1 process | 0.192301 |
| MPI | 2 processes | 0.093517 |
| MPI | 4 processes | 0.054491 |
| CUDA | RTX 2050, kernel timing | 0.012869 |

The OpenMP measurements show that the 4-thread configuration was faster than the tested 2-thread configuration for this particular 100K-row run, while the 8-thread configuration was slower than 4 threads. Because this VM has only two virtual CPUs and these entries are single runs, repeat measurements are needed before treating the ordering as stable.

MPI was tested at 1, 2, and 4 processes. Since data distribution is excluded from the MPI timer, these results describe the measured distributed-inference phase, not full input-to-output latency.

### 11.2 Dataset size: 1,000,000 rows

The following mean times come from three recorded runs per configuration:

| Implementation | Configuration | Runs | Mean inference time (s) |
|---|---:|---:|---:|
| Sequential | 1 CPU process | 3 | 1.935797 |
| OpenMP | 2 threads | 3 | 0.886268 |
| MPI | 4 processes | 3 | 0.526114 |
| CUDA | RTX 2050, kernel timing | 3 | 0.093940 |

All implementations reported the same summary values for this subset:

| Metric | Value |
|---|---:|
| Rows processed | 1,000,000 |
| Correct predictions | 718,779 |
| Accuracy / correct-prediction fraction | 0.718779 |
| Average probability | 0.531864 |

These measurements establish consistent outputs for the benchmark subset. However, MPI excludes input distribution, CUDA excludes parsing and transfers, and the implementations ran on different systems. This is a reference summary, not a fully normalized cross-platform performance ranking.

### 11.3 Dataset size: 5,000,000 rows

The available large-subset measurements are:

| Implementation | Configuration | Recorded inference time (s) |
|---|---:|---:|
| Sequential | 1 CPU process | 9.408730 |
| OpenMP | 2 threads | 4.445460 |
| OpenMP | 4 threads | 5.139790 |
| MPI | 4 processes | 2.349000 |

Each 5M entry is a single recorded run. The MPI timing caveat still applies. Repeated measurements would improve confidence in the performance trend.

## 12. Speedup and Parallel Efficiency

Speedup is defined as:

~~~text
Speedup S = T_baseline / T_parallel
~~~

Parallel efficiency is:

~~~text
Efficiency E = S / P
~~~

where P is the number of threads or processes. Efficiency may be shown as a ratio or multiplied by 100 to express a percentage.

For the OpenMP graph, the baseline is the sequential 100K-row measurement. For the MPI graph, the baseline is the MPI 100K-row, one-process measurement. These baselines are kept separate because MPI has a different timing boundary.

The graph script calculates speedup and efficiency only for these specific configurations. Interpret the values as results from the current benchmark setup, not universal properties of OpenMP or MPI.

## 13. Graphs and Result Files

The graph generator reads the raw benchmark CSV and writes grouped statistics and PNG charts.

- [Raw benchmark measurements](results/benchmark_results.csv)
- [Derived benchmark summary](results/derived_summary.csv)
- [Graph generation script](scripts/generate_graphs.py)

Expected generated figures:

- **graphs/cpu_execution_time_vs_rows.png** — CPU inference time versus row count.
- **graphs/cuda_kernel_time_vs_rows.png** — CUDA kernel timing versus row count.
- **graphs/openmp_speedup_100k.png** — OpenMP speedup versus thread count.
- **graphs/openmp_efficiency_100k.png** — OpenMP efficiency versus thread count.
- **graphs/mpi_speedup_100k.png** — MPI speedup versus process count.
- **graphs/mpi_efficiency_100k.png** — MPI efficiency versus process count.

When committed to GitHub, these links will open the corresponding graph images:

- [CPU execution-time graph](graphs/cpu_execution_time_vs_rows.png)
- [CUDA kernel-time graph](graphs/cuda_kernel_time_vs_rows.png)
- [OpenMP speedup graph](graphs/openmp_speedup_100k.png)
- [OpenMP efficiency graph](graphs/openmp_efficiency_100k.png)
- [MPI speedup graph](graphs/mpi_speedup_100k.png)
- [MPI efficiency graph](graphs/mpi_efficiency_100k.png)

Regenerate locally by installing Matplotlib and running from the repository root:

~~~bash
python3 -m pip install matplotlib
python3 scripts/generate_graphs.py
~~~

On Windows:

~~~bat
py -m pip install matplotlib
py scripts\generate_graphs.py
~~~

The script reads the 41 recorded measurements, groups results by implementation, dataset size, and concurrency configuration, and writes results/derived_summary.csv and the PNG figures.

## 14. Architectural Comparison

| Feature | Sequential CPU | OpenMP | MPI | CUDA |
|---|---|---|---|---|
| Execution model | One CPU execution context | Shared-memory threads | Separate processes with message passing | Many GPU threads in a kernel |
| Main parallelism | None | Thread/loop parallelism | Process/rank parallelism | Fine-grained data parallelism |
| Memory model | Host memory | Shared host address space | Separate process memory; explicit communication | Host memory plus GPU device memory |
| Work distribution | Serial loop | OpenMP runtime work sharing | Root distributes work and ranks compute chunks | GPU grid maps threads to rows |
| Communication overhead | None between workers | Shared memory and synchronization | Message passing/data distribution | Host-device transfer |
| Typical use | Baseline and smaller jobs | Multicore CPU loops | Multi-process or multi-node workloads | Data-parallel workloads with sufficient GPU work |

No paradigm is always fastest. Performance depends on workload size, hardware allocation, memory movement, runtime overhead, process/thread placement, and how much work is included in the timer.

## 15. Correctness Verification and Interpretation

For the tested subsets, the CPU and CUDA implementations reported matching values for correct predictions, accuracy, and average probability. At one million rows, all reported:

- Correct predictions: 718,779
- Accuracy / correct-prediction fraction: 0.718779
- Average probability: 0.531864

Matching outputs provide a useful cross-implementation check that the programs are performing equivalent inference calculations on this subset, subject to numerical precision and aggregation differences.

This does **not** establish an unbiased machine-learning generalization score because benchmark rows may overlap with training data, as described in Section 2. It also does not imply that the kernel-only CUDA timer is directly comparable to complete CPU or MPI application latency.

## 16. Compilation and Execution Guide

### 16.1 Install Ubuntu dependencies

~~~bash
sudo apt update
sudo apt install -y build-essential zlib1g-dev openmpi-bin libopenmpi-dev
~~~

### 16.2 Build CPU programs

Run from the repository root:

~~~bash
g++ -O2 src/ai_sequential.cpp -o ai_sequential -lz
g++ -O2 -fopenmp src/ai_openmp.cpp -o ai_openmp -lz
mpic++ -O2 src/ai_mpi.cpp -o ai_mpi -lz
~~~

The commands create executables in the current directory. The exported model_weights.txt must be present in the working directory used for execution.

### 16.3 Run CPU programs

~~~bash
./ai_sequential HIGGS_1M.csv.gz 1000000
OMP_NUM_THREADS=2 ./ai_openmp HIGGS_1M.csv.gz 1000000
mpirun -np 4 --hostfile hostfile ./ai_mpi HIGGS_1M.csv.gz 1000000
~~~

Ensure the dataset and model paths match each machine's setup. For MPI runs, configure the host file and make the executable and model available on the cluster nodes as required by the cluster setup.

### 16.4 Build and run CUDA on Windows

Open the Visual Studio x64 Native Tools / Developer Command Prompt. Confirm nvcc and cl.exe are available, then compile for the tested SM 8.6 GPU target.

~~~bat
nvcc -O2 -std=c++17 -arch=compute_86 -code=sm_86 src\ai_cuda.cu -o ai_cuda.exe
ai_cuda.exe C:\path\to\HIGGS_1M.csv 1000000 C:\path\to\model_weights.txt
~~~

Replace the example paths with the real locations of the plain CSV and model file.

## 17. Limitations and Future Work

1. **Different hardware environments:** CPU/MPI run in Ubuntu VMs while CUDA runs on the physical Windows GPU. Results should clearly separate hardware effects from programming-model effects.
2. **Different timing scopes:** MPI excludes initial data distribution; CUDA excludes parsing and transfers. An end-to-end benchmark should measure setup, transfer/distribution, inference, and output stages separately.
3. **Limited repeats:** Several dataset-size/configuration pairs have only one timing sample. Repeat all configurations and report variation before drawing strong conclusions.
4. **Model evaluation split:** Inference subsets may overlap with model-training data. Create an explicit held-out test set and do not use training rows to report generalization accuracy.
5. **Reproducible model training:** Check in and document the training/export workflow after the scripts, package versions, and model-generation process are finalized.
6. **Input reproducibility:** Dataset and model artifacts are local. Record the exact subset selection, data source, and checksums for each benchmark input.
7. **End-to-end GPU comparison:** Record host-to-device transfer and kernel timing separately, as well as total application latency.
8. **Large-scale MPI evaluation:** Repeat the same row counts with consistent process placements and include communication costs when measuring the complete distributed pipeline.

## 18. Key Observations

1. All four implementations have been exercised on the HIGGS neural-network inference workload.
2. For one million rows, repeated runs averaged approximately 1.935797 s sequentially, 0.886268 s with two OpenMP threads, 0.526114 s with four MPI processes, and 0.093940 s for the CUDA kernel-focused timing interval.
3. OpenMP performance did not improve monotonically as thread count increased. On the tested two-vCPU VM, additional threads can introduce scheduling and synchronization overhead.
4. MPI performance depends on process count and cluster configuration; recorded inference timings exclude initial data distribution.
5. CUDA reported over 10 million rows per second for the one-million-row kernel-focused benchmark. This is not end-to-end application throughput because preprocessing and transfer costs are excluded.
6. Matching prediction outputs across the implementations is a useful correctness check, but it does not support an unqualified timing ranking across different systems.
7. Keeping raw CSV records and a plotting script makes the experiment easier to reproduce, inspect, and extend.

## 19. Conclusion

This project implements a common neural-network inference workload across four parallel-computing paradigms: sequential CPU execution, OpenMP shared-memory threading, MPI distributed processing, and CUDA GPU execution.

The experiments show how runtime changes with dataset size and concurrency, while highlighting the influence of hardware allocation, data distribution, synchronization, memory transfers, and timing boundaries. Repeated one-million-row results showed stable model outputs across the implementations and provided a useful reference for performance analysis.

The next step toward a rigorous comparative study is to align end-to-end timing scopes, repeat every configuration, record complete system metadata, and evaluate the model on a genuinely held-out dataset. These improvements will make the results more defensible for the final report and viva.

---

## Technologies Used

- **Languages:** C++ and CUDA C++
- **CPU parallelism:** OpenMP
- **Distributed parallelism:** MPI / Open MPI
- **GPU computing:** NVIDIA CUDA
- **Machine learning:** scikit-learn MLP and feature standardization
- **Dataset:** UCI HIGGS
- **Input compression:** gzip and zlib
- **Data analysis and plotting:** Python, CSV, Matplotlib
- **Compilers:** GCC and NVIDIA nvcc with MSVC build tools
- **Version control:** Git and GitHub

## Experiment Summary

| Parameter | Value |
|---|---|
| Dataset | HIGGS |
| Features per row | 28 |
| Neural-network architecture | 28 → 32 → 16 → 1 |
| Implementations | Sequential, OpenMP, MPI, CUDA |
| Largest recorded CPU/MPI subset | 5,000,000 rows |
| CUDA repeated benchmark | 1,000,000 rows, 3 runs |
| CUDA mean kernel timing | 0.093940 seconds |
| One-million-row correct predictions | 718,779 |
| One-million-row reported accuracy fraction | 0.718779 |
| One-million-row average probability | 0.531864 |
| Raw measurement records | 41 |
| Derived grouped summary records | 29 |

## Author

**Sankalp Prakash Patil**  
B.Tech — Computer Science & Artificial Intelligence  
KLE Technological University
