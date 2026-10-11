# Inference source code

This directory contains the four implementations of the same HIGGS neural-network inference workload.

| File | Model | Compile command |
|---|---|---|
| `ai_sequential.cpp` | Single-process CPU baseline | `g++ -O2 src/ai_sequential.cpp -o ai_sequential -lz` |
| `ai_openmp.cpp` | Shared-memory CPU threads | `g++ -O2 -fopenmp src/ai_openmp.cpp -o ai_openmp -lz` |
| `ai_mpi.cpp` | Distributed processes | `mpic++ -O2 src/ai_mpi.cpp -o ai_mpi -lz` |
| `ai_cuda.cu` | CUDA GPU kernel | `nvcc -O2 -std=c++17 -arch=compute_86 -code=sm_86 src/ai_cuda.cu -o ai_cuda.exe` (tested on RTX 2050 / SM 8.6) |

## Required local input

- `model_weights.txt`: created by `python scripts/train_model.py ...` followed by `python scripts/export_model.py`.
- CPU and MPI implementations expect a gzip-compressed CSV with the label in column 0 followed by 28 features.
- The CUDA implementation expects an uncompressed CSV with the same column layout.

Run commands from the repository root so the CPU programs can find `model_weights.txt`. See the root README for dataset preparation, training/export instructions, MPI host configuration, timing methodology, and benchmark caveats.

Do not commit generated binaries, the multi-gigabyte dataset, local subsets, or model artifacts.
