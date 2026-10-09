# HIGGS Parallel Inference
## Sequential vs OpenMP vs MPI vs CUDA

A parallel-computing study comparing neural-network inference performance across sequential CPU execution, shared-memory parallelism with OpenMP, distributed-memory parallelism with MPI, and GPU execution with CUDA. The workload uses a trained multilayer perceptron on the HIGGS particle-physics dataset.

> **Status:** Implementations have been exercised locally. Benchmark methodology, source organization, repeat measurements, and plots are being consolidated for reproducible reporting.

## Project goals

- Implement the same inference workload using Sequential C++, OpenMP, MPI, and CUDA.
- Verify correctness by comparing prediction counts, accuracy, and average predicted probability.
- Measure execution time at multiple dataset sizes and concurrency levels.
- Analyze throughput, speedup, scalability, and parallel efficiency while documenting measurement boundaries and hardware differences.

## Implementations

| Variant | Execution model | Current environment |
|---|---|---|
| Sequential | Single CPU process | Ubuntu VM |
| OpenMP | Shared-memory CPU threads | Ubuntu VM |
| MPI | Distributed processes across VMs | Ubuntu VM cluster |
| CUDA | GPU kernel execution | Windows host with NVIDIA GeForce RTX 2050 (4 GB) |

The CUDA implementation runs on the physical Windows host because the Ubuntu virtual machines use VMware virtual graphics and do not directly access the NVIDIA GPU. Hardware and operating-system differences must be considered when interpreting comparisons.

## Workload

The model is a feed-forward neural network with architecture:

`28 → 32 (ReLU) → 16 (ReLU) → 1 (sigmoid)`

Input standardization parameters and learned weights are exported from a trained scikit-learn model. The HIGGS dataset has a label and 28 numerical features per example. See the [official UCI HIGGS dataset page](https://archive.ics.uci.edu/dataset/280/higgs) for dataset details and access.

## Repository layout

```text
.
├── src/                  # Sequential, OpenMP, MPI, and CUDA implementations
├── scripts/              # Model training and weight-export scripts
├── model/                # Model format and provenance documentation
├── data/                 # Dataset acquisition and preparation instructions
├── results/              # Raw benchmark measurements and derived tables
├── graphs/               # Generated benchmark visualizations
├── report/               # Project report material
├── presentation/         # Presentation material
└── legacy/               # Original synthetic-workload implementation, if retained
```

Files will be added as the existing working implementations are consolidated. Large datasets and machine-specific executables are intentionally not stored in Git.

## Dataset and model setup

1. Obtain the HIGGS dataset through the official UCI page linked above and follow its license/usage terms.
2. Prepare the CSV or compressed CSV locally. Keep large data files out of version control.
3. Train/export the model with the scripts added to `scripts/`, or use a compatible exported model file.
4. Follow each implementation's build instructions once the source files are in `src/`.

The exported model weights are required for inference but are not yet committed. The `model/` directory will document the exact file format and generation steps.

## Benchmark methodology

- Report dataset size, process/thread count, CPU/GPU model, operating system, compiler/toolkit versions, and repeated-run measurements.
- Keep data parsing outside the measured inference interval when comparing inference-only results.
- Record whether communication and transfers are included. The initial MPI timing measured distributed inference after rank 0 had distributed the data, so it excludes distribution time.
- The CUDA timer measures GPU kernel execution and excludes CSV parsing, host-to-device transfer, and model loading. Do **not** compare this directly with CPU timings as a fully end-to-end speedup without aligning timing boundaries.
- Report repeated runs and arithmetic mean, and retain raw runs instead of reporting only the mean.
- Validate correctness before interpreting performance.

## Initial CUDA validation (1,000,000 rows)

GPU: NVIDIA GeForce RTX 2050 (4 GB)

| Run | GPU inference time | Reported throughput |
|---|---:|---:|
| 1 | 0.093202 s | 10,729,418 rows/s |
| 2 | 0.094690 s | 10,560,733 rows/s |
| 3 | 0.093929 s | 10,646,339 rows/s |
| **Mean** | **0.093940 s** | **approximately 10.65 million rows/s** |

All three runs reported 718,779 correct predictions, accuracy 0.718779, and average predicted probability 0.531864. These results are a preliminary record; final tables and graphs will be generated from a checked-in raw benchmark CSV.

## Planned outputs

- Raw benchmark CSV across implementations, dataset sizes, and concurrency levels
- Execution-time, speedup, efficiency, and scalability graphs where the measurement model supports them
- Reproducible build/run commands for each implementation
- Technical report and viva/presentation notes

## Limitations

- The implementations run on different machines and operating systems; comparisons should state this clearly.
- Initial MPI timings exclude data distribution/communication overhead.
- Initial CUDA timings cover kernel execution only and exclude data loading and memory transfers.
- Speedup and efficiency claims should only be made after the raw measurements and timing boundaries have been reviewed.

## Citation

Dataset: Baldi, P., Sadowski, P., & Whiteson, D. “Searching for exotic particles in high-energy physics with deep learning.” *Nature Communications* 5, 4308 (2014). Dataset page: https://archive.ics.uci.edu/dataset/280/higgs
