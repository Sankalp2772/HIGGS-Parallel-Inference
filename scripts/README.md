# Reproducible scripts

This directory contains the supporting code for dataset subset preparation, model training, model-weight export, and benchmark visualization.

## Dependencies

From the repository root:

```bash
python -m pip install -r requirements.txt
```

## 1. Prepare a manageable HIGGS subset

Download the official HIGGS dataset from the [UCI dataset page](https://archive.ics.uci.edu/dataset/280/higgs). The expected input is a headerless CSV with 29 columns: the label first, followed by 28 numerical features.

Create a compressed 1-million-row subset for the CPU and MPI implementations:

```bash
python scripts/prepare_subset.py data/HIGGS.csv.gz --rows 1000000 --output data/HIGGS_1M.csv.gz
```

Create a plain CSV for the CUDA implementation:

```bash
python scripts/prepare_subset.py data/HIGGS.csv.gz --rows 1000000 --output data/HIGGS_1M.csv
```

The script streams the file and writes only the requested number of rows. Output compression is inferred from the output filename: `.gz` produces gzip output; `.csv` produces plain CSV. The source dataset and generated subsets are ignored by Git.

## 2. Train the MLP

```bash
python scripts/train_model.py data/HIGGS.csv.gz --rows 100000
```

This fits a `StandardScaler` and a scikit-learn `MLPClassifier` with hidden layers `(32, 16)`, ReLU activation, Adam, batch size 256, maximum 20 iterations, and random seed 42. The default output is `model/trained_model.joblib`.

The script reports training accuracy for visibility. This is not an unbiased test score; use a separate held-out test set for a generalization claim. Because the initial benchmark subsets were drawn from the start of the dataset, they can overlap with the initial 100,000 training rows.

## 3. Export weights for the C++ and CUDA programs

```bash
python scripts/export_model.py
```

This reads `model/trained_model.joblib` and writes `model_weights.txt` at the repository root. The exporter validates the scaler and dense-layer shapes. The generated layout is compatible with all four inference loaders.

The exported weights and trained model bundle are derived artifacts and are excluded from Git. Recreate them locally after obtaining the dataset.

## 4. Generate benchmark tables and graphs

After updating measurements in `results/benchmark_results.csv`, run:

```bash
python scripts/generate_graphs.py
```

The script writes `results/derived_summary.csv` and PNG charts under `graphs/`. Commit regenerated outputs when updating the raw benchmark file so the published graphs remain synchronized with the data.

## Notes

- Do not commit the original multi-gigabyte dataset, large derived subsets, model bundle, or local model-weight file.
- Record package versions, data subset selection, and checksums for formal reproducibility.
- CPU and MPI inference expect gzip-compressed CSV input; CUDA expects a plain CSV.
