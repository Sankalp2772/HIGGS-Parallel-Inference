# Dataset

Use the official [UCI HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs). Do not commit the original dataset or large derived CSV subsets to this repository.

The expected format is a headerless CSV with 29 columns: binary label first, followed by 28 numerical features. A downloaded gzip-compressed file can be stored locally as `data/HIGGS.csv.gz`.

## Prepare benchmark subsets

Install dependencies using `python -m pip install -r requirements.txt`, then run from the repository root.

Gzip subset for the Sequential, OpenMP, and MPI programs:

```bash
python scripts/prepare_subset.py data/HIGGS.csv.gz --rows 1000000 --output data/HIGGS_1M.csv.gz
```

Plain CSV subset for the CUDA program:

```bash
python scripts/prepare_subset.py data/HIGGS.csv.gz --rows 1000000 --output data/HIGGS_1M.csv
```

The script streams rows rather than loading the complete dataset into memory. A `.gz` output extension enables compression; any other output extension is written as plain CSV.

For reproducibility, record the dataset version, subset-selection method, row count, and a checksum when creating new benchmark subsets. Keep generated data local; the ignore rules exclude HIGGS dataset files from Git.
