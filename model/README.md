# Model artifacts

The four inference programs expect a local `model_weights.txt` file. The Sequential, OpenMP, and MPI programs load it from the current working directory; the CUDA program accepts an explicit model path as its third argument (or uses `model_weights.txt` by default).

The file contains:

- StandardScaler means and scales for 28 input features.
- Weights and biases for dense layers 28 → 32, 32 → 16, and 16 → 1.
- Layer-dimension headers parsed explicitly by CUDA and skipped by the CPU loaders.

## Recreate the model artifact

From the repository root, after obtaining the official HIGGS dataset locally:

```bash
python -m pip install -r requirements.txt
python scripts/train_model.py data/HIGGS.csv.gz --rows 100000
python scripts/export_model.py
```

The training bundle is written to `model/trained_model.joblib`, and the exported inference file is written to `model_weights.txt` at the repository root. Both are generated artifacts ignored by Git.

The documented initial model used 100,000 rows for training. The original benchmark subsets were also drawn from the beginning of the dataset and may overlap with the training sample; therefore, the reported accuracy is a cross-implementation correctness observation rather than a held-out generalization score.
