# Model artifact

The inference programs expect an exported `model_weights.txt` containing input-standardization parameters and the learned neural-network layer weights and biases. The artifact is currently kept locally and is not committed.

When adding the exporter, document the file format, model architecture, training configuration, dataset split/sample, dependencies, and provenance. Do not commit opaque model artifacts without this documentation.
