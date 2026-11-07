# Own Xception Fine-Tuning

This folder contains a grouped training pipeline for fine-tuning an Xception model on the local `Dataset/` directory.

## Why this pipeline is stronger than the old notebook

- It uses a **grouped identity split** instead of random frame splitting.
- It produces **train, validation, and test metrics** separately.
- It saves **best and last checkpoints**.
- It exports **learning curves, confusion matrices, ROC curves, and error galleries** for report writing.
- It tracks **precision, recall, F1, accuracy, and ROC AUC**.

## Dataset assumption

This code is designed for the current dataset structure:

```text
Dataset/
|-- real/
|   |-- 000/
|   |   |-- 000.png
|   |   `-- ...
|   `-- ...
`-- fake/
    |-- Deepfakes/
    |   |-- 000_003/
    |   |   |-- 000.png
    |   |   `-- ...
    |   `-- ...
    |-- Face2Face/
    |-- FaceShifter/
    |-- FaceSwap/
    `-- NeuralTextures/
```

## Split strategy

The script first splits **real identity IDs** into train / val / test.

Then it keeps fake pair folders only if **both underlying IDs belong to the same split**. This avoids a much more subtle leakage problem where a source identity appears in both training and validation/test through paired fake folders.

## Install requirements

```powershell
pip install -r .\own_train\requirements.txt
```

## Run training

From the repository root:

```powershell
python -m own_train.train_xception
```

Example with custom settings:

```powershell
python -m own_train.train_xception --dataset-root Dataset --epochs 15 --batch-size 24 --image-size 299 --learning-rate 1e-4
```

## Important outputs

Each run creates a timestamped folder under:

```text
own_train/runs/
```

Inside each run:

```text
config.json
split_summary.json
history.json
final_report.json
checkpoints/
|-- best_model.pt
`-- last_model.pt
plots/
|-- learning_curves.png
|-- val_confusion_matrix.png
|-- test_confusion_matrix.png
|-- val_roc_curve.png
|-- test_roc_curve.png
|-- val_error_gallery.png
`-- test_error_gallery.png
reports/
|-- best_val_predictions.csv
|-- val_predictions.csv
`-- test_predictions.csv
```

## Metrics you can use in your report

- Train loss / validation loss
- Train accuracy / validation accuracy
- Train F1 / validation F1
- Train ROC AUC / validation ROC AUC
- Final validation and test accuracy
- Final validation and test precision / recall / F1 / ROC AUC
- Confusion matrices
- ROC curves
- Error galleries showing difficult mistakes

## Notes

- Default image size is `299`, which matches Xception more naturally than the older `128x128` notebook setup.
- The training pipeline uses ImageNet normalization because this is a proper fine-tuning setup with a pretrained Xception backbone.
- The training loop starts with the backbone frozen for a short warmup, then fine-tunes the full model.
