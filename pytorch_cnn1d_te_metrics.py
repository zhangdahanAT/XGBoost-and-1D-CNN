# -*- coding: utf-8 -*-
"""
PyTorch 1D CNN for continuous TE prediction.

Compatible with PyTorch 2.0.1. The script automatically uses CUDA when it is
available; otherwise it runs on CPU. The current user's torch 2.0.1+cpu is
fully supported.

The network is trained as a regression model. Classification metrics are
derived by binarizing true and predicted TE values with CLASS_THRESHOLD. AUC
uses the raw continuous prediction as its ranking score.

Install dependencies:
    pip install torch numpy pandas scipy scikit-learn matplotlib openpyxl joblib

Run:
    python pytorch_cnn1d_te_metrics.py --file "C:/path/assembled_features.xlsx"

Main outputs in RESULT_DIR:
    metrics_summary.xlsx / metrics_summary.csv
    predictions.xlsx / predictions.csv
    training_history.csv / training_history.png
    roc_curve_test.png
    pytorch_cnn1d_te_model.pt
    preprocessing.joblib
    used_features.csv
    run_config.json
"""

import argparse
import copy
import json
import os
import random
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ======================== User-editable configuration ========================
FILE = r"C:/Users/Administrator/Desktop/xgboost-new/data/assembled_features.xlsx"
SHEET_FEATURES = "filtered_tezheng"
SHEET_TARGET = "filtered_TE"
RESULT_DIR = "pytorch_cnn1d_te_results"

# The feature order is the order along the CNN's 1D axis. Edit as needed.
WANTED_FEATURES = [
    "stem_ratio_5utr",
    "overall_gc_5utr",
    "uorf",
    "kozak_flag",
    "len_cds",
    "overall_gc_cds",
]

TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.2
RANDOM_STATE = 42
CLASS_THRESHOLD = 0.50

EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 40
REDUCE_LR_PATIENCE = 12
NUM_WORKERS = 0  # Keep 0 on Windows for the most reliable execution.
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PyTorch 1D CNN for TE prediction")
    parser.add_argument("--file", default=FILE, help="Input Excel workbook")
    parser.add_argument("--features-sheet", default=SHEET_FEATURES)
    parser.add_argument("--target-sheet", default=SHEET_TARGET)
    parser.add_argument("--result-dir", default=RESULT_DIR)
    parser.add_argument("--threshold", type=float, default=CLASS_THRESHOLD)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def set_reproducible_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_align(file, feature_sheet, target_sheet):
    """Read, de-duplicate and align the two Excel sheets by gene."""
    feat_df = pd.read_excel(file, sheet_name=feature_sheet, engine="openpyxl")
    target_df = pd.read_excel(file, sheet_name=target_sheet, engine="openpyxl")

    if feat_df.shape[1] < 2:
        raise ValueError("The feature sheet must contain a gene column and features.")
    if target_df.shape[1] < 2:
        raise ValueError("The target sheet must contain gene and TE columns.")

    feat_df = feat_df.rename(columns={feat_df.columns[0]: "gene"})
    target_df = target_df.rename(
        columns={target_df.columns[0]: "gene", target_df.columns[1]: "TE"}
    )
    feat_df = feat_df.drop_duplicates(subset="gene", keep="first")
    target_df = target_df.drop_duplicates(subset="gene", keep="first")

    aligned = pd.merge(
        feat_df,
        target_df[["gene", "TE"]],
        on="gene",
        how="inner",
        validate="one_to_one",
    )
    aligned["TE"] = pd.to_numeric(aligned["TE"], errors="coerce")
    aligned = aligned.dropna(subset=["TE"]).reset_index(drop=True)

    present = [name for name in WANTED_FEATURES if name in aligned.columns]
    missing = [name for name in WANTED_FEATURES if name not in aligned.columns]
    if missing:
        print("[Warning] Missing features will be skipped:", ", ".join(missing))
    if not present:
        raise ValueError("None of WANTED_FEATURES were found in the feature sheet.")
    if len(aligned) < 10:
        raise ValueError("Too few aligned samples. At least 10 samples are recommended.")

    X = aligned[present].apply(pd.to_numeric, errors="coerce")
    y = aligned["TE"].astype(float)
    genes = aligned["gene"].astype(str)
    return X, y, genes


def safe_stratify(y, threshold):
    """Use threshold classes for stratification when both classes are viable."""
    classes = (np.asarray(y) >= threshold).astype(int)
    counts = np.bincount(classes, minlength=2)
    return classes if np.all(counts >= 2) else None


def robust_train_test_split(indices, test_size, seed, strata):
    """Fall back to an unstratified split when a small class cannot be split."""
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=strata,
        )
    except ValueError:
        print("[Warning] Stratified split was not possible; using a random split.")
        return train_test_split(indices, test_size=test_size, random_state=seed)


def split_data(X, y, genes, threshold):
    indices = np.arange(len(y))
    trainval_idx, test_idx = robust_train_test_split(
        indices, TEST_SIZE, RANDOM_STATE, safe_stratify(y, threshold)
    )
    train_idx, val_idx = robust_train_test_split(
        trainval_idx,
        VALIDATION_SIZE_WITHIN_TRAIN,
        RANDOM_STATE,
        safe_stratify(y.iloc[trainval_idx], threshold),
    )

    def take(idx):
        return (
            X.iloc[idx].reset_index(drop=True),
            y.iloc[idx].reset_index(drop=True),
            genes.iloc[idx].reset_index(drop=True),
        )

    return take(train_idx), take(val_idx), take(test_idx)


def fit_preprocessor(X_train, X_val, X_test):
    """Fit imputation and scaling on training data only."""
    input_features = list(X_train.columns)
    imputer_features = X_train.columns[~X_train.isna().all()].tolist()
    if not imputer_features:
        raise ValueError("All selected features are completely missing in the training set.")
    all_missing = [name for name in input_features if name not in imputer_features]
    if all_missing:
        print("[Warning] All-missing training features will be skipped:", ", ".join(all_missing))

    imputer = SimpleImputer(strategy="median")
    train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train[imputer_features]), columns=imputer_features
    )
    val_imputed = pd.DataFrame(
        imputer.transform(X_val[imputer_features]), columns=imputer_features
    )
    test_imputed = pd.DataFrame(
        imputer.transform(X_test[imputer_features]), columns=imputer_features
    )

    keep_features = train_imputed.columns[
        train_imputed.nunique(dropna=False) > 1
    ].tolist()
    if not keep_features:
        raise ValueError("All selected features are constant in the training set.")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed[keep_features])
    val_scaled = scaler.transform(val_imputed[keep_features])
    test_scaled = scaler.transform(test_imputed[keep_features])

    # Conv1d input shape: (samples, channels, sequence_length).
    as_cnn = lambda array: np.asarray(array, dtype=np.float32)[:, np.newaxis, :]
    preprocessing = {
        "imputer": imputer,
        "scaler": scaler,
        "input_features": input_features,
        "imputer_features": imputer_features,
        "used_features": keep_features,
    }
    return as_cnn(train_scaled), as_cnn(val_scaled), as_cnn(test_scaled), preprocessing


class CNN1DRegressor(nn.Module):
    """Small 1D CNN suitable for a short ordered feature vector."""

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.regressor(self.feature_extractor(x)).squeeze(1)


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.as_tensor(np.asarray(y), dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * len(targets)
            total_n += len(targets)
    return total_loss / total_n


def train_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            total_n += len(targets)

        train_loss = total_loss / total_n
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "Epoch": epoch,
                "Train_MSE": train_loss,
                "Validation_MSE": val_loss,
                "Train_RMSE": float(np.sqrt(train_loss)),
                "Validation_RMSE": float(np.sqrt(val_loss)),
                "Learning_rate": learning_rate,
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | train RMSE={np.sqrt(train_loss):.5f} "
                f"| val RMSE={np.sqrt(val_loss):.5f} | lr={learning_rate:.2e}"
            )

        if val_loss < best_val_loss - 1e-10:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"[Early stopping] Stopped at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_val_loss


def predict(model, X, batch_size, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    outputs = []
    model.eval()
    with torch.no_grad():
        for (inputs,) in loader:
            outputs.append(model(inputs.to(device)).cpu().numpy())
    return np.concatenate(outputs).ravel()


def safe_correlation(function, y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(function(y_true, y_pred)[0])


def calculate_metrics(split_name, y_true, y_pred, threshold):
    """Calculate regression plus threshold-based classification metrics."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    true_binary = (y_true >= threshold).astype(int)
    pred_binary = (y_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        true_binary, pred_binary, labels=[0, 1]
    ).ravel()
    auc = (
        float(roc_auc_score(true_binary, y_pred))
        if np.unique(true_binary).size == 2
        else np.nan
    )
    return {
        "Dataset": split_name,
        "N": int(len(y_true)),
        "AUC": auc,
        "PPV_Precision": float(
            precision_score(true_binary, pred_binary, zero_division=0)
        ),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "Pearson": safe_correlation(pearsonr, y_true, y_pred),
        "Spearman": safe_correlation(spearmanr, y_true, y_pred),
        "Sensitivity_Recall": float(
            recall_score(true_binary, pred_binary, zero_division=0)
        ),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Accuracy": float(accuracy_score(true_binary, pred_binary)),
        "F1": float(f1_score(true_binary, pred_binary, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Classification_threshold": float(threshold),
    }


def make_prediction_table(split_name, genes, y_true, y_pred, threshold):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return pd.DataFrame(
        {
            "Dataset": split_name,
            "gene": np.asarray(genes),
            "y_true_TE": y_true,
            "y_pred_TE": y_pred,
            "true_class": (y_true >= threshold).astype(int),
            "predicted_class": (y_pred >= threshold).astype(int),
        }
    )


def save_plots(history_df, y_test, pred_test, threshold, result_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(history_df["Epoch"], history_df["Train_MSE"], label="train")
    axes[0].plot(
        history_df["Epoch"], history_df["Validation_MSE"], label="validation"
    )
    axes[0].set(title="MSE loss", xlabel="Epoch", ylabel="MSE")
    axes[0].legend()
    axes[1].plot(history_df["Epoch"], history_df["Train_RMSE"], label="train")
    axes[1].plot(
        history_df["Epoch"], history_df["Validation_RMSE"], label="validation"
    )
    axes[1].set(title="RMSE", xlabel="Epoch", ylabel="RMSE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "training_history.png"), dpi=220)
    plt.close(fig)

    true_binary = (np.asarray(y_test) >= threshold).astype(int)
    if np.unique(true_binary).size == 2:
        fpr, tpr, _ = roc_curve(true_binary, np.asarray(pred_test).ravel())
        auc = roc_auc_score(true_binary, np.asarray(pred_test).ravel())
        plt.figure(figsize=(5.5, 5))
        plt.plot(fpr, tpr, label=f"PyTorch 1D CNN (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Test ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "roc_curve_test.png"), dpi=220)
        plt.close()


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    set_reproducible_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device} | PyTorch {torch.__version__}")

    X, y, genes = load_and_align(args.file, args.features_sheet, args.target_sheet)
    train_set, val_set, test_set = split_data(X, y, genes, args.threshold)
    X_train_raw, y_train, genes_train = train_set
    X_val_raw, y_val, genes_val = val_set
    X_test_raw, y_test, genes_test = test_set

    X_train, X_val, X_test, preprocessing = fit_preprocessor(
        X_train_raw, X_val_raw, X_test_raw
    )
    used_features = preprocessing["used_features"]
    pd.Series(used_features, name="used_feature").to_csv(
        os.path.join(args.result_dir, "used_features.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(preprocessing, os.path.join(args.result_dir, "preprocessing.joblib"))

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    model = CNN1DRegressor().to(device)
    model, history_df, best_val_loss = train_model(
        model, train_loader, val_loader, args.epochs, device
    )

    model_path = os.path.join(args.result_dir, "pytorch_cnn1d_te_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "CNN1DRegressor",
            "used_features": used_features,
            "classification_threshold": args.threshold,
            "pytorch_version": torch.__version__,
        },
        model_path,
    )

    predictions = {
        "Train": predict(model, X_train, args.batch_size, device),
        "Validation": predict(model, X_val, args.batch_size, device),
        "Test": predict(model, X_test, args.batch_size, device),
    }
    metrics_df = pd.DataFrame(
        [
            calculate_metrics("Train", y_train, predictions["Train"], args.threshold),
            calculate_metrics(
                "Validation", y_val, predictions["Validation"], args.threshold
            ),
            calculate_metrics("Test", y_test, predictions["Test"], args.threshold),
        ]
    )
    prediction_df = pd.concat(
        [
            make_prediction_table(
                "Train", genes_train, y_train, predictions["Train"], args.threshold
            ),
            make_prediction_table(
                "Validation",
                genes_val,
                y_val,
                predictions["Validation"],
                args.threshold,
            ),
            make_prediction_table(
                "Test", genes_test, y_test, predictions["Test"], args.threshold
            ),
        ],
        ignore_index=True,
    )

    metrics_df.to_csv(
        os.path.join(args.result_dir, "metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    prediction_df.to_csv(
        os.path.join(args.result_dir, "predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    history_df.to_csv(
        os.path.join(args.result_dir, "training_history.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    with pd.ExcelWriter(
        os.path.join(args.result_dir, "metrics_summary.xlsx"), engine="openpyxl"
    ) as writer:
        metrics_df.to_excel(writer, sheet_name="All_metrics", index=False)
        prediction_df.to_excel(writer, sheet_name="Predictions", index=False)
        history_df.to_excel(writer, sheet_name="Training_history", index=False)

    config = {
        "input_file": os.path.abspath(args.file),
        "feature_sheet": args.features_sheet,
        "target_sheet": args.target_sheet,
        "used_features": used_features,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "random_state": RANDOM_STATE,
        "classification_threshold": args.threshold,
        "test_size": TEST_SIZE,
        "validation_size_within_train": VALIDATION_SIZE_WITHIN_TRAIN,
        "epochs_requested": args.epochs,
        "epochs_completed": int(len(history_df)),
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_validation_rmse": float(np.sqrt(best_val_loss)),
    }
    with open(
        os.path.join(args.result_dir, "run_config.json"), "w", encoding="utf-8"
    ) as file_handle:
        json.dump(config, file_handle, indent=2, ensure_ascii=False)

    save_plots(history_df, y_test, predictions["Test"], args.threshold, args.result_dir)
    print("\n[OK] Training completed. Metrics summary:")
    print(metrics_df.to_string(index=False))
    print("[OK] Results saved to:", os.path.abspath(args.result_dir))


if __name__ == "__main__":
    main()
