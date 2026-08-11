# -*- coding: utf-8 -*-
"""
PyTorch 1D CNN for Human HEK293 TE prediction and cross-species comparison.

This script intentionally matches the earlier PyTorch 1D CNN workflow:
  * same semantic six-feature order
  * same 64%/16%/20% train/validation/test split procedure
  * same random seed, regression architecture, optimizer and early stopping
  * same TE >= 0.5 classification rule and quantitative metrics

The Human HEK293 source names ``kozak`` and ``cds_length`` are renamed internally
to the cross-species canonical names ``kozak_flag`` and ``len_cds``. This is
important because Conv1D is sensitive to feature order.

The original human workflow's 0 < TE < 10 filter and log1p target transform are
preserved as explicit settings. Predictions and reported metrics are converted
back to the original TE scale.

Compatible with PyTorch 2.0.1+cpu.

Install:
    pip install torch numpy pandas scipy scikit-learn matplotlib openpyxl joblib

Run:
    python pytorch_cnn1d_human_cross_species.py --file "C:/path/data.xlsx"
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
SPECIES_NAME = "Human_HEK293"
MODEL_NAME = "PyTorch_1D_CNN"
FEATURE_SET_VERSION = "TE_6_features_v1"

FILE = (
    r"C:/Users/Administrator/Desktop/xgboost-new/data"
    r"assemble_data_human_RNAgt5.xlsx"
)
SHEET_FEATURES = "filtered_tezheng"
SHEET_TARGET = "filtered_TE"
RESULT_DIR = "pytorch_cnn1d_human_cross_species_results"

# Source-column -> canonical-column mapping. Dictionary order defines CNN order.
# This canonical order is identical to the earlier PyTorch CNN script.
FEATURE_MAPPING = {
    "stem_ratio_5utr": "stem_ratio_5utr",
    "overall_gc_5utr": "overall_gc_5utr",
    "uorf": "uorf",
    "kozak": "kozak_flag",
    "cds_length": "len_cds",
    "overall_gc_cds": "overall_gc_cds",
}
CANONICAL_FEATURES = list(FEATURE_MAPPING.values())

TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.20
RANDOM_STATE = 42
CLASS_THRESHOLD = 0.50

# Human-specific target handling, identical to the human XGBoost version.
APPLY_TE_FILTER = True
TE_MIN_EXCLUSIVE = 0.0
TE_MAX_EXCLUSIVE = 10.0
USE_LOG1P_TARGET = True

EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 40
REDUCE_LR_PATIENCE = 12
NUM_WORKERS = 0
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Human HEK293 PyTorch 1D CNN for cross-species comparison"
    )
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
    """Load sheets, align by gene, and standardize feature names and order."""
    feature_df = pd.read_excel(file, sheet_name=feature_sheet, engine="openpyxl")
    target_df = pd.read_excel(file, sheet_name=target_sheet, engine="openpyxl")
    if feature_df.shape[1] < 2:
        raise ValueError("The feature sheet must contain a gene column and features.")
    if target_df.shape[1] < 2:
        raise ValueError("The target sheet must contain gene and TE columns.")

    feature_df = feature_df.rename(columns={feature_df.columns[0]: "gene"})
    target_df = target_df.rename(
        columns={target_df.columns[0]: "gene", target_df.columns[1]: "TE"}
    )
    feature_df = feature_df.drop_duplicates(subset="gene", keep="first")
    target_df = target_df.drop_duplicates(subset="gene", keep="first")
    aligned = pd.merge(
        feature_df,
        target_df[["gene", "TE"]],
        on="gene",
        how="inner",
        validate="one_to_one",
    )
    aligned["TE"] = pd.to_numeric(aligned["TE"], errors="coerce")
    aligned = aligned.dropna(subset=["TE"]).reset_index(drop=True)

    source_features = list(FEATURE_MAPPING.keys())
    missing = [name for name in source_features if name not in aligned.columns]
    if missing:
        raise ValueError(
            "Cross-species comparison requires all six features. Missing: "
            + ", ".join(missing)
        )

    X = aligned[source_features].apply(pd.to_numeric, errors="coerce")
    X = X.rename(columns=FEATURE_MAPPING)
    X = X[CANONICAL_FEATURES]
    y = aligned["TE"].astype(float)
    genes = aligned["gene"].astype(str)
    if len(aligned) < 10:
        raise ValueError("Too few aligned samples. At least 10 samples are required.")
    return X, y, genes, aligned


def safe_stratify(y, threshold):
    classes = (np.asarray(y) >= threshold).astype(int)
    counts = np.bincount(classes, minlength=2)
    return classes if np.all(counts >= 2) else None


def robust_train_test_split(indices, test_size, seed, strata):
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

    def take(index):
        return (
            X.iloc[index].reset_index(drop=True),
            y.iloc[index].reset_index(drop=True),
            genes.iloc[index].reset_index(drop=True),
        )

    return take(train_idx), take(val_idx), take(test_idx)


def filter_target_range(X, y, genes):
    """Apply the human-data TE filter before splitting."""
    if not APPLY_TE_FILTER:
        return X.reset_index(drop=True), y.reset_index(drop=True), genes.reset_index(drop=True)
    mask = (y > TE_MIN_EXCLUSIVE) & (y < TE_MAX_EXCLUSIVE)
    print(
        f"[Target filter] {TE_MIN_EXCLUSIVE} < TE < {TE_MAX_EXCLUSIVE}: "
        f"{len(y)} -> {int(mask.sum())} samples"
    )
    if int(mask.sum()) < 10:
        raise ValueError("Too few samples remain after TE filtering.")
    return (
        X.loc[mask].reset_index(drop=True),
        y.loc[mask].reset_index(drop=True),
        genes.loc[mask].reset_index(drop=True),
    )


def transform_target(y):
    values = np.asarray(y, dtype=np.float32)
    return np.log1p(values) if USE_LOG1P_TARGET else values


def inverse_target(prediction):
    values = np.asarray(prediction, dtype=float)
    return np.expm1(values) if USE_LOG1P_TARGET else values


def fit_preprocessor(X_train, X_val, X_test):
    """Fit median imputation and standardization on training data only."""
    all_missing = X_train.columns[X_train.isna().all()].tolist()
    if all_missing:
        raise ValueError(
            "Cross-species comparison cannot retain all six features because "
            "these are completely missing in training data: " + ", ".join(all_missing)
        )

    imputer = SimpleImputer(strategy="median")
    train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=CANONICAL_FEATURES
    )
    val_imputed = pd.DataFrame(
        imputer.transform(X_val), columns=CANONICAL_FEATURES
    )
    test_imputed = pd.DataFrame(
        imputer.transform(X_test), columns=CANONICAL_FEATURES
    )

    constant_features = train_imputed.columns[
        train_imputed.nunique(dropna=False) <= 1
    ].tolist()
    if constant_features:
        print(
            "[Warning] Constant features are retained to preserve the identical "
            "cross-species CNN input: " + ", ".join(constant_features)
        )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    val_scaled = scaler.transform(val_imputed)
    test_scaled = scaler.transform(test_imputed)

    # PyTorch Conv1d shape: samples, channels, ordered feature positions.
    as_cnn = lambda values: np.asarray(values, dtype=np.float32)[:, np.newaxis, :]
    preprocessing = {
        "imputer": imputer,
        "scaler": scaler,
        "source_to_canonical_mapping": FEATURE_MAPPING,
        "canonical_feature_order": CANONICAL_FEATURES,
    }
    return as_cnn(train_scaled), as_cnn(val_scaled), as_cnn(test_scaled), preprocessing


class CNN1DRegressor(nn.Module):
    """The same architecture used in the earlier species-specific CNN."""

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

    def forward(self, inputs):
        return self.regressor(self.feature_extractor(inputs)).squeeze(1)


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
            loss = criterion(model(inputs), targets)
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
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    target_sd = float(np.std(y_true, ddof=0))
    return {
        "Species": SPECIES_NAME,
        "Model": MODEL_NAME,
        "Feature_set": FEATURE_SET_VERSION,
        "Target_training_scale": "log1p_TE" if USE_LOG1P_TARGET else "raw_TE",
        "TE_filter": (
            f"{TE_MIN_EXCLUSIVE}<TE<{TE_MAX_EXCLUSIVE}"
            if APPLY_TE_FILTER
            else "none"
        ),
        "Dataset": split_name,
        "N": int(len(y_true)),
        "Positive_rate": float(np.mean(true_binary)),
        "AUC": auc,
        "PPV_Precision": float(
            precision_score(true_binary, pred_binary, zero_division=0)
        ),
        "RMSE": rmse,
        "NRMSE_by_target_SD": float(rmse / target_sd) if target_sd > 0 else np.nan,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "Pearson": safe_correlation(pearsonr, y_true, y_pred),
        "Spearman": safe_correlation(spearmanr, y_true, y_pred),
        "Sensitivity_Recall": float(
            recall_score(true_binary, pred_binary, zero_division=0)
        ),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Balanced_accuracy": float(
            0.5
            * (
                recall_score(true_binary, pred_binary, zero_division=0)
                + (tn / (tn + fp) if (tn + fp) else np.nan)
            )
        ),
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
            "Species": SPECIES_NAME,
            "Model": MODEL_NAME,
            "Dataset": split_name,
            "gene": np.asarray(genes),
            "y_true_TE": y_true,
            "y_pred_TE": y_pred,
            "true_class": (y_true >= threshold).astype(int),
            "predicted_class": (y_pred >= threshold).astype(int),
        }
    )


def save_plots(history_df, y_test, prediction_test, threshold, result_dir):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
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
    figure.suptitle(f"{SPECIES_NAME}: PyTorch 1D CNN training")
    figure.tight_layout()
    figure.savefig(os.path.join(result_dir, "training_history.png"), dpi=220)
    plt.close(figure)

    true_binary = (np.asarray(y_test) >= threshold).astype(int)
    if np.unique(true_binary).size == 2:
        fpr, tpr, _ = roc_curve(true_binary, prediction_test)
        auc = roc_auc_score(true_binary, prediction_test)
        plt.figure(figsize=(5.5, 5))
        plt.plot(fpr, tpr, label=f"1D CNN (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"{SPECIES_NAME}: Test ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "roc_curve_test.png"), dpi=220)
        plt.close()


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    set_reproducible_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Species] {SPECIES_NAME}")
    print(f"[Device] {device} | PyTorch {torch.__version__}")
    print("[Canonical feature order]", ", ".join(CANONICAL_FEATURES))

    X, y_raw, genes, aligned = load_and_align(
        args.file, args.features_sheet, args.target_sheet
    )
    aligned_sample_count_before_filter = len(y_raw)
    X, y_raw, genes = filter_target_range(X, y_raw, genes)
    train_set, val_set, test_set = split_data(X, y_raw, genes, args.threshold)
    X_train_raw, y_train_raw, genes_train = train_set
    X_val_raw, y_val_raw, genes_val = val_set
    X_test_raw, y_test_raw, genes_test = test_set
    X_train, X_val, X_test, preprocessing = fit_preprocessor(
        X_train_raw, X_val_raw, X_test_raw
    )
    y_train_model = transform_target(y_train_raw)
    y_val_model = transform_target(y_val_raw)

    pd.DataFrame(
        {
            "Position": np.arange(1, len(CANONICAL_FEATURES) + 1),
            "Source_feature": list(FEATURE_MAPPING.keys()),
            "Canonical_feature": CANONICAL_FEATURES,
        }
    ).to_csv(
        os.path.join(args.result_dir, "feature_mapping_and_order.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(preprocessing, os.path.join(args.result_dir, "preprocessing.joblib"))

    print(
        f"[Data] Train={len(y_train_raw)}, Validation={len(y_val_raw)}, "
        f"Test={len(y_test_raw)}"
    )
    print(f"[Target transform] {'log1p' if USE_LOG1P_TARGET else 'raw TE'}")
    train_loader = make_loader(
        X_train, y_train_model, args.batch_size, shuffle=True
    )
    validation_loader = make_loader(
        X_val, y_val_model, args.batch_size, shuffle=False
    )
    model = CNN1DRegressor().to(device)
    model, history_df, best_val_loss = train_model(
        model, train_loader, validation_loader, args.epochs, device
    )

    model_path = os.path.join(args.result_dir, "pytorch_cnn1d_human_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "CNN1DRegressor",
            "species": SPECIES_NAME,
            "feature_set": FEATURE_SET_VERSION,
            "source_to_canonical_mapping": FEATURE_MAPPING,
            "canonical_feature_order": CANONICAL_FEATURES,
            "classification_threshold": args.threshold,
            "apply_te_filter": APPLY_TE_FILTER,
            "te_min_exclusive": TE_MIN_EXCLUSIVE,
            "te_max_exclusive": TE_MAX_EXCLUSIVE,
            "target_transform": "log1p" if USE_LOG1P_TARGET else "none",
            "pytorch_version": torch.__version__,
        },
        model_path,
    )

    model_scale_predictions = {
        "Train": predict(model, X_train, args.batch_size, device),
        "Validation": predict(model, X_val, args.batch_size, device),
        "Test": predict(model, X_test, args.batch_size, device),
    }
    predictions = {
        name: inverse_target(values)
        for name, values in model_scale_predictions.items()
    }
    metrics_df = pd.DataFrame(
        [
            calculate_metrics(
                "Train", y_train_raw, predictions["Train"], args.threshold
            ),
            calculate_metrics(
                "Validation", y_val_raw, predictions["Validation"], args.threshold
            ),
            calculate_metrics(
                "Test", y_test_raw, predictions["Test"], args.threshold
            ),
        ]
    )
    prediction_df = pd.concat(
        [
            make_prediction_table(
                "Train",
                genes_train,
                y_train_raw,
                predictions["Train"],
                args.threshold,
            ),
            make_prediction_table(
                "Validation",
                genes_val,
                y_val_raw,
                predictions["Validation"],
                args.threshold,
            ),
            make_prediction_table(
                "Test",
                genes_test,
                y_test_raw,
                predictions["Test"],
                args.threshold,
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
        "species": SPECIES_NAME,
        "model": MODEL_NAME,
        "feature_set": FEATURE_SET_VERSION,
        "input_file": os.path.abspath(args.file),
        "feature_sheet": args.features_sheet,
        "target_sheet": args.target_sheet,
        "source_to_canonical_mapping": FEATURE_MAPPING,
        "canonical_feature_order": CANONICAL_FEATURES,
        "aligned_sample_count_before_filter": int(aligned_sample_count_before_filter),
        "sample_count_after_filter": int(len(y_raw)),
        "train_sample_count": int(len(y_train_raw)),
        "validation_sample_count": int(len(y_val_raw)),
        "test_sample_count": int(len(y_test_raw)),
        "device": str(device),
        "pytorch_version": torch.__version__,
        "random_state": RANDOM_STATE,
        "classification_threshold": args.threshold,
        "apply_te_filter": APPLY_TE_FILTER,
        "te_min_exclusive": TE_MIN_EXCLUSIVE,
        "te_max_exclusive": TE_MAX_EXCLUSIVE,
        "target_transform": "log1p" if USE_LOG1P_TARGET else "none",
        "metrics_scale": "original_TE",
        "test_size": TEST_SIZE,
        "validation_size_within_train": VALIDATION_SIZE_WITHIN_TRAIN,
        "epochs_requested": args.epochs,
        "epochs_completed": int(len(history_df)),
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_validation_rmse_model_scale": float(np.sqrt(best_val_loss)),
    }
    with open(
        os.path.join(args.result_dir, "run_config.json"), "w", encoding="utf-8"
    ) as file_handle:
        json.dump(config, file_handle, indent=2, ensure_ascii=False)

    save_plots(
        history_df,
        y_test_raw,
        predictions["Test"],
        args.threshold,
        args.result_dir,
    )
    print("\n[OK] Training completed. Cross-species-ready metrics summary:")
    print(metrics_df.to_string(index=False))
    print("[OK] Results saved to:", os.path.abspath(args.result_dir))


if __name__ == "__main__":
    main()
