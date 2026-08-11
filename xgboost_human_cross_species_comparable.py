# -*- coding: utf-8 -*-
"""
Human HEK293 XGBoost TE model for within- and cross-species comparison.

This version matches xgboost_te_metrics_comparable.py as closely as possible:
same semantic feature order, split procedure, seed, threshold, 40-trial/5-fold
search, original parameter ranges, native xgb.train behavior, early stopping,
direct booster.predict calls, and metric/output schema.

The original human workflow's 0 < TE < 10 filter and log1p target transform are
preserved as explicit configuration. Predictions and all reported metrics are
converted back to the original TE scale.

Install:
    pip install xgboost numpy pandas scipy scikit-learn matplotlib openpyxl joblib

Run:
    python xgboost_human_cross_species_comparable.py --file "C:/path/data.xlsx"
"""

import argparse
import json
import os
import random
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
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
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")


# ======================== User-editable configuration ========================
SPECIES_NAME = "Human_HEK293"
MODEL_NAME = "XGBoost"
FEATURE_SET_VERSION = "TE_6_features_v1"

FILE = (
    r"C:/Users/Administrator/Desktop/xgboost-new/data/"
    r"assemble_data_human_RNAgt5.xlsx"
)
SHEET_FEATURES = "filtered_tezheng"
SHEET_TARGET = "filtered_TE"
RESULT_DIR = "xgboost_human_cross_species_results"

# Source -> canonical mapping. Order matches the CNN and the previous species.
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

# Human-specific target handling retained from the uploaded script.
APPLY_TE_FILTER = True
TE_MIN_EXCLUSIVE = 0.0
TE_MAX_EXCLUSIVE = 10.0
USE_LOG1P_TARGET = True

N_SPLITS = 5
N_TRIALS = 40
EARLY_STOPPING_ROUNDS = 200
TREE_METHOD = "hist"
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Human HEK293 XGBoost model for cross-species comparison"
    )
    parser.add_argument("--file", default=FILE, help="Input Excel workbook")
    parser.add_argument("--features-sheet", default=SHEET_FEATURES)
    parser.add_argument("--target-sheet", default=SHEET_TARGET)
    parser.add_argument("--result-dir", default=RESULT_DIR)
    parser.add_argument("--threshold", type=float, default=CLASS_THRESHOLD)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--cv-folds", type=int, default=N_SPLITS)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_and_align(file, feature_sheet, target_sheet):
    """Align by gene and map source feature names to canonical names."""
    feature_df = pd.read_excel(file, sheet_name=feature_sheet, engine="openpyxl")
    target_df = pd.read_excel(file, sheet_name=target_sheet, engine="openpyxl")
    if feature_df.shape[1] < 2 or target_df.shape[1] < 2:
        raise ValueError("Both sheets must contain at least two columns.")

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
    missing = [feature for feature in source_features if feature not in aligned.columns]
    if missing:
        raise ValueError(
            "Cross-species comparison requires all six features. Missing: "
            + ", ".join(missing)
        )
    X = aligned[source_features].apply(pd.to_numeric, errors="coerce")
    X = X.rename(columns=FEATURE_MAPPING)[CANONICAL_FEATURES]
    y = aligned["TE"].astype(float)
    genes = aligned["gene"].astype(str)
    if len(y) < 10:
        raise ValueError("Too few aligned samples; at least 10 are required.")
    return X, y, genes, aligned


def safe_stratify(y, threshold):
    classes = (np.asarray(y) >= threshold).astype(int)
    counts = np.bincount(classes, minlength=2)
    return classes if np.all(counts >= 2) else None


def robust_train_test_split(indices, test_size, seed, strata):
    try:
        return train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=strata
        )
    except ValueError:
        print("[Warning] Stratified split was not possible; using a random split.")
        return train_test_split(indices, test_size=test_size, random_state=seed)


def split_data(X, y, genes, threshold):
    """Same 64%/16%/20% split procedure as the comparable CNN/XGBoost code."""
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
    """Apply the original human-data 0 < TE < 10 filter before splitting."""
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
    """Convert raw TE to the scale used to train XGBoost."""
    values = np.asarray(y, dtype=float)
    return np.log1p(values) if USE_LOG1P_TARGET else values


def inverse_target(prediction):
    """Return model predictions to the original TE scale."""
    values = np.asarray(prediction, dtype=float)
    return np.expm1(values) if USE_LOG1P_TARGET else values


def fit_preprocessor(X_train, X_val, X_test):
    """Fit median imputation on training data only; XGBoost needs no scaling."""
    all_missing = X_train.columns[X_train.isna().all()].tolist()
    if all_missing:
        raise ValueError(
            "These required features are completely missing in training data: "
            + ", ".join(all_missing)
        )
    imputer = SimpleImputer(strategy="median")
    train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=CANONICAL_FEATURES
    )
    val_imputed = pd.DataFrame(imputer.transform(X_val), columns=CANONICAL_FEATURES)
    test_imputed = pd.DataFrame(
        imputer.transform(X_test), columns=CANONICAL_FEATURES
    )
    used_features = train_imputed.columns[
        train_imputed.nunique(dropna=False) > 1
    ].tolist()
    if not used_features:
        raise ValueError("All six features are constant in the training data.")
    dropped = [feature for feature in CANONICAL_FEATURES if feature not in used_features]
    if dropped:
        print("[Warning] Constant features removed:", ", ".join(dropped))
    preprocessing = {
        "imputer": imputer,
        "source_to_canonical_mapping": FEATURE_MAPPING,
        "canonical_feature_order": CANONICAL_FEATURES,
        "used_features": used_features,
    }
    return (
        train_imputed[used_features],
        val_imputed[used_features],
        test_imputed[used_features],
        preprocessing,
    )


def sample_params(rng):
    """Same parameter ranges and RNG call order as the previous XGBoost code."""
    return {
        "learning_rate": float(rng.uniform(0.01, 0.20)),
        "max_depth": int(rng.integers(3, 11)),
        "min_child_weight": int(rng.integers(1, 10)),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.5, 1.0)),
        "colsample_bynode": float(rng.uniform(0.5, 1.0)),
        "reg_alpha": float(rng.uniform(0.0, 1.0)),
        "reg_lambda": float(rng.uniform(0.0, 5.0)),
        "max_bin": int(rng.integers(128, 513)),
        "n_estimators": int(rng.integers(1500, 7001)),
    }


def base_xgb_params(seed):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": TREE_METHOD,
        "nthread": -1,
        "seed": seed,
    }


def original_best_iteration(booster):
    """Record zero-based best_iteration exactly as in the earlier code."""
    return int(getattr(booster, "best_iteration", len(booster.get_dump())))


def predict_like_original(booster, dmatrix):
    """Direct predict call, intentionally matching the earlier XGBoost code."""
    return booster.predict(dmatrix)


def cv_score(params, X, y, n_splits, seed):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rmses = []
    best_iterations = []
    native_params = base_xgb_params(seed)
    # Preserve earlier behavior: n_estimators is retained in this dictionary.
    native_params.update(params)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
        dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
        booster = xgb.train(
            native_params,
            dtrain,
            num_boost_round=params.get("n_estimators", 4000),
            evals=[(dval, "valid")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        prediction = predict_like_original(booster, dval)
        fold_rmse = float(np.sqrt(mean_squared_error(y.iloc[val_idx], prediction)))
        best_iteration = original_best_iteration(booster)
        fold_rmses.append(fold_rmse)
        best_iterations.append(best_iteration)
        print(
            f"[CV] Fold {fold}: RMSE={fold_rmse:.4f}, "
            f"best_iter={best_iteration}"
        )
    return float(np.mean(fold_rmses)), float(np.std(fold_rmses)), best_iterations


def tune_hyperparameters(X_train, y_train, n_trials, n_splits, seed):
    if n_trials < 1:
        raise ValueError("--trials must be at least 1.")
    if n_splits < 2 or n_splits > len(X_train):
        raise ValueError("--cv-folds must be between 2 and the training sample count.")
    rng = np.random.default_rng(seed)
    records = []
    best = {"rmse_mean": float("inf")}
    for trial in range(1, n_trials + 1):
        params = sample_params(rng)
        mean_rmse, std_rmse, iterations = cv_score(
            params, X_train, y_train, n_splits, seed
        )
        records.append(
            {
                "Trial": trial,
                "CV_RMSE_mean": mean_rmse,
                "CV_RMSE_std": std_rmse,
                "CV_best_iteration_mean_zero_based": float(np.mean(iterations)),
                **params,
            }
        )
        print(
            f"[TUNE] Trial {trial:02d}: "
            f"RMSE={mean_rmse:.4f} +/- {std_rmse:.4f}"
        )
        if mean_rmse < best["rmse_mean"]:
            best = {
                "rmse_mean": mean_rmse,
                "params": params.copy(),
                "iterations": iterations,
            }
    trials_df = pd.DataFrame(records).sort_values("CV_RMSE_mean").reset_index(drop=True)
    return best["params"], best["iterations"], trials_df


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    native_params = base_xgb_params(RANDOM_STATE)
    native_params.update(best_params)
    booster = xgb.train(
        native_params,
        dtrain,
        num_boost_round=best_params.get("n_estimators", 5000),
        evals=[(dval, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    return booster, original_best_iteration(booster)


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
    sensitivity = float(recall_score(true_binary, pred_binary, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
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
        "Sensitivity_Recall": sensitivity,
        "Specificity": specificity,
        "Balanced_accuracy": float(0.5 * (sensitivity + specificity)),
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


def feature_importance_table(booster, features):
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    table = pd.DataFrame(
        {
            "Feature": features,
            "Gain": [gain.get(feature, 0.0) for feature in features],
            "Weight": [weight.get(feature, 0.0) for feature in features],
            "Cover": [cover.get(feature, 0.0) for feature in features],
        }
    )
    for column in ["Gain", "Weight", "Cover"]:
        total = table[column].sum()
        if total > 0:
            table[column] = table[column] / total
    return table.sort_values("Gain", ascending=False).reset_index(drop=True)


def save_plots(trials, importance, y_test, pred_test, threshold, result_dir):
    ordered = trials.sort_values("Trial")
    plt.figure(figsize=(7, 4.5))
    plt.plot(ordered["Trial"], ordered["CV_RMSE_mean"], marker="o", markersize=3)
    plt.xlabel("Random-search trial")
    plt.ylabel("Mean CV RMSE")
    plt.title(f"{SPECIES_NAME}: XGBoost tuning")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "tuning_curve.png"), dpi=220)
    plt.close()

    plot_data = importance.sort_values("Gain")
    plt.figure(figsize=(8, max(4, 0.45 * len(plot_data) + 1.5)))
    plt.barh(plot_data["Feature"], plot_data["Gain"])
    plt.xlabel("Normalized gain")
    plt.title(f"{SPECIES_NAME}: XGBoost feature importance")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "feature_importance.png"), dpi=220)
    plt.close()

    true_binary = (np.asarray(y_test) >= threshold).astype(int)
    if np.unique(true_binary).size == 2:
        fpr, tpr, _ = roc_curve(true_binary, pred_test)
        auc = roc_auc_score(true_binary, pred_test)
        plt.figure(figsize=(5.5, 5))
        plt.plot(fpr, tpr, label=f"XGBoost (AUC={auc:.3f})")
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
    set_seed(RANDOM_STATE)
    print(f"[Species] {SPECIES_NAME}")
    print(f"[XGBoost] version={xgb.__version__} | tree_method={TREE_METHOD}")
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
    used_features = preprocessing["used_features"]
    y_train_model = transform_target(y_train_raw)
    y_val_model = transform_target(y_val_raw)

    mapping_df = pd.DataFrame(
        {
            "Position": np.arange(1, len(CANONICAL_FEATURES) + 1),
            "Source_feature": list(FEATURE_MAPPING.keys()),
            "Canonical_feature": CANONICAL_FEATURES,
        }
    )
    mapping_df.to_csv(
        os.path.join(args.result_dir, "feature_mapping_and_order.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.Series(used_features, name="used_feature").to_csv(
        os.path.join(args.result_dir, "used_features.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(preprocessing, os.path.join(args.result_dir, "preprocessing.joblib"))

    print(
        f"[Data] Train={len(y_train_raw)}, Validation={len(y_val_raw)}, "
        f"Test={len(y_test_raw)}"
    )
    print(f"[Target transform] {'log1p' if USE_LOG1P_TARGET else 'raw TE'}")
    print(f"[Tuning] {args.trials} trials x {args.cv_folds} folds")
    best_params, cv_best_iterations, trials_df = tune_hyperparameters(
        X_train, pd.Series(y_train_model), args.trials, args.cv_folds, RANDOM_STATE
    )
    booster, final_best_iteration = train_final_model(
        X_train, y_train_model, X_val, y_val_model, best_params
    )
    booster.save_model(os.path.join(args.result_dir, "xgboost_human_model.json"))

    matrices = {
        "Train": xgb.DMatrix(X_train),
        "Validation": xgb.DMatrix(X_val),
        "Test": xgb.DMatrix(X_test),
    }
    model_scale_predictions = {
        name: predict_like_original(booster, matrix)
        for name, matrix in matrices.items()
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
    importance_df = feature_importance_table(booster, used_features)

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
    trials_df.to_csv(
        os.path.join(args.result_dir, "tuning_trials.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    importance_df.to_csv(
        os.path.join(args.result_dir, "feature_importance.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    with pd.ExcelWriter(
        os.path.join(args.result_dir, "metrics_summary.xlsx"), engine="openpyxl"
    ) as writer:
        metrics_df.to_excel(writer, sheet_name="All_metrics", index=False)
        prediction_df.to_excel(writer, sheet_name="Predictions", index=False)
        importance_df.to_excel(writer, sheet_name="Feature_importance", index=False)
        trials_df.to_excel(writer, sheet_name="Tuning_trials", index=False)

    serializable_params = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in best_params.items()
    }
    with open(
        os.path.join(args.result_dir, "best_params.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(serializable_params, handle, indent=2, ensure_ascii=False)

    config = {
        "species": SPECIES_NAME,
        "model": MODEL_NAME,
        "feature_set": FEATURE_SET_VERSION,
        "input_file": os.path.abspath(args.file),
        "feature_sheet": args.features_sheet,
        "target_sheet": args.target_sheet,
        "source_to_canonical_mapping": FEATURE_MAPPING,
        "canonical_feature_order": CANONICAL_FEATURES,
        "used_features": used_features,
        "aligned_sample_count_before_filter": int(aligned_sample_count_before_filter),
        "sample_count_after_filter": int(len(y_raw)),
        "train_sample_count": int(len(y_train_raw)),
        "validation_sample_count": int(len(y_val_raw)),
        "test_sample_count": int(len(y_test_raw)),
        "xgboost_version": xgb.__version__,
        "tree_method": TREE_METHOD,
        "random_state": RANDOM_STATE,
        "classification_threshold": args.threshold,
        "apply_te_filter": APPLY_TE_FILTER,
        "te_min_exclusive": TE_MIN_EXCLUSIVE,
        "te_max_exclusive": TE_MAX_EXCLUSIVE,
        "target_transform": "log1p" if USE_LOG1P_TARGET else "none",
        "metrics_scale": "original_TE",
        "test_size": TEST_SIZE,
        "validation_size_within_train": VALIDATION_SIZE_WITHIN_TRAIN,
        "n_trials": args.trials,
        "cv_folds": args.cv_folds,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "cv_best_iterations_zero_based": [int(value) for value in cv_best_iterations],
        "final_best_iteration_zero_based": int(final_best_iteration),
    }
    with open(
        os.path.join(args.result_dir, "run_config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    save_plots(
        trials_df,
        importance_df,
        y_test_raw,
        predictions["Test"],
        args.threshold,
        args.result_dir,
    )
    print("\n[OK] XGBoost training completed. Cross-species-ready metrics:")
    print(metrics_df.to_string(index=False))
    print(f"[OK] Best iteration (zero-based): {final_best_iteration}")
    print("[OK] Results saved to:", os.path.abspath(args.result_dir))


if __name__ == "__main__":
    main()
