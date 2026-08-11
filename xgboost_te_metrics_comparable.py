# -*- coding: utf-8 -*-
"""
XGBoost regression model directly comparable with pytorch_cnn1d_te_metrics.py.

The script uses the same input sheets, selected features, random seed,
train/validation/test split, and classification threshold as the PyTorch code.
It performs random-search hyperparameter tuning with K-fold CV on TRAINING DATA
ONLY, then uses the validation set for early stopping. The test set is used only
once for final evaluation.

Install:
    pip install xgboost numpy pandas scipy scikit-learn matplotlib openpyxl joblib

Run:
    python xgboost_te_metrics_comparable.py --file "C:/path/assembled_features.xlsx"

For a quick trial:
    python xgboost_te_metrics_comparable.py --file "C:/path/data.xlsx" --trials 5

Main outputs in RESULT_DIR:
    metrics_summary.xlsx / metrics_summary.csv
    predictions.xlsx / predictions.csv
    tuning_trials.csv / tuning_curve.png
    feature_importance.csv / feature_importance.png
    roc_curve_test.png
    xgboost_te_model.json
    preprocessing.joblib
    best_params.json / run_config.json
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
FILE = r"C:/Users/Administrator/Desktop/xgboost-new/data/assembled_features.xlsx"
SHEET_FEATURES = "filtered_tezheng"
SHEET_TARGET = "filtered_TE"
RESULT_DIR = "xgboost_te_comparable_results"

# Identical to the active feature list in the PyTorch script.
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

N_SPLITS = 5
N_TRIALS = 40
EARLY_STOPPING_ROUNDS = 200
TREE_METHOD = "hist"  # CPU-compatible and appropriate for the current setup.
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Comparable XGBoost TE model")
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
    """Exactly the same splitting procedure as the PyTorch script."""
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
    """Median imputation fitted on training data only; no scaling is needed."""
    input_features = list(X_train.columns)
    imputer_features = X_train.columns[~X_train.isna().all()].tolist()
    if not imputer_features:
        raise ValueError("All selected features are completely missing in training data.")

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
    used_features = train_imputed.columns[
        train_imputed.nunique(dropna=False) > 1
    ].tolist()
    if not used_features:
        raise ValueError("All selected features are constant in the training set.")

    preprocessing = {
        "imputer": imputer,
        "input_features": input_features,
        "imputer_features": imputer_features,
        "used_features": used_features,
    }
    return (
        train_imputed[used_features],
        val_imputed[used_features],
        test_imputed[used_features],
        preprocessing,
    )


def sample_params(rng):
    """Random-search ranges adapted from the user's original XGBoost code."""
    return {
        "learning_rate": float(rng.uniform(0.01, 0.20)),
        "max_depth": int(rng.integers(2, 9)),
        "min_child_weight": float(rng.uniform(1.0, 10.0)),
        "subsample": float(rng.uniform(0.55, 1.0)),
        "colsample_bytree": float(rng.uniform(0.55, 1.0)),
        "reg_alpha": float(rng.uniform(0.0, 1.0)),
        "reg_lambda": float(rng.uniform(0.0, 5.0)),
        "gamma": float(rng.uniform(0.0, 1.0)),
        "max_bin": int(rng.integers(128, 513)),
        "n_estimators": int(rng.integers(800, 4001)),
    }


def base_xgb_params(seed):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": TREE_METHOD,
        "nthread": -1,
        "seed": seed,
    }


def best_iteration_count(booster, fallback):
    """Return the number of trees to use (best_iteration is zero-based)."""
    best = getattr(booster, "best_iteration", None)
    return int(best + 1) if best is not None else int(fallback)


def predict_best(booster, dmatrix, n_trees):
    return booster.predict(dmatrix, iteration_range=(0, int(n_trees)))


def cv_score(params, X, y, n_splits, seed):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rmses = []
    best_tree_counts = []
    num_boost_round = params["n_estimators"]
    train_params = base_xgb_params(seed)
    train_params.update({k: v for k, v in params.items() if k != "n_estimators"})

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
        dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
        booster = xgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        n_trees = best_iteration_count(booster, num_boost_round)
        prediction = predict_best(booster, dval, n_trees)
        fold_rmse = float(
            np.sqrt(mean_squared_error(y.iloc[val_idx], prediction))
        )
        fold_rmses.append(fold_rmse)
        best_tree_counts.append(n_trees)
        print(f"    Fold {fold}: RMSE={fold_rmse:.5f}, trees={n_trees}")

    return (
        float(np.mean(fold_rmses)),
        float(np.std(fold_rmses)),
        best_tree_counts,
    )


def tune_hyperparameters(X_train, y_train, n_trials, n_splits, seed):
    if n_trials < 1:
        raise ValueError("--trials must be at least 1.")
    if n_splits < 2 or n_splits > len(X_train):
        raise ValueError("--cv-folds must be between 2 and the training sample count.")

    rng = np.random.default_rng(seed)
    records = []
    best_params = None
    best_rmse = float("inf")
    best_tree_counts = None

    for trial in range(1, n_trials + 1):
        params = sample_params(rng)
        mean_rmse, std_rmse, tree_counts = cv_score(
            params, X_train, y_train, n_splits, seed
        )
        records.append(
            {
                "Trial": trial,
                "CV_RMSE_mean": mean_rmse,
                "CV_RMSE_std": std_rmse,
                "CV_best_trees_mean": float(np.mean(tree_counts)),
                **params,
            }
        )
        print(
            f"[Tuning {trial:02d}/{n_trials}] "
            f"CV RMSE={mean_rmse:.5f} +/- {std_rmse:.5f}"
        )
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params.copy()
            best_tree_counts = tree_counts

    trials_df = pd.DataFrame(records).sort_values("CV_RMSE_mean").reset_index(drop=True)
    return best_params, best_tree_counts, trials_df


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    num_boost_round = best_params["n_estimators"]
    params = base_xgb_params(RANDOM_STATE)
    params.update({k: v for k, v in best_params.items() if k != "n_estimators"})
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )
    n_trees = best_iteration_count(booster, num_boost_round)
    return booster, n_trees


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


def feature_importance_table(booster, features):
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    table = pd.DataFrame(
        {
            "Feature": features,
            "Gain": [gain.get(name, 0.0) for name in features],
            "Weight": [weight.get(name, 0.0) for name in features],
            "Cover": [cover.get(name, 0.0) for name in features],
        }
    )
    for column in ["Gain", "Weight", "Cover"]:
        total = table[column].sum()
        if total > 0:
            table[column] = table[column] / total
    return table.sort_values("Gain", ascending=False).reset_index(drop=True)


def save_plots(trials_df, importance_df, y_test, pred_test, threshold, result_dir):
    ordered = trials_df.sort_values("Trial")
    plt.figure(figsize=(7, 4.5))
    plt.plot(ordered["Trial"], ordered["CV_RMSE_mean"], marker="o", markersize=3)
    plt.xlabel("Random-search trial")
    plt.ylabel("Mean CV RMSE")
    plt.title("XGBoost hyperparameter tuning")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "tuning_curve.png"), dpi=220)
    plt.close()

    plot_data = importance_df.sort_values("Gain")
    plt.figure(figsize=(8, max(4, 0.45 * len(plot_data) + 1.5)))
    plt.barh(plot_data["Feature"], plot_data["Gain"])
    plt.xlabel("Normalized gain")
    plt.title("XGBoost feature importance")
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
        plt.title("Test ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "roc_curve_test.png"), dpi=220)
        plt.close()


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    set_seed(RANDOM_STATE)
    print(f"[XGBoost] version={xgb.__version__} | tree_method={TREE_METHOD}")

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

    print(
        f"[Data] Train={len(y_train)}, Validation={len(y_val)}, Test={len(y_test)}"
    )
    print(f"[Tuning] {args.trials} trials x {args.cv_folds} folds")
    best_params, cv_best_trees, trials_df = tune_hyperparameters(
        X_train, y_train, args.trials, args.cv_folds, RANDOM_STATE
    )
    trials_df.to_csv(
        os.path.join(args.result_dir, "tuning_trials.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    booster, best_n_trees = train_final_model(
        X_train, y_train, X_val, y_val, best_params
    )
    booster.save_model(os.path.join(args.result_dir, "xgboost_te_model.json"))

    data_matrices = {
        "Train": xgb.DMatrix(X_train),
        "Validation": xgb.DMatrix(X_val),
        "Test": xgb.DMatrix(X_test),
    }
    predictions = {
        name: predict_best(booster, matrix, best_n_trees)
        for name, matrix in data_matrices.items()
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
    ) as file_handle:
        json.dump(serializable_params, file_handle, indent=2, ensure_ascii=False)

    config = {
        "input_file": os.path.abspath(args.file),
        "feature_sheet": args.features_sheet,
        "target_sheet": args.target_sheet,
        "used_features": used_features,
        "xgboost_version": xgb.__version__,
        "tree_method": TREE_METHOD,
        "random_state": RANDOM_STATE,
        "classification_threshold": args.threshold,
        "test_size": TEST_SIZE,
        "validation_size_within_train": VALIDATION_SIZE_WITHIN_TRAIN,
        "n_trials": args.trials,
        "cv_folds": args.cv_folds,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "cv_best_tree_counts": [int(value) for value in cv_best_trees],
        "final_best_n_trees": int(best_n_trees),
    }
    with open(
        os.path.join(args.result_dir, "run_config.json"), "w", encoding="utf-8"
    ) as file_handle:
        json.dump(config, file_handle, indent=2, ensure_ascii=False)

    save_plots(
        trials_df,
        importance_df,
        y_test,
        predictions["Test"],
        args.threshold,
        args.result_dir,
    )
    print("\n[OK] XGBoost training completed. Metrics summary:")
    print(metrics_df.to_string(index=False))
    print(f"[OK] Best number of trees: {best_n_trees}")
    print("[OK] Results saved to:", os.path.abspath(args.result_dir))


if __name__ == "__main__":
    main()
