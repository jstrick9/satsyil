# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Metadata
# MAGIC - **Created by:** Drew McPherson
# MAGIC - **Created on:** 2026-07-14
# MAGIC - **Last updated by:** Drew McPherson
# MAGIC - **Last updated on:** 2026-07-14
# MAGIC
# MAGIC ## Changelog
# MAGIC - **2026-07-14 (Drew McPherson):** Initial notebook deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# DBTITLE 1,Import Libraries
# Import libraries
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
import mlflow
import uuid
from mlflow.models import infer_signature   
from mlflow.tracking import MlflowClient

# COMMAND ----------

# DBTITLE 1,Environmental Settings
dbutils.widgets.text("dbx_env", "dev")
dbx_env = dbutils.widgets.get("dbx_env")

config_file_name = "trmreports-conf.yaml"
config_file = f"../../config/{dbx_env}/{config_file_name}"


print(f"{config_file=}, {dbx_env=}")

# COMMAND ----------

# DBTITLE 1,Common Functions and Configs
# MAGIC %run ./../shared/ntb_common_func_and_params

# COMMAND ----------

# DBTITLE 1,Configuration Constants
common_configs = read_yaml(config_file)

# ────────────────────────────────────────────────────────────────────────────
# Configuration Constants
# ────────────────────────────────────────────────────────────────────────────

# Reporting catalog
reporting_catalog = common_configs["schema"]["reporting_catalog"]

# MLflow & Model Registry Configuration
TRAINING_TABLE = f"{reporting_catalog}.gold.case_outcome_features"

# Grid Search Configuration
ABN_GRID_SIZE = 200
MIN_PRECISION_PRIMARY = 0.925
MIN_RECALL_SECONDARY = 0.90
MIN_RECALL_TERTIARY = 0.95
MIN_ELIGIBLE_RECORDS = 50

# Model Hyperparameters
REG_PARAMS = {
    'iterations': 270,
    'depth': 8,
    'learning_rate': 0.148781,
    'l2_leaf_reg': 1.7357,
    'bagging_temperature': 0.1554,
    'random_strength': 6.9908,
    'border_count': 128,
    'loss_function': 'Logloss',
    'auto_class_weights': 'Balanced',
    'random_state': 42,
    'used_ram_limit': '12GB',
}

ABN_EXA_PARAMS = {
    'iterations': 300,
    'depth': 10,
    'learning_rate': 0.111390,
    'l2_leaf_reg': 9.8348,
    'bagging_temperature': 0.8485,
    'random_strength': 8.0429,
    'border_count': 64,
    'loss_function': 'Logloss',
    'auto_class_weights': 'Balanced',
    'random_state': 42,
    'used_ram_limit': '12GB',
}

NOA_PARAMS = {
    'iterations': 300,
    'depth': 10,
    'learning_rate': 0.111390,
    'l2_leaf_reg': 9.8348,
    'bagging_temperature': 0.8485,
    'random_strength': 8.0429,
    'border_count': 64,
    'loss_function': 'Logloss',
    'auto_class_weights': 'Balanced',
    'random_state': 42,
    'used_ram_limit': '12GB',
}

# Training Data Configuration
TRAINING_LOOKBACK_MONTHS = 72
TRAINING_WINDOW_MONTHS = 36

# MLflow Experiment
# Stable, path-independent name — runs log here regardless of where the notebook is deployed.
# NOTE: This value is also hardcoded in ntb_ml_registration_inference (Configuration cell).
# If you rename this experiment, update both notebooks.
MLFLOW_EXPERIMENT_NAME = "/Shared/ml_registration_prediction"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# DBTITLE 1,Model Settings
# ────────────────────────────────────────────────────────────────────────────
# Model Settings
# Target precision for threshold calibration across all three models.
# Increase to trade recall for precision; decrease to trade precision for recall.
# ────────────────────────────────────────────────────────────────────────────

# Create widgets for runtime configuration
dbutils.widgets.text("target_precision", "0.95", "Target Precision")
dbutils.widgets.text("target_recall", "0.95", "Target Recall")

TARGET_PRECISION = float(dbutils.widgets.get("target_precision") or 0.95)
TARGET_RECALL = float(dbutils.widgets.get("target_recall") or 0.95)
TRAINING_BATCH_ID = str(uuid.uuid4()) # Creates a unique ID to tag all 4 MLflow runs (collectively)
print(f"Target precision  : {TARGET_PRECISION:.0%}")
print(f"Target recall     : {TARGET_RECALL:.0%}")
print(f"Training batch ID : {TRAINING_BATCH_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

# DBTITLE 1,Model Training & Evaluation Functions
import builtins
from catboost import Pool

# Since we are training 3 models Registration (reg), Notice of Allowance (noa), and Examination Abandonments (reg_exa), it is convenient move the training and evaluation cells into functions rather than repeating them three times. Model-specific hyperparameters are defined in the notebook's setup cells.

# Train the model using the predefined hyperparameters
def train_catboost_model(X_train, y_train, X_val, y_val, categorical_cols, hyperparams, model_name):
    """
    Train a CatBoost classifier with the specified hyperparameters.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    categorical_cols : list
        List of categorical column names
    hyperparams : dict
        Dictionary of CatBoost hyperparameters
    model_name : str
        Name of the model for logging purposes
    
    Returns
    -------
    CatBoostClassifier
        Trained model
    """
    model = CatBoostClassifier(train_dir='/tmp/catboost_info', **hyperparams)
    
    model.fit(
        X_train,
        y_train,
        cat_features=categorical_cols,
        eval_set=(X_val, y_val),
        verbose=20
    )
    
    print(f"\n{model_name} model training complete.")
    gc.collect() # Cleaning memory to ensure space for multiple training runs
    return model

# Define the positive prediction probability threshold for each model (the level of confidence the model needs to predict registration/noa/abandonment). Calibrate for different performance thresholds of precision and recall to enable different use cases.
def calibrate_thresholds(model, X_val, y_val, target_precision, target_recall, model_name):
    """
    Find optimal thresholds for precision and recall targets.
    
    Parameters
    ----------
    model : CatBoostClassifier
        Trained model
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    target_precision : float
        Target precision for primary threshold
    target_recall : float
        Target recall for secondary threshold
    model_name : str
        Name of the model for logging purposes
    
    Returns
    -------
    dict
        Dictionary containing thresholds, metrics, and PR curve figure
    """
    # Predict probabilities on validation data
    val_probs = model.predict_proba(X_val)[:, 1]
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
    
    # ────────────────────────────────────────────────────────────────────────
    # Find threshold for target precision
    # ────────────────────────────────────────────────────────────────────────
    valid_idx = np.where(precision[:-1] >= target_precision)[0]
    
    if len(valid_idx) == 0:
        print(f"Warning: No threshold achieves {target_precision:.0%} precision on validation.")
        best_idx = np.argmax(precision[:-1])
        threshold_precision = thresholds[best_idx]
    else:
        best_idx = valid_idx[np.argmax(recall[valid_idx])]
        threshold_precision = thresholds[best_idx]
    
    print(f"\n{model_name} Model Threshold Selection:")
    print(f"Selected threshold: {threshold_precision:.4f}")
    print(f"Validation precision: {precision[best_idx]:.4f}")
    print(f"Validation recall: {recall[best_idx]:.4f}")
    
    # Verify with sklearn metrics
    val_pred = (val_probs >= threshold_precision).astype(int)
    print(f"\nVerification with sklearn metrics:")
    print(f"Precision: {precision_score(y_val, val_pred):.4f}")
    print(f"Recall: {recall_score(y_val, val_pred):.4f}")
    print(f"F1: {f1_score(y_val, val_pred):.4f}")
    
    # ────────────────────────────────────────────────────────────────────────
    # Find recall-based threshold
    # ────────────────────────────────────────────────────────────────────────
    valid_recall_idx = np.where(recall[:-1] >= target_recall)[0]
    if len(valid_recall_idx) == 0:
        print(f"\nWarning: No threshold achieves {target_recall:.0%} recall on validation.")
        best_recall_idx = np.argmax(recall[:-1])
        threshold_recall = thresholds[best_recall_idx]
    else:
        best_recall_idx = valid_recall_idx[np.argmax(precision[valid_recall_idx])]
        threshold_recall = thresholds[best_recall_idx]
    
    val_pred_recall = (val_probs >= threshold_recall).astype(int)
    print(f"\nRecall-optimized threshold ({target_recall:.0%} recall target):")
    print(f"Selected threshold : {threshold_recall:.4f}")
    print(f"Validation precision: {precision_score(y_val, val_pred_recall):.4f}")
    print(f"Validation recall  : {recall_score(y_val, val_pred_recall):.4f}")
    
    # ────────────────────────────────────────────────────────────────────────
    # Create Precision Recall Curve to log in MLflow
    # ────────────────────────────────────────────────────────────────────────
    fig_pr, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="steelblue")
    ax.scatter([recall[best_idx]], [precision[best_idx]], color="blue", zorder=5,
               label=f"Prec. target ({target_precision:.0%}) — threshold {threshold_precision:.3f}")
    ax.scatter([recall[best_recall_idx]], [precision[best_recall_idx]], color="green", zorder=5,
               label=f"Recall target ({target_recall:.0%}) — threshold {threshold_recall:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} Model — Precision-Recall Curve (Validation)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    display(fig_pr)
    
    return {
        'threshold_precision': threshold_precision,
        'threshold_recall': threshold_recall,
        'val_probs': val_probs,
        'val_pred': val_pred,
        'val_pred_recall': val_pred_recall,
        'fig_pr': fig_pr
    }

# The model has been validated/calibrated on the validation set, but still needs to be evaluated based on the hold-out set. 
def evaluate_model_performance(model, X_test, y_test, threshold, model_name):
    """
    Evaluate model performance on test set.
    
    Parameters
    ----------
    model : CatBoostClassifier
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    threshold : float
        Classification threshold
    model_name : str
        Name of the model for logging purposes
    
    Returns
    -------
    dict
        Dictionary containing test probabilities and predictions
    """
    # Predict on test set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )
    
    print(f"\n{model_name} Model - Test Set Confusion Matrix:")
    display(cm_df)
    
    # Metrics
    print(f"\n{model_name} Model - Test Set Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, test_pred):.4f}")
    print(f"Recall: {recall_score(y_test, test_pred):.4f}")
    print(f"F1: {f1_score(y_test, test_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    return {
        'test_probs': test_probs,
        'test_pred': test_pred
    }

# Take the model and the artifacts (e.g. PR curve, calibration thresholds, evaluation metrics) and register them together in Unity Catalog
def register_model_to_uc(model, X_train, y_train, y_val, y_test, calibration_results, test_results, 
                          model_name, hyperparams, training_metadata):
    """
    Register model to Workspace Model Registry with MLflow.
    
    Parameters
    ----------
    model : CatBoostClassifier
        Trained model
    X_train : pd.DataFrame
        Training features for signature
    y_train, y_val, y_test : pd.Series
        Labels for computing rates
    calibration_results : dict
        Results from calibrate_thresholds
    test_results : dict
        Results from evaluate_model_performance
    model_name : str
        Model name for registration
    hyperparams : dict
        Model hyperparameters
    training_metadata : dict
        Training metadata (batch_id, training_start, training_end, target_precision, target_recall)
    
    Returns
    -------
    str
        MLflow run ID
    """

    
    mlflow.set_registry_uri("databricks")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Build signature & input example from training data
    sample_input = X_train[:5]
    sample_output = model.predict_proba(sample_input)[:, 1]
    signature = infer_signature(sample_input, sample_output)
    
    # Log model and register to Unity Catalog
    with mlflow.start_run(run_name=f"{model_name}_catboost") as run:
        # Hyperparameters, calibration settings & data split
        mlflow.log_params({
            **hyperparams,
            "target_precision": training_metadata['target_precision'],
            "target_recall": training_metadata['target_recall'],
            "n_train": X_train.shape[0],
            "n_val": len(y_val),
            "n_test": len(y_test),
            "n_features": X_train.shape[1],
        })
        
        # Lineage tags
        mlflow.set_tags({
            "training_batch_id": training_metadata['training_batch_id'],
            "training_table": TRAINING_TABLE,
            "training_start": str(training_metadata['training_start']),
            "training_end": str(training_metadata['training_end']),
        })
        
        # Recall-threshold test predictions (computed once for the metrics below)
        test_pred_recall = (test_results['test_probs'] >= calibration_results['threshold_recall']).astype(int)

        mlflow.log_metrics({
            f"threshold_{int(training_metadata['target_precision']*100)}pct_precision": calibration_results['threshold_precision'],
            f"threshold_{int(training_metadata['target_recall']*100)}pct_recall": calibration_results['threshold_recall'],
            "pos_rate_train": float(y_train.mean()),
            # Precision-threshold operating point
            "val_precision_at_precision_threshold": precision_score(y_val, calibration_results['val_pred']),
            "val_recall_at_precision_threshold": recall_score(y_val, calibration_results['val_pred']),
            "val_f1_at_precision_threshold": f1_score(y_val, calibration_results['val_pred']),
            "test_precision_at_precision_threshold": precision_score(y_test, test_results['test_pred']),
            "test_recall_at_precision_threshold": recall_score(y_test, test_results['test_pred']),
            "test_f1_at_precision_threshold": f1_score(y_test, test_results['test_pred']),
            "test_accuracy_at_precision_threshold": accuracy_score(y_test, test_results['test_pred']),
            # Recall-threshold operating point
            "val_precision_at_recall_threshold": precision_score(y_val, calibration_results['val_pred_recall']),
            "val_recall_at_recall_threshold": recall_score(y_val, calibration_results['val_pred_recall']),
            "val_f1_at_recall_threshold": f1_score(y_val, calibration_results['val_pred_recall']),
            "test_precision_at_recall_threshold": precision_score(y_test, test_pred_recall),
            "test_recall_at_recall_threshold": recall_score(y_test, test_pred_recall),
            "test_f1_at_recall_threshold": f1_score(y_test, test_pred_recall),
            "test_accuracy_at_recall_threshold": accuracy_score(y_test, test_pred_recall),
        })
        
        mlflow.log_figure(calibration_results['fig_pr'], "pr_curve.png")

        # ────────────────────────────────────────────────────────────────────────
        # Global SHAP feature importance (CatBoost native, sample of training data)
        # ────────────────────────────────────────────────────────────────────────
        shap_n = builtins.min(2000, len(X_train))
        rng = np.random.default_rng(seed=42)
        X_shap = X_train.iloc[rng.choice(len(X_train), size=shap_n, replace=False)]

        shap_pool = Pool(X_shap, cat_features=categorical_cols)
        shap_matrix = model.get_feature_importance(type='ShapValues', data=shap_pool)

        # Last column is the bias term — exclude it
        mean_abs_shap = np.abs(shap_matrix[:, :-1]).mean(axis=0)
        shap_series = pd.Series(mean_abs_shap, index=X_train.columns).sort_values(ascending=False)

        # Log raw values
        mlflow.log_dict(shap_series.to_dict(), "shap_mean_abs.json")

        # Log top-20 bar chart
        n_top = builtins.min(20, len(shap_series))
        fig_shap, ax_shap = plt.subplots(figsize=(8, builtins.max(4, n_top * 0.4)))
        shap_series.head(n_top).sort_values().plot.barh(ax=ax_shap, color="steelblue")
        ax_shap.set_xlabel("Mean |SHAP value|")
        ax_shap.set_title(f"{model_name} — Global Feature Importance (SHAP, n={shap_n:,})")
        plt.tight_layout()
        mlflow.log_figure(fig_shap, "shap_importance.png")
        plt.close(fig_shap)

        model_info = mlflow.catboost.log_model(
            model,
            name="model",
            signature=signature,
            input_example=sample_input,
            registered_model_name=model_name,
        )
        
        plt.close(calibration_results['fig_pr'])
        run_id = run.info.run_id
        print(f"Registered model : {model_name}")
        print(f"Model URI        : {model_info.model_uri}")
        print(f"Run ID           : {run_id}")
        
        return run_id

# To maximize performance, a secondary ensemble calibration is used. The ensemble run combines information from multiple models (e.g. likelihood of registration, likelihood of abandonment) and uses this combined information to make classification determinations, enabling a more favorable precision/recall split. This phase requires optimization between the multiple thresholds, and is thus more complex than the single-threshold approach.
def run_ensemble_grid_search(val_reg_arr, val_abn_arr, y_val_arr, total_positives,
                              min_precision=None, min_recall=None, grid_size=200, objective='recall'):
    """
    Run grid search over (reg_threshold, abn_exa_ceiling) combinations.

    Evaluates all REG thresholds via a cumulative-sum scan — O(N log N) per ABN
    ceiling instead of the previous O(N²) threshold loop.

    Parameters
    ----------
    val_reg_arr : np.ndarray
        Registration probabilities on validation set
    val_abn_arr : np.ndarray
        Abandonment probabilities on validation set
    y_val_arr : np.ndarray
        True labels on validation set
    total_positives : int
        Total positive cases (for recall calculation)
    min_precision : float, optional
        Minimum precision constraint (for recall optimization)
    min_recall : float, optional
        Minimum recall constraint (for precision optimization)
    grid_size : int
        Number of abn_exa ceiling values to test
    objective : str
        'recall' to maximize recall, 'precision' to maximize precision

    Returns
    -------
    tuple
        (sweep_df, optimal_reg_threshold, optimal_abn_ceiling, best_row)
    """
    abn_ceilings = np.linspace(0.0, 0.99, grid_size)
    sweep_results = []

    for abn_t in abn_ceilings:
        eligible = val_abn_arr <= abn_t
        n_eligible = int(eligible.sum())

        if n_eligible < MIN_ELIGIBLE_RECORDS or y_val_arr[eligible].sum() == 0:
            continue

        reg_sub = val_reg_arr[eligible]
        y_sub = y_val_arr[eligible]

        # Sort by REG probability descending — O(N log N), done once per ABN ceiling
        sort_idx = np.argsort(reg_sub)[::-1]
        y_sorted = y_sub[sort_idx]
        reg_sorted = reg_sub[sort_idx]

        # Cumulative TP and PP at every rank cutoff — O(N)
        cum_tp = np.cumsum(y_sorted)
        pp = np.arange(1, n_eligible + 1)
        precision_arr = cum_tp / pp
        recall_arr = cum_tp / total_positives

        # Find the rank that optimises the objective subject to the constraint
        if objective == 'recall':
            if min_precision is None:
                continue
            masked = np.where(precision_arr >= min_precision, recall_arr, -1.0)
        else:
            if min_recall is None:
                continue
            masked = np.where(recall_arr >= min_recall, precision_arr, -1.0)

        if masked.max() < 0:
            continue

        best_k = int(np.argmax(masked))

        sweep_results.append({
            'abn_exa_ceiling': builtins.round(float(abn_t), 5),
            'reg_threshold':   builtins.round(float(reg_sorted[best_k]), 5),
            'val_precision':   builtins.round(float(precision_arr[best_k]), 5),
            'val_recall':      builtins.round(float(recall_arr[best_k]), 5),
            'val_predictions': int(pp[best_k]),
            'n_eligible':      n_eligible,
        })

    sweep_df = pd.DataFrame(sweep_results)

    if len(sweep_df) == 0:
        return sweep_df, None, None, None

    # Find global optimum
    if objective == 'recall':
        best_row = sweep_df.loc[sweep_df['val_recall'].idxmax()]
    else:
        best_row = sweep_df.loc[sweep_df['val_precision'].idxmax()]

    opt_reg_threshold = best_row['reg_threshold']
    opt_abn_ceiling   = best_row['abn_exa_ceiling']

    return sweep_df, opt_reg_threshold, opt_abn_ceiling, best_row

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation

# COMMAND ----------

# DBTITLE 1,Training Start Date
# ────────────────────────────────────────────────────────────────────────────
# Training window: records filed between 72 and 36 months ago.
# This ensures cases are old enough to have a known outcome (registered or
# abandoned) while staying within a recent, representative time range.
# ────────────────────────────────────────────────────────────────────────────

today = date.today()
training_start = today - relativedelta(months=TRAINING_LOOKBACK_MONTHS)   # 72 months ago
training_end   = today - relativedelta(months=36)                          # 36 months ago

print(f"Today           : {today}")
print(f"Training start  : {training_start}  ({TRAINING_LOOKBACK_MONTHS} months ago)")
print(f"Training end    : {training_end}  (36 months ago)")
print(f"Window          : {(training_end - training_start).days} days")

# COMMAND ----------

# DBTITLE 1,Initial Dataframe
features_df = spark.sql(f"""
SELECT * 
FROM {TRAINING_TABLE}
WHERE
  serial_number IS NOT NULL
  AND disposal_nul_flag != 1
  AND filing_dt >= '{training_start}'
  AND filing_dt <= '{training_end}'
""")

# Convert numeric columns that should be treated as categorical
numeric_class_predictors = [
    "entity_type",
    "filing_day_of_month",
    "filing_month"
]

for col_name in numeric_class_predictors:
    features_df = features_df.withColumn(
        col_name, features_df[col_name].cast("string")
    )

features_df.createOrReplaceTempView("features")
#display(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unified Training Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Shared Feature Set
# MAGIC
# MAGIC All three models will use the same feature set for consistency.

# COMMAND ----------

# DBTITLE 1,Feature Set
# Select features for modeling (excluding target variables and identifiers)
feature_columns = [
    'filing_basis_fil',
    'num_classes_at_application',
    'legal_representation',
    'mark_description_length',
    'us_filing',
    'entity_type',
    'gs_top_100',
    'gs_mid_101_999',
    'gs_low_1000_plus',
    'gs_number',
    'goods',
    'services',
    'class_combo',
    # Individual class indicators (45 binary features)
    'has_class_1', 'has_class_2', 'has_class_3', 'has_class_4', 'has_class_5',
    'has_class_6', 'has_class_7', 'has_class_8', 'has_class_9', 'has_class_10',
    'has_class_11', 'has_class_12', 'has_class_13', 'has_class_14', 'has_class_15',
    'has_class_16', 'has_class_17', 'has_class_18', 'has_class_19', 'has_class_20',
    'has_class_21', 'has_class_22', 'has_class_23', 'has_class_24', 'has_class_25',
    'has_class_26', 'has_class_27', 'has_class_28', 'has_class_29', 'has_class_30',
    'has_class_31', 'has_class_32', 'has_class_33', 'has_class_34', 'has_class_35',
    'has_class_36', 'has_class_37', 'has_class_38', 'has_class_39', 'has_class_40',
    'has_class_41', 'has_class_42', 'has_class_43', 'has_class_44', 'has_class_45',
    'country_of_origin',
    'state_of_origin',
    'mark_dwg_desc',
    'is_high_filing_week',
    'filing_month',
    'filing_day',
    'filing_day_of_month',
    'pseudo',
    'designs',
    'disclaimer',
    'color'
]

# Define categorical columns for CatBoost
categorical_cols = [
    'filing_basis_fil',
    'legal_representation',
    'entity_type',
    'class_combo',
    'country_of_origin',
    'state_of_origin',
    'mark_dwg_desc',
    'filing_month',
    'filing_day',
    'filing_day_of_month'
]

# Convert to pandas for sklearn/catboost
model_df = features_df.select(
    'serial_number',
    'disposal_reg_flag',
    'disposal_abn_flag',
    'disposal_noa_flag',
    'abn_exa_flag',
    *feature_columns
).toPandas()

print(f"Model dataset shape: {model_df.shape}")
print(f"\nTarget variable distributions:")
print(f"Registration: {model_df['disposal_reg_flag'].sum()} ({100*model_df['disposal_reg_flag'].mean():.1f}%)")
print(f"Abandonment: {model_df['disposal_abn_flag'].sum()} ({100*model_df['disposal_abn_flag'].mean():.1f}%)")
print(f"NOA: {model_df['disposal_noa_flag'].sum()} ({100*model_df['disposal_noa_flag'].mean():.1f}%)")

# Clean up Spark DataFrames to free memory
del features_df
gc.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shared Train/Validation/Test Split
# MAGIC
# MAGIC Create a single train/val/test split that all three models will use.

# COMMAND ----------

# DBTITLE 1,Train/Test/Val Split
# We will first split into train and test, and will later split train in train and validate, in order to achieve a 70%, 20%, 10% train/test/val distribution. We need to maintain class balance for target variables across sets.

# Store serial numbers for later joining
serial_numbers = model_df['serial_number']

# Extract features
X_all = model_df[feature_columns]

# Store all target variables
y_reg = model_df['disposal_reg_flag']
y_abn = model_df['disposal_abn_flag']
y_noa = model_df['disposal_noa_flag']
y_abn_exa = model_df['abn_exa_flag']

# Create multi-label stratification key
# Combines all three targets to ensure balanced distribution across all outcomes
stratify_key = y_reg.astype(str) + '_' + y_abn.astype(str) + '_' + y_noa.astype(str)

print("Multi-label stratification key distribution:")
print(stratify_key.value_counts())
print(f"\nTotal unique combinations: {stratify_key.nunique()}")

# Create train/temp split (80/20) with multi-label stratification
X_temp, X_test, y_reg_temp, y_reg_test, y_abn_temp, y_abn_test, y_noa_temp, y_noa_test, y_abn_exa_temp, y_abn_exa_test, serial_temp, serial_test, stratify_temp, stratify_test = train_test_split(
    X_all, y_reg, y_abn, y_noa, y_abn_exa, serial_numbers, stratify_key,
    test_size=0.2,
    random_state=42,
    stratify=stratify_key
)

# Create train/val split from temp (70/10 of original) with multi-label stratification
X_train, X_val, y_reg_train, y_reg_val, y_abn_train, y_abn_val, y_noa_train, y_noa_val, y_abn_exa_train, y_abn_exa_val, serial_train, serial_val, stratify_train, stratify_val = train_test_split(
    X_temp, y_reg_temp, y_abn_temp, y_noa_temp, y_abn_exa_temp, serial_temp, stratify_temp,
    test_size=0.125,
    random_state=42,
    stratify=stratify_temp
)

# Handle missing values in categorical columns
for c in categorical_cols:
    X_train[c] = X_train[c].astype(object).fillna("<<MISSING>>")
    X_val[c] = X_val[c].astype(object).fillna("<<MISSING>>")
    X_test[c] = X_test[c].astype(object).fillna("<<MISSING>>")

print(f"Training set size: {X_train.shape[0]} ({100*X_train.shape[0]/len(X_all):.1f}%)")
print(f"Validation set size: {X_val.shape[0]} ({100*X_val.shape[0]/len(X_all):.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({100*X_test.shape[0]/len(X_all):.1f}%)")

# Verify class balance across splits
print(f"\n{'='*60}")
print("CLASS BALANCE VERIFICATION")
print(f"{'='*60}")
print(f"\nRegistration rates:")
print(f"  Train: {y_reg_train.mean():.1%} ({y_reg_train.sum()} cases)")
print(f"  Val:   {y_reg_val.mean():.1%} ({y_reg_val.sum()} cases)")
print(f"  Test:  {y_reg_test.mean():.1%} ({y_reg_test.sum()} cases)")

print(f"\nAbandonment rates:")
print(f"  Train: {y_abn_train.mean():.1%} ({y_abn_train.sum()} cases)")
print(f"  Val:   {y_abn_val.mean():.1%} ({y_abn_val.sum()} cases)")
print(f"  Test:  {y_abn_test.mean():.1%} ({y_abn_test.sum()} cases)")

print(f"\nNOA rates:")
print(f"  Train: {y_noa_train.mean():.1%} ({y_noa_train.sum()} cases)")
print(f"  Val:   {y_noa_val.mean():.1%} ({y_noa_val.sum()} cases)")
print(f"  Test:  {y_noa_test.mean():.1%} ({y_noa_test.sum()} cases)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 1: Registration Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Registration Model

# COMMAND ----------

# DBTITLE 1,REG: Training
reg_model = train_catboost_model(
    X_train, y_reg_train, X_val, y_reg_val,
    categorical_cols, REG_PARAMS, "Registration"
)

# COMMAND ----------

# DBTITLE 1,REG: Calibration
reg_calibration = calibrate_thresholds(
    reg_model, X_val, y_reg_val,
    TARGET_PRECISION, TARGET_RECALL, "REG"
)
reg_threshold = reg_calibration['threshold_precision']
reg_threshold_recall = reg_calibration['threshold_recall']

# COMMAND ----------

# DBTITLE 1,REG: Evaluation
reg_test = evaluate_model_performance(
    reg_model, X_test, y_reg_test,
    reg_threshold, "Registration"
)

# COMMAND ----------

# DBTITLE 1,REG: Save Model
training_metadata = {
    'training_batch_id': TRAINING_BATCH_ID,
    'training_start': training_start,
    'training_end': training_end,
    'target_precision': TARGET_PRECISION,
    'target_recall': TARGET_RECALL
}

reg_run_id = register_model_to_uc(
    reg_model, X_train, y_reg_train, y_reg_val, y_reg_test,
    reg_calibration, reg_test,
    "reg_model",
    REG_PARAMS, training_metadata
)

# Clean up to free memory
del reg_model
gc.collect()
print("REG model cleaned from memory")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 2: Examination Abandonment (EXA) Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train EXA Model

# COMMAND ----------

# DBTITLE 1,ABN_EXA: Training
abn_exa_model = train_catboost_model(
    X_train, y_abn_exa_train, X_val, y_abn_exa_val,
    categorical_cols, ABN_EXA_PARAMS, "Examination Abandonment"
)

# COMMAND ----------

# DBTITLE 1,ABN_EXA: Calibration
abn_exa_calibration = calibrate_thresholds(
    abn_exa_model, X_val, y_abn_exa_val,
    TARGET_PRECISION, TARGET_RECALL, "ABN_EXA"
)
abn_exa_threshold = abn_exa_calibration['threshold_precision']
abn_exa_threshold_recall = abn_exa_calibration['threshold_recall']

# COMMAND ----------

# DBTITLE 1,ABN_EXA: Evaluation
abn_exa_test = evaluate_model_performance(
    abn_exa_model, X_test, y_abn_exa_test,
    abn_exa_threshold, "Examination Abandonment"
)

# COMMAND ----------

# DBTITLE 1,ABN_EXA: Save Model
abn_exa_run_id = register_model_to_uc(
    abn_exa_model, X_train, y_abn_exa_train, y_abn_exa_val, y_abn_exa_test,
    abn_exa_calibration, abn_exa_test,
    "abn_exa_model",
    ABN_EXA_PARAMS, training_metadata
)

# Clean up to free memory
del abn_exa_model
gc.collect()
print("ABN_EXA model cleaned from memory")

# COMMAND ----------

# DBTITLE 1,Untitled
# MAGIC %md
# MAGIC ## Model 3: NOA Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train NOA Model

# COMMAND ----------

# DBTITLE 1,NOA: Training
noa_model = train_catboost_model(
    X_train, y_noa_train, X_val, y_noa_val,
    categorical_cols, NOA_PARAMS, "NOA"
)

# COMMAND ----------

# DBTITLE 1,NOA: Calibration
noa_calibration = calibrate_thresholds(
    noa_model, X_val, y_noa_val,
    TARGET_PRECISION, TARGET_RECALL, "NOA"
)
noa_threshold = noa_calibration['threshold_precision']
noa_threshold_recall = noa_calibration['threshold_recall']

# COMMAND ----------

# DBTITLE 1,NOA: Evaluation
noa_test = evaluate_model_performance(
    noa_model, X_test, y_noa_test,
    noa_threshold, "NOA"
)

# COMMAND ----------

# DBTITLE 1,NOA: Save Model
noa_run_id = register_model_to_uc(
    noa_model, X_train, y_noa_train, y_noa_val, y_noa_test,
    noa_calibration, noa_test,
    "noa_model",
    NOA_PARAMS, training_metadata
)

# Clean up to free memory
del noa_model
gc.collect()
print("NOA model cleaned from memory")

# COMMAND ----------

# DBTITLE 1,Ensemble Optimization Header
# MAGIC %md
# MAGIC ## Ensemble Threshold Optimization
# MAGIC
# MAGIC Automated grid search over two parameters:
# MAGIC - **REG threshold** — minimum registration probability to predict positive
# MAGIC - **ABN-EXA ceiling** — maximum abandonment probability allowed (records above this are excluded)
# MAGIC
# MAGIC Objective: **maximize recall** subject to **precision ≥ 92.5%**.
# MAGIC
# MAGIC All optimization is performed on the **validation set**. The test set is used only once, at the end, to report honest out-of-sample metrics.

# COMMAND ----------

# DBTITLE 1,Ensemble: 92.5% Precision
# ────────────────────────────────────────────────────────────────────────────
# Grid search over (reg_threshold, abn_exa_ceiling) on the VALIDATION SET
# For each ABN-EXA ceiling, find the REG threshold that maximises recall
# at precision >= 92.5%, then select the globally best point.
#
# IMPORTANT: Recall is computed against the FULL validation set denominator
# (total positives regardless of ABN filter), not just the eligible subset.
# This prevents the optimizer from selecting degenerate solutions where a
# tight ABN ceiling yields 100% recall on a tiny subset but near-zero
# recall on the full population.
# ────────────────────────────────────────────────────────────────────────────

import builtins
round = builtins.round  # pyspark.sql.functions.round shadows the built-in

val_reg_arr = np.array(reg_calibration['val_probs'])
val_abn_arr = np.array(abn_exa_calibration['val_probs'])
y_val_arr = np.array(y_reg_val)

# Full-population positive count — denominator for recall at every grid point
total_positives = y_val_arr.sum()

sweep_df, opt_reg_threshold, opt_abn_ceiling, best_row = run_ensemble_grid_search(
    val_reg_arr, val_abn_arr, y_val_arr, total_positives,
    min_precision=MIN_PRECISION_PRIMARY,
    grid_size=ABN_GRID_SIZE,
    objective='recall'
)

print('=' * 70)
print('GRID SEARCH COMPLETE  —  calibrated on validation set')
print('=' * 70)
print(f'\n  REG threshold   (>=): {opt_reg_threshold:.4f}')
print(f'  ABN-EXA ceiling (<=): {opt_abn_ceiling:.4f}')
print(f'\n  Validation precision : {best_row["val_precision"]:.4f}')
print(f'  Validation recall    : {best_row["val_recall"]:.4f}')
print(f'  Validation predictions: {best_row["val_predictions"]}'
      f' ({100*best_row["val_predictions"]/len(val_reg_arr):.1f}% of val set)')

# COMMAND ----------

# DBTITLE 1,Ensemble: 90% Recall
# ────────────────────────────────────────────────────────────────────────────
# Secondary optimisation: maximise PRECISION subject to RECALL ≥ 90%
# Same grid structure as above, different objective function.
# Uses the FULL validation set denominator for recall (same as primary).
# ────────────────────────────────────────────────────────────────────────────

sweep_df_90r, opt_reg_threshold_90r, opt_abn_ceiling_90r, best_row_90r = run_ensemble_grid_search(
    val_reg_arr, val_abn_arr, y_val_arr, total_positives,
    min_recall=MIN_RECALL_SECONDARY,
    grid_size=ABN_GRID_SIZE,
    objective='precision'
)

if opt_reg_threshold_90r is None:
    print("WARNING: No configuration achieves >= 90% recall on the validation set.")
else:
    print('=' * 70)
    print('SECONDARY GRID SEARCH  —  maximise precision at recall >= 90%')
    print('=' * 70)
    print(f'\n  REG threshold   (>=): {opt_reg_threshold_90r:.4f}')
    print(f'  ABN-EXA ceiling (<=): {opt_abn_ceiling_90r:.4f}')
    print(f'\n  Validation precision : {best_row_90r["val_precision"]:.4f}')
    print(f'  Validation recall    : {best_row_90r["val_recall"]:.4f}')
    print(f'  Validation predictions: {best_row_90r["val_predictions"]}'
          f' ({100*best_row_90r["val_predictions"]/len(val_reg_arr):.1f}% of val set)')



# COMMAND ----------

# DBTITLE 1,Ensemble: 95% Recall
# ────────────────────────────────────────────────────────────────────────────
# Tertiary optimisation: maximise PRECISION subject to RECALL >= 95%
# Same grid structure as above, different recall floor.
# Uses the FULL validation set denominator for recall (same as primary).
# ────────────────────────────────────────────────────────────────────────────

sweep_df_95r, opt_reg_threshold_95r, opt_abn_ceiling_95r, best_row_95r = run_ensemble_grid_search(
    val_reg_arr, val_abn_arr, y_val_arr, total_positives,
    min_recall=MIN_RECALL_TERTIARY,
    grid_size=ABN_GRID_SIZE,
    objective='precision'
)

if opt_reg_threshold_95r is None:
    print("WARNING: No configuration achieves >= 95% recall on the validation set.")
else:
    print('=' * 70)
    print('TERTIARY GRID SEARCH  —  maximise precision at recall >= 95%')
    print('=' * 70)
    print(f'\n  REG threshold   (>=): {opt_reg_threshold_95r:.4f}')
    print(f'  ABN-EXA ceiling (<=): {opt_abn_ceiling_95r:.4f}')
    print(f'\n  Validation precision : {best_row_95r["val_precision"]:.4f}')
    print(f'  Validation recall    : {best_row_95r["val_recall"]:.4f}')
    print(f'  Validation predictions: {best_row_95r["val_predictions"]}'
          f' ({100*best_row_95r["val_predictions"]/len(val_reg_arr):.1f}% of val set)')



# COMMAND ----------

# DBTITLE 1,Ensemble: Optimized Test Evaluation
# ────────────────────────────────────────────────────────────────────────────
# Apply optimal thresholds to TEST SET
# ────────────────────────────────────────────────────────────────────────────

test_reg_arr = np.array(reg_test['test_probs'])
test_abn_arr = np.array(abn_exa_test['test_probs'])

test_preds_opt = (
    (test_reg_arr >= opt_reg_threshold) &
    (test_abn_arr <= opt_abn_ceiling)
).astype(int)

# Baselines for comparison
test_preds_reg_only = (test_reg_arr >= reg_threshold).astype(int)
# Secondary optimisation: recall >= 90% operating point
if opt_reg_threshold_90r is not None:
    test_preds_90r = (
        (test_reg_arr >= opt_reg_threshold_90r) &
        (test_abn_arr <= opt_abn_ceiling_90r)
    ).astype(int)
else:
    test_preds_90r = np.zeros_like(test_preds_opt)

# Tertiary optimisation: recall >= 95% operating point
if opt_reg_threshold_95r is not None:
    test_preds_95r = (
        (test_reg_arr >= opt_reg_threshold_95r) &
        (test_abn_arr <= opt_abn_ceiling_95r)
    ).astype(int)
else:
    test_preds_95r = np.zeros_like(test_preds_opt)

rows = []
for label, preds in [
    ('REG-only (95% prec threshold)', test_preds_reg_only),
    ('Optimised (max recall, prec>=92.5%)', test_preds_opt),
    ('Optimised (max prec, recall>=90%)', test_preds_90r),
    ('Optimised (max prec, recall>=95%)', test_preds_95r),
]:
    n = preds.sum()
    prec = precision_score(y_reg_test, preds, zero_division=0)
    rec = recall_score(y_reg_test, preds, zero_division=0)
    f1 = f1_score(y_reg_test, preds, zero_division=0)
    rows.append({
        'Approach': label,
        'Predictions': n,
        'Coverage %': round(100 * preds.mean(), 1),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
    })

print('=' * 70)
print('TEST SET RESULTS')
print('=' * 70)
display(pd.DataFrame(rows))

cm = confusion_matrix(y_reg_test, test_preds_opt)
display(pd.DataFrame(cm,
    index=['Actual Neg', 'Actual Pos'],
    columns=['Pred Neg', 'Pred Pos']
))

# COMMAND ----------

# DBTITLE 1,Ensemble: Log Optimization Results to MLflow
# ────────────────────────────────────────────────────────────────────────────
# Dedicated MLflow run for ensemble threshold optimization
# Links to individual model runs via training_batch_id tag.
# Prediction notebooks query this run to retrieve production thresholds.
# ────────────────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="ensemble_optimization") as ensemble_run:

    # Link to individual model runs and training metadata
    mlflow.set_tags({
        "training_batch_id": TRAINING_BATCH_ID,
        "reg_model_run_id": reg_run_id,
        "abn_exa_run_id": abn_exa_run_id,
        "noa_run_id": noa_run_id,
        "training_table": TRAINING_TABLE,
        "training_start": str(training_start),
        "training_end": str(training_end),
        "production_thresholds": "true",
        "optimization_complete": "true",
    })

    # Optimization objective configuration
    mlflow.log_params({
        "min_precision_primary": MIN_PRECISION_PRIMARY,
        "min_recall_secondary": MIN_RECALL_SECONDARY,
        "min_recall_tertiary": MIN_RECALL_TERTIARY,
        "abn_grid_size": ABN_GRID_SIZE,
    })

    # ────────────────────────────────────────────────────────────────────────
    # Primary: maximize recall at precision ≥ 92.5%
    # ────────────────────────────────────────────────────────────────────────
    mlflow.log_metrics({
        "primary_reg_threshold": float(opt_reg_threshold),
        "primary_abn_ceiling": float(opt_abn_ceiling),
        "primary_val_precision": float(best_row["val_precision"]),
        "primary_val_recall": float(best_row["val_recall"]),
        "primary_test_precision": precision_score(y_reg_test, test_preds_opt),
        "primary_test_recall": recall_score(y_reg_test, test_preds_opt),
        "primary_test_f1": f1_score(y_reg_test, test_preds_opt),
    })

    # ────────────────────────────────────────────────────────────────────────
    # Secondary: maximize precision at recall ≥ 90%
    # ────────────────────────────────────────────────────────────────────────
    if opt_reg_threshold_90r is not None:
        mlflow.log_metrics({
            "secondary_reg_threshold": float(opt_reg_threshold_90r),
            "secondary_abn_ceiling": float(opt_abn_ceiling_90r),
            "secondary_val_precision": float(best_row_90r["val_precision"]),
            "secondary_val_recall": float(best_row_90r["val_recall"]),
            "secondary_test_precision": precision_score(y_reg_test, test_preds_90r),
            "secondary_test_recall": recall_score(y_reg_test, test_preds_90r),
            "secondary_test_f1": f1_score(y_reg_test, test_preds_90r),
        })

    # ────────────────────────────────────────────────────────────────────────
    # Tertiary: maximize precision at recall ≥ 95%
    # ────────────────────────────────────────────────────────────────────────
    if opt_reg_threshold_95r is not None:
        mlflow.log_metrics({
            "tertiary_reg_threshold": float(opt_reg_threshold_95r),
            "tertiary_abn_ceiling": float(opt_abn_ceiling_95r),
            "tertiary_val_precision": float(best_row_95r["val_precision"]),
            "tertiary_val_recall": float(best_row_95r["val_recall"]),
            "tertiary_test_precision": precision_score(y_reg_test, test_preds_95r),
            "tertiary_test_recall": recall_score(y_reg_test, test_preds_95r),
            "tertiary_test_f1": f1_score(y_reg_test, test_preds_95r),
        })

    # ────────────────────────────────────────────────────────────────────────
    # Export threshold configuration as JSON artifact
    # ────────────────────────────────────────────────────────────────────────
    threshold_config = {
        "config_version": "1.0",
        "training_batch_id": TRAINING_BATCH_ID,
        "optimization_timestamp": str(pd.Timestamp.now()),
        "optimization_objectives": {
            "primary": f"maximize_recall_at_precision>={MIN_PRECISION_PRIMARY}",
            "secondary": f"maximize_precision_at_recall>={MIN_RECALL_SECONDARY}",
            "tertiary": f"maximize_precision_at_recall>={MIN_RECALL_TERTIARY}"
        },
        "reg_threshold_95_precision": float(reg_threshold),
        "primary_thresholds": {
            "reg_threshold": float(opt_reg_threshold),
            "abn_ceiling": float(opt_abn_ceiling),
            "target_precision": MIN_PRECISION_PRIMARY,
            "val_precision": float(best_row["val_precision"]),
            "val_recall": float(best_row["val_recall"])
        },
        "secondary_thresholds": {
            "reg_threshold": float(opt_reg_threshold_90r) if opt_reg_threshold_90r else None,
            "abn_ceiling": float(opt_abn_ceiling_90r) if opt_abn_ceiling_90r else None,
            "target_recall": MIN_RECALL_SECONDARY,
            "val_precision": float(best_row_90r["val_precision"]) if opt_reg_threshold_90r else None,
            "val_recall": float(best_row_90r["val_recall"]) if opt_reg_threshold_90r else None
        },
        "tertiary_thresholds": {
            "reg_threshold": float(opt_reg_threshold_95r) if opt_reg_threshold_95r else None,
            "abn_ceiling": float(opt_abn_ceiling_95r) if opt_abn_ceiling_95r else None,
            "target_recall": MIN_RECALL_TERTIARY,
            "val_precision": float(best_row_95r["val_precision"]) if opt_reg_threshold_95r else None,
            "val_recall": float(best_row_95r["val_recall"]) if opt_reg_threshold_95r else None
        }
    }
    
    mlflow.log_dict(threshold_config, "threshold_config.json")
    print("✓ Logged threshold configuration artifact")

    # ────────────────────────────────────────────────────────────────────────
    # Export feature configuration as JSON artifact
    # ────────────────────────────────────────────────────────────────────────
    feature_config = {
        "config_version": "1.0",
        "feature_columns": feature_columns,
        "categorical_cols": categorical_cols,
        "numeric_class_predictors": numeric_class_predictors,
        "n_features": len(feature_columns),
        "n_categorical": len(categorical_cols)
    }
    
    mlflow.log_dict(feature_config, "feature_config.json")
    print("✓ Logged feature configuration artifact")

    # ────────────────────────────────────────────────────────────────────────
    # Validate thresholds
    # ────────────────────────────────────────────────────────────────────────
    validation_passed = True
    validation_warnings = []
    
    # Check threshold ranges
    if not (0 <= opt_reg_threshold <= 1):
        validation_warnings.append(f"Primary REG threshold out of range: {opt_reg_threshold}")
        validation_passed = False
    
    if not (0 <= opt_abn_ceiling < 1):
        validation_warnings.append(f"Primary ABN ceiling out of range: {opt_abn_ceiling}")
        validation_passed = False
    
    # Check that precision/recall targets were met on validation
    if best_row["val_precision"] < MIN_PRECISION_PRIMARY - 0.001:  # Small tolerance for rounding
        validation_warnings.append(
            f"Primary validation precision {best_row['val_precision']:.4f} < target {MIN_PRECISION_PRIMARY}"
        )
    
    # Check reasonable threshold values
    if opt_reg_threshold < 0.3:
        validation_warnings.append(f"Primary REG threshold unusually low: {opt_reg_threshold:.4f}")
    
    mlflow.log_metrics({
        "threshold_validation_passed": 1.0 if validation_passed else 0.0,
        "threshold_validation_warnings": float(len(validation_warnings))
    })
    
    if validation_warnings:
        print("\n⚠ VALIDATION WARNINGS:")
        for warning in validation_warnings:
            print(f"  - {warning}")
    else:
        print("✓ All threshold validations passed")

ensemble_run_id = ensemble_run.info.run_id
print(f"Ensemble run ID   : {ensemble_run_id}")
print(f"Training batch ID : {TRAINING_BATCH_ID}")
print(f"\nTo load thresholds in a prediction notebook:")
print(f"  from mlflow.tracking import MlflowClient")
print(f"  run = MlflowClient().get_run('{ensemble_run_id}')")
print(f"  opt_reg_threshold = run.data.metrics['primary_reg_threshold']")
print(f"  opt_abn_ceiling   = run.data.metrics['primary_abn_ceiling']")