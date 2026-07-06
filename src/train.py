import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from imblearn.over_sampling import SMOTE

# Set style for modern dark aesthetics matching our web UI
plt.style.use('dark_background')
sns.set_theme(style="dark", rc={
    "grid.color": "#2c2c35",
    "axes.facecolor": "#1a1a24",
    "figure.facecolor": "#0d0d12",
    "text.color": "#e2e8f0",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
})

def train_and_evaluate(models_dir="models", plots_dir="static/plots"):
    print("--- Phase 3: Model Training & Evaluation ---")
    
    split_data_path = os.path.join(models_dir, "split_data.npz")
    if not os.path.exists(split_data_path):
        raise FileNotFoundError(f"Split data not found at {split_data_path}. Please run preprocess.py first.")
        
    print(f"Loading split data from {split_data_path}...")
    data = np.load(split_data_path, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = list(data["columns"])
    
    print(f"Raw Train Shape: {X_train.shape}, Raw Test Shape: {X_test.shape}")
    print(f"Fraud count in train: {np.sum(y_train)}, Fraud count in test: {np.sum(y_test)}")
    
    # 1. Class Imbalance Handling using SMOTE
    print("\nApplying SMOTE (Synthetic Minority Over-sampling Technique) on training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled Train Shape: {X_train_res.shape}")
    print(f"Fraud count in resampled train: {np.sum(y_train_res)} (Legitimate count: {len(y_train_res) - np.sum(y_train_res)})")
    
    # 2. Model Training
    # Model A: Logistic Regression (Baseline)
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr_model.fit(X_train_res, y_train_res)
    print("Logistic Regression training completed.")
    
    # Model B: XGBoost Classifier (Advanced Model)
    # Using tree_method='hist' for fast training on CPU
    print("Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_res, y_train_res)
    print("XGBoost training completed.")
    
    # 3. Evaluation on Test Set
    print("\nEvaluating models on raw test set...")
    models = {
        "Logistic Regression": lr_model,
        "XGBoost": xgb_model
    }
    
    metrics_summary = {}
    curves_data = {}
    
    for name, model in models.items():
        print(f"\n--- {name} Results ---")
        
        # Predict classes and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        print(classification_report(y_test, y_pred))
        print(f"PR-AUC (Average Precision): {pr_auc:.5f}")
        print(f"ROC-AUC: {roc_auc:.5f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Feature Importances / Coefficients
        if name == "Logistic Regression":
            importances = np.abs(model.coef_[0])
        else:
            importances = model.feature_importances_
            
        # Normalize importances for consistent scaling
        importances = importances / np.sum(importances)
        
        feat_imp_dict = {
            feature_names[i]: float(importances[i])
            for i in range(len(feature_names))
        }
        # Sort features by importance
        sorted_feat_imp = dict(sorted(feat_imp_dict.items(), key=lambda item: item[1], reverse=True))
        
        metrics_summary[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "feature_importances": sorted_feat_imp
        }
        
        # Precision-Recall & ROC Curve Points for plotting
        prec_points, rec_points, thresholds_pr = precision_recall_curve(y_test, y_prob)
        fpr_points, tpr_points, thresholds_roc = roc_curve(y_test, y_prob)
        
        # Downsample points to make JSON smaller and plotting faster
        downsample_factor = max(1, len(prec_points) // 200)
        curves_data[name] = {
            "pr": {
                "precision": prec_points[::downsample_factor].tolist(),
                "recall": rec_points[::downsample_factor].tolist()
            },
            "roc": {
                "fpr": fpr_points[::downsample_factor].tolist(),
                "tpr": tpr_points[::downsample_factor].tolist()
            }
        }
        
    # Save Models using Pickle
    with open(os.path.join(models_dir, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr_model, f)
    with open(os.path.join(models_dir, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    print("\nSaved trained models in 'models/' directory.")
    
    # Save Metrics to JSON
    metrics_json_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved model metrics to: {metrics_json_path}")
    
    # 4. Generate Performance Plots
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # Plot 1: Precision-Recall Curves
    print("\nPlotting Precision-Recall Curves...")
    plt.figure(figsize=(10, 6))
    colors = {"Logistic Regression": "#a855f7", "XGBoost": "#10b981"}
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prec_pts, rec_pts, _ = precision_recall_curve(y_test, y_prob)
        pr_score = metrics_summary[name]["pr_auc"]
        plt.plot(rec_pts, prec_pts, label=f"{name} (PR-AUC = {pr_score:.4f})", color=colors[name], linewidth=2.5)
        
    plt.title("Precision-Recall Curve (Imbalanced Data Evaluation)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Recall (Sensitivity)", fontsize=12, labelpad=10)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, labelpad=10)
    plt.legend(facecolor='#1a1a24', edgecolor='#2c2c35')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plot_pr_path = os.path.join(plots_dir, "pr_curve.png")
    plt.savefig(plot_pr_path, dpi=300, facecolor='#0d0d12')
    plt.close()
    print(f"Saved PR Curve to: {plot_pr_path}")
    
    # Plot 2: Confusion Matrices Side-by-Side
    print("Plotting Confusion Matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Custom labeling
        group_names = ['True Neg (Legit)', 'False Pos (Alarm)', 'False Neg (Missed)', 'True Pos (Caught)']
        group_counts = [f"{value:d}" for value in cm.flatten()]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
        labels = np.asarray(labels).reshape(2, 2)
        
        cmap_colors = sns.light_palette(colors[name], as_cmap=True)
        
        sns.heatmap(cm, annot=labels, fmt="", cmap=cmap_colors, ax=axes[idx], cbar=False,
                    square=True, linewidths=1, annot_kws={"size": 11, "weight": "bold"})
        
        axes[idx].set_title(f"{name} Confusion Matrix", fontsize=13, fontweight='bold', pad=10)
        axes[idx].set_xlabel("Predicted Class", labelpad=10)
        axes[idx].set_ylabel("Actual Class", labelpad=10)
        axes[idx].set_xticklabels(['Legit', 'Fraud'])
        axes[idx].set_yticklabels(['Legit', 'Fraud'], rotation=0)
        
    plt.tight_layout()
    plot_cm_path = os.path.join(plots_dir, "confusion_matrices.png")
    plt.savefig(plot_cm_path, dpi=300, facecolor='#0d0d12')
    plt.close()
    print(f"Saved confusion matrices to: {plot_cm_path}")
    print("Model Training & Evaluation completed successfully!\n")

if __name__ == "__main__":
    train_and_evaluate()
