import ast
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from backend.agents.workflows.ml_state import ml_state


def compute_roc_auc(y_real, y_proba):
    """
    Compute ROC curve and ROC-AUC for binary classification.

    Accepts Python lists, NumPy arrays, or strings representing lists.
    """

    # Create output directory
    output_dir = "outputs/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    # Convert string representations of lists
    if isinstance(y_real, str):
        y_real = ast.literal_eval(y_real)

    if isinstance(y_proba, str):
        y_proba = ast.literal_eval(y_proba)

    # Convert to NumPy arrays
    y_real = np.asarray(y_real, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    # Validate lengths
    if len(y_real) != len(y_proba):
        raise ValueError(
            f"y_real and y_proba must have the same length. "
            f"Got {len(y_real)} and {len(y_proba)}."
        )

    # Validate binary classification
    unique_classes = np.unique(y_real)

    if len(unique_classes) != 2:
        raise ValueError(
            "ROC-AUC in this function requires binary classification. "
            f"Found classes: {unique_classes.tolist()}"
        )

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(
        y_real,
        y_proba
    )

    # Calculate AUC
    auc_score = roc_auc_score(
        y_real,
        y_proba
    )

    # Create ROC plot
    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC curve (AUC = {auc_score:.3f})"
    )

    # Random classifier baseline
    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save plot
    output_path = os.path.join(
        output_dir,
        "roc_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Return JSON-friendly result
    return {
        "auc": float(auc_score),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "plot_path": output_path
    }


def roc_auc_node(state: ml_state):
    """
    LangGraph node for ROC-AUC evaluation.
    """

    y_real = state["y_real"]
    y_proba = state["y_proba"]

    return compute_roc_auc(
        y_real,
        y_proba
    )