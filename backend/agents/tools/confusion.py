import ast
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from backend.agents.workflows.ml_state import ml_state


def plot_confusion_matrix(y_real, y_pred):
    """
    Compute and plot a confusion matrix.

    Accepts either actual Python lists/arrays or strings
    representing lists.
    """

    # Make sure output directory exists
    output_dir = "outputs/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    # Convert string representations of lists into actual lists
    if isinstance(y_real, str):
        y_real = ast.literal_eval(y_real)

    if isinstance(y_pred, str):
        y_pred = ast.literal_eval(y_pred)

    # Convert to numpy arrays
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    # Validate lengths
    if len(y_real) != len(y_pred):
        raise ValueError(
            f"y_real and y_pred must have the same length. "
            f"Got {len(y_real)} and {len(y_pred)}."
        )

    # Compute confusion matrix
    cm = confusion_matrix(y_real, y_pred)

    # Create confusion matrix plot
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(
        output_dir,
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Return JSON-friendly result
    return {
        "confusion_matrix": cm.tolist(),
        "plot_path": output_path
    }


def confusion_matrix_node(state: ml_state):
    """
    LangGraph node for computing the confusion matrix.
    """

    y_real = state["y_real"]
    y_pred = state["y_pred"]

    return plot_confusion_matrix(
        y_real,
        y_pred
    )