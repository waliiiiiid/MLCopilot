import os
import numpy as np
import matplotlib.pyplot as plt
from backend.agents.workflows.ml_state import ml_state

def compute_real_vs_predicted(y_real, y_pred):

    # Convert to numeric arrays
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Make sure output directory exists
    output_dir = "outputs/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 6))

    # Scatter plot
    plt.scatter(
        y_real,
        y_pred,
        alpha=0.7
    )

    # Perfect prediction line
    min_value = min(y_real.min(), y_pred.min())
    max_value = max(y_real.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "--",
        label="Perfect prediction"
    )

    plt.xlabel("Real Values")
    plt.ylabel("Predicted Values")
    plt.title("Real vs Predicted")
    plt.legend()
    plt.grid(True)

    # Save plot
    output_path = os.path.join(
        output_dir,
        "real_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "plot_path": output_path
    }


def real_vs_predicted(state: ml_state):

    y_real = state["y_real"]
    y_pred = state["y_pred"]

    return compute_real_vs_predicted(
        y_real,
        y_pred
    )