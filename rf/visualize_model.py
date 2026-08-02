"""
Random Forest Model Structure Visualization
===========================================

Generates PNG images showing the structure of the most recently trained
Random Forest model (from rf/models/):

    results/rf_forest_overview.png   - ensemble-level view: depth and size
                                       of every tree in the forest
    results/rf_tree_<i>_structure.png - one full decision tree drawn as a
                                       flowchart (splits, thresholds,
                                       class balance per node)

A forest of 100 trees cannot be drawn in one readable image, so the tree
plot shows ONE tree (default: the first), truncated to a readable depth
(default: 3 levels — the full trees go 15 deep).

Usage (after training, from inside the rf folder):

    python visualize_model.py                 # tree 0, display depth 3
    python visualize_model.py --tree 7        # a different tree
    python visualize_model.py --depth 4       # show more levels
"""

import argparse
import glob
import json
import os

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed — write files only
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_latest_model():
    """Load the newest trained model and its feature names."""
    candidates = sorted(glob.glob(
        os.path.join(MODELS_DIR, "rf_vortex_detector_*.pkl")))
    if not candidates:
        raise FileNotFoundError(
            f"No trained model found in {MODELS_DIR}. "
            "Run 'python run.py' first to train one.")

    model_path = candidates[-1]
    model = joblib.load(model_path)

    timestamp = os.path.basename(model_path).replace(
        "rf_vortex_detector_", "").replace(".pkl", "")
    feature_names = None
    metadata_path = os.path.join(MODELS_DIR, f"model_metadata_{timestamp}.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path) as f:
            feature_names = json.load(f).get("feature_names")

    print(f"Loaded model: {model_path}")
    print(f"  Trees in forest: {len(model.estimators_)}")
    return model, feature_names


def plot_forest_overview(model, out_path):
    """Ensemble-level structure: depth and node count of every tree."""
    depths = [est.get_depth() for est in model.estimators_]
    n_nodes = [est.tree_.node_count for est in model.estimators_]
    n_leaves = [est.get_n_leaves() for est in model.estimators_]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(depths)), depths, color="#2b7a78", width=1.0)
    axes[0].axhline(np.mean(depths), color="#d64545", linestyle="--",
                    label=f"mean depth = {np.mean(depths):.1f}")
    axes[0].set_xlabel("Tree index")
    axes[0].set_ylabel("Depth")
    axes[0].set_title(f"Depth of each of the {len(depths)} trees")
    axes[0].legend()

    axes[1].bar(range(len(n_nodes)), n_nodes, color="#3aafa9", width=1.0,
                label="total nodes")
    axes[1].bar(range(len(n_leaves)), n_leaves, color="#17252a", width=1.0,
                label="leaves")
    axes[1].set_xlabel("Tree index")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Size of each tree (nodes / leaves)")
    axes[1].legend()

    fig.suptitle("Random Forest — ensemble structure overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_single_tree(model, feature_names, tree_idx, display_depth, out_path):
    """One decision tree drawn as a flowchart, truncated for readability."""
    if not (0 <= tree_idx < len(model.estimators_)):
        raise ValueError(f"--tree must be in [0, {len(model.estimators_) - 1}]")

    estimator = model.estimators_[tree_idx]

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=["No vortex", "Vortex"],
        filled=True,          # color by majority class / purity
        rounded=True,
        impurity=True,
        proportion=True,      # show class fractions, not raw counts
        max_depth=display_depth,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(
        f"Random Forest — tree {tree_idx} of {len(model.estimators_)} "
        f"(showing top {display_depth} of {estimator.get_depth()} levels)",
        fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate(tree_idx=0, depth=3):
    """
    Generate all structure PNGs from the latest trained model.

    Called automatically by run.py after training, or via the CLI below.
    """
    model, feature_names = load_latest_model()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nGenerating structure visualizations...")
    plot_forest_overview(
        model, os.path.join(RESULTS_DIR, "rf_forest_overview.png"))
    plot_single_tree(
        model, feature_names, tree_idx, depth,
        os.path.join(RESULTS_DIR, f"rf_tree_{tree_idx}_structure.png"))

    print("\nDone. Open the PNGs in rf/results/ to inspect the model.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PNGs of the trained Random Forest's structure.")
    parser.add_argument("--tree", type=int, default=0,
                        help="index of the tree to draw (default 0)")
    parser.add_argument("--depth", type=int, default=3,
                        help="number of levels to display (default 3)")
    args = parser.parse_args()
    generate(tree_idx=args.tree, depth=args.depth)


if __name__ == "__main__":
    main()
