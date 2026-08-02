"""
XGBoost Model Structure Visualization
=====================================

Generates PNG images showing the structure of the most recently trained
XGBoost model (from xgboost/models/):

    results/xgb_ensemble_overview.png  - ensemble-level view: depth and
                                         size of every boosted tree
    results/xgb_tree_<i>_structure.png - one boosted tree drawn as a
                                         flowchart (split feature +
                                         threshold per node, output
                                         value per leaf)

The tree is rendered with plain matplotlib from the booster's own dump
(trees_to_dataframe) — deliberately NOT xgboost.plot_tree, which requires
the Graphviz system binary to be installed separately.

Unlike a Random Forest (trees vote independently), boosted trees are
sequential correction steps: leaf values are log-odds contributions that
are SUMMED across all trees, so early trees carry the coarse signal and
later trees encode refinements.

Usage (after training, from inside the xgboost folder):

    python visualize_model.py               # tree 0, full depth
    python visualize_model.py --tree 12     # a later (refinement) tree
    python visualize_model.py --depth 3     # truncate display to 3 levels
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed — write files only
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_latest_model():
    """Load the newest trained model (native .json format)."""
    candidates = sorted(glob.glob(
        os.path.join(MODELS_DIR, "xgb_vortex_detector_*.json")))
    if not candidates:
        raise FileNotFoundError(
            f"No trained model found in {MODELS_DIR}. "
            "Run 'python run.py' first to train one.")

    model_path = candidates[-1]
    model = XGBClassifier()
    model.load_model(model_path)
    print(f"Loaded model: {model_path}")
    return model

# =============================================================================
# ENSEMBLE OVERVIEW
# =============================================================================

def tree_depths(trees_df):
    """Compute the depth of every tree by walking Yes/No pointers."""
    depths = []
    for t, tdf in trees_df.groupby("Tree"):
        children = {}
        for _, row in tdf.iterrows():
            if row["Feature"] != "Leaf":
                children[row["ID"]] = (row["Yes"], row["No"])
        max_depth = 0
        stack = [(f"{t}-0", 0)]
        while stack:
            node_id, d = stack.pop()
            max_depth = max(max_depth, d)
            if node_id in children:
                yes_id, no_id = children[node_id]
                stack.append((yes_id, d + 1))
                stack.append((no_id, d + 1))
        depths.append(max_depth)
    return depths


def plot_ensemble_overview(trees_df, out_path):
    """Ensemble-level structure: depth and size of every boosted tree."""
    grouped = trees_df.groupby("Tree")
    n_nodes = grouped.size().values
    n_leaves = grouped.apply(
        lambda g: int((g["Feature"] == "Leaf").sum())).values
    depths = tree_depths(trees_df)
    n_trees = len(n_nodes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(n_trees), depths, color="#2b7a78", width=1.0)
    axes[0].axhline(np.mean(depths), color="#d64545", linestyle="--",
                    label=f"mean depth = {np.mean(depths):.1f}")
    axes[0].set_xlabel("Boosting round (tree index)")
    axes[0].set_ylabel("Depth")
    axes[0].set_title(f"Depth of each of the {n_trees} boosted trees")
    axes[0].legend()

    axes[1].bar(range(n_trees), n_nodes, color="#3aafa9", width=1.0,
                label="total nodes")
    axes[1].bar(range(n_trees), n_leaves, color="#17252a", width=1.0,
                label="leaves")
    axes[1].set_xlabel("Boosting round (tree index)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Size of each tree (nodes / leaves)")
    axes[1].legend()

    fig.suptitle("XGBoost — ensemble structure overview "
                 "(trees are sequential correction steps)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# =============================================================================
# SINGLE-TREE FLOWCHART (pure matplotlib, no Graphviz)
# =============================================================================

def plot_single_tree(trees_df, tree_idx, display_depth, out_path):
    """Draw one boosted tree as a flowchart."""
    tdf = trees_df[trees_df["Tree"] == tree_idx]
    if tdf.empty:
        raise ValueError(
            f"--tree must be in [0, {trees_df['Tree'].max()}]")

    nodes = {row["ID"]: row for _, row in tdf.iterrows()}
    root_id = f"{tree_idx}-0"

    # --- layout: leaves get consecutive x slots, parents sit centered ---
    positions = {}   # node_id -> (x, depth, kind) kind: 'split'|'leaf'|'cut'
    x_slot = [0]

    def layout(node_id, depth):
        node = nodes[node_id]
        is_leaf = node["Feature"] == "Leaf"
        truncated = (display_depth is not None and depth >= display_depth
                     and not is_leaf)
        if is_leaf or truncated:
            x = x_slot[0]
            x_slot[0] += 1
            positions[node_id] = (x, depth, "leaf" if is_leaf else "cut")
            return x
        xl = layout(node["Yes"], depth + 1)
        xr = layout(node["No"], depth + 1)
        x = (xl + xr) / 2.0
        positions[node_id] = (x, depth, "split")
        return x

    layout(root_id, 0)

    n_slots = max(x_slot[0], 2)
    max_d = max(d for _, d, _ in positions.values())
    fig_w = max(10, min(1.9 * n_slots, 60))
    fig_h = max(6, 2.2 * (max_d + 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.7, n_slots - 0.3)
    ax.set_ylim(-max_d - 0.6, 0.6)
    ax.axis("off")

    def draw_box(x, y, text, facecolor):
        ax.add_patch(FancyBboxPatch(
            (x - 0.42, y - 0.28), 0.84, 0.56,
            boxstyle="round,pad=0.02",
            facecolor=facecolor, edgecolor="#333333", linewidth=0.8))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5)

    # edges first (under the boxes)
    for node_id, (x, d, kind) in positions.items():
        if kind != "split":
            continue
        node = nodes[node_id]
        for child_id, edge_label in [(node["Yes"], "yes"), (node["No"], "no")]:
            cx, cd, _ = positions[child_id]
            ax.plot([x, cx], [-d - 0.28, -cd + 0.28],
                    color="#888888", linewidth=0.9, zorder=1)
            ax.text((x + cx) / 2, (-d - 0.28 + -cd + 0.28) / 2, edge_label,
                    fontsize=6.5, color="#555555",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1",
                              facecolor="white", edgecolor="none"))

    # boxes
    for node_id, (x, d, kind) in positions.items():
        node = nodes[node_id]
        if kind == "split":
            text = (f"{node['Feature']}\n< {node['Split']:.4g}\n"
                    f"gain {node['Gain']:.3g}")
            draw_box(x, -d, text, "#cfe8ff")
        elif kind == "leaf":
            value = node["Gain"]  # for leaves, 'Gain' holds the output value
            color = "#ffd6d6" if value > 0 else "#d6f5d6"
            direction = "vortex" if value > 0 else "no vortex"
            draw_box(x, -d, f"leaf\n{value:+.4f}\n({direction})", color)
        else:  # truncated subtree
            draw_box(x, -d, "subtree\n(depth\ntruncated)", "#eeeeee")

    depth_note = ("full depth" if display_depth is None
                  else f"top {display_depth} levels shown")
    n_total = int(trees_df["Tree"].max()) + 1
    ax.set_title(
        f"XGBoost — tree {tree_idx} of {n_total} ({depth_note})\n"
        "Leaf values are log-odds contributions summed across all trees; "
        "positive pushes toward 'vortex'.",
        fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# =============================================================================
# MAIN
# =============================================================================

def generate(tree_idx=0, depth=None, model=None):
    """
    Generate all structure PNGs.

    Called automatically by run.py after training (which passes the
    freshly trained model in memory), or via the CLI below (which loads
    the latest saved model from models/).
    """
    if model is None:
        model = load_latest_model()

    trees_df = model.get_booster().trees_to_dataframe()
    n_trees = int(trees_df["Tree"].max()) + 1
    print(f"  Boosted trees in model: {n_trees}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nGenerating structure visualizations...")
    plot_ensemble_overview(
        trees_df, os.path.join(RESULTS_DIR, "xgb_ensemble_overview.png"))
    plot_single_tree(
        trees_df, tree_idx, depth,
        os.path.join(RESULTS_DIR, f"xgb_tree_{tree_idx}_structure.png"))

    print("\nDone. Open the PNGs in xgboost/results/ to inspect the model.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PNGs of the trained XGBoost model's structure.")
    parser.add_argument("--tree", type=int, default=0,
                        help="index of the boosted tree to draw (default 0)")
    parser.add_argument("--depth", type=int, default=None,
                        help="levels to display (default: full tree)")
    args = parser.parse_args()
    generate(tree_idx=args.tree, depth=args.depth)


if __name__ == "__main__":
    main()
