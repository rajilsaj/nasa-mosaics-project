import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from typing import List, Dict, Union
import os


def analyze_feature_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Union[str, Path],
    model_name: str,
    do_permutation: bool = False,
    n_permutations: int = 100,
    random_state: int = 42
) -> Dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # === Gini Importance ===
    gini_importance = model.feature_importances_
    
    # === Feature Correlations ===
    feature_correlations = np.corrcoef(X_test.T)
    
    # === Gini Importance Plot ===
    gini_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': gini_importance
    }).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(12, 6))
    plt.barh(gini_df["Feature"], gini_df["Importance"], color='skyblue')
    plt.title(f"Gini Feature Importance - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_dir / "gini_importance.png")
    plt.close()

    # === Feature Correlation Heatmap ===
    plt.figure(figsize=(10, 8))
    sns.heatmap(feature_correlations, 
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt=".2f")
    plt.title(f"Feature Correlations - {model_name}")
    plt.tight_layout()
    plt.savefig(save_dir / "feature_correlations.png")
    plt.close()

    results = {
        'feature_names': feature_names,
        'gini_importance': gini_importance,
        'feature_correlations': feature_correlations
    }

    if do_permutation:
        print("Calculating permutation importance...")
        base_score = model.score(X_test, y_test)
        permutation_scores = np.zeros((len(feature_names), n_permutations))

        for i in range(n_permutations):
            for j in range(len(feature_names)):
                X_test_perm = X_test.copy()
                np.random.seed(random_state + i)
                np.random.shuffle(X_test_perm[:, j])
                perm_score = model.score(X_test_perm, y_test)
                permutation_scores[j, i] = base_score - perm_score

        mean_perm = np.mean(permutation_scores, axis=1)
        std_perm = np.std(permutation_scores, axis=1)

        # Plot permutation importance
        plt.figure(figsize=(12, 6))
        perm_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": mean_perm,
            "Std": std_perm
        }).sort_values(by="Importance", ascending=True)

        plt.barh(perm_df["Feature"], perm_df["Importance"], xerr=perm_df["Std"], color="orange")
        plt.title(f"Permutation Importance - {model_name}")
        plt.xlabel("Score Drop")
        plt.tight_layout()
        plt.savefig(save_dir / "permutation_importance.png")
        plt.close()

        results["permutation_importance"] = mean_perm
        results["permutation_std"] = std_perm

    # === Save Metrics CSV ===
    metrics_df = pd.DataFrame({
        "Feature": feature_names,
        "Gini_Importance": gini_importance
    })

    if do_permutation:
        metrics_df["Permutation_Importance"] = mean_perm
        metrics_df["Permutation_Std"] = std_perm

    metrics_df.to_csv(save_dir / "feature_importance_metrics.csv", index=False)

    # === Optional HTML Report ===
    html_path = save_dir / "feature_importance_report.html"
    with open(html_path, "w") as f:
        f.write(f"<html><body><h1>Feature Importance - {model_name}</h1>")
        f.write(metrics_df.to_html(index=False))
        f.write("<br><img src='gini_importance.png'><br>")
        f.write("<img src='feature_correlations.png'><br>")
        if do_permutation:
            f.write("<img src='permutation_importance.png'><br>")
        f.write("</body></html>")

    print(f" Feature importance results saved to {save_dir}")
    return results
