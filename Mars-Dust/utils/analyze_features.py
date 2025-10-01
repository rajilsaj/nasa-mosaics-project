import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from typing import List, Dict, Union, Optional

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
    """
    Perform comprehensive feature importance analysis for a Random Forest model.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        The trained Random Forest model
    feature_names : List[str]
        Names of the features
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        True labels
    save_dir : Union[str, Path]
        Directory to save the analysis results
    model_name : str
        Name of the model for plot titles
    do_permutation : bool
        Whether to perform permutation importance analysis (can be computationally expensive)
    n_permutations : int
        Number of permutations for permutation importance
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    Dict
        Dictionary containing all feature importance metrics
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style for all plots
    plt.style.use('default')
    
    # 1. Gini Importance
    gini_importance = model.feature_importances_
    
    # 2. Feature Correlations
    feature_correlations = np.corrcoef(X_test.T)
    
    # Create results dictionary
    results = {
        'feature_names': feature_names,
        'gini_importance': gini_importance,
        'feature_correlations': feature_correlations
    }
    
    # Visualizations
    
    # 1. Gini Importance Plot
    plt.figure(figsize=(12, 6))
    gini_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': gini_importance
    }).sort_values('Importance', ascending=True)
    
    plt.barh(range(len(gini_df)), gini_df['Importance'])
    plt.yticks(range(len(gini_df)), gini_df['Feature'])
    plt.title(f'Gini Feature Importance - {model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(save_dir / 'gini_importance.png')
    plt.close()
    
    # 2. Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(feature_correlations, 
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f')
    plt.title(f'Feature Correlations - {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_correlations.png')
    plt.close()
    
    # 3. Permutation Importance (if requested)
    if do_permutation:
        print("\nCalculating permutation importance (this may take a while)...")
        base_score = model.score(X_test, y_test)
        permutation_importance = np.zeros((len(feature_names), n_permutations))
        
        for i in range(n_permutations):
            for j, feature in enumerate(feature_names):
                X_test_permuted = X_test.copy()
                np.random.seed(random_state + i)
                np.random.shuffle(X_test_permuted[:, j])
                score = model.score(X_test_permuted, y_test)
                permutation_importance[j, i] = base_score - score
        
        mean_perm_importance = np.mean(permutation_importance, axis=1)
        std_perm_importance = np.std(permutation_importance, axis=1)
        
        # Add to results dictionary
        results['permutation_importance'] = mean_perm_importance
        results['permutation_std'] = std_perm_importance
        
        # Create permutation importance plot
        plt.figure(figsize=(12, 6))
        perm_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_perm_importance,
            'Std': std_perm_importance
        }).sort_values('Importance', ascending=True)
        
        plt.barh(range(len(perm_df)), perm_df['Importance'], xerr=perm_df['Std'])
        plt.yticks(range(len(perm_df)), perm_df['Feature'])
        plt.title(f'Permutation Feature Importance - {model_name}')
        plt.xlabel('Importance (Score Decrease)')
        plt.tight_layout()
        plt.savefig(save_dir / 'permutation_importance.png')
        plt.close()
        
        # Create combined importance plot
        plt.figure(figsize=(12, 6))
        combined_df = pd.DataFrame({
            'Feature': feature_names,
            'Gini': gini_importance,
            'Permutation': mean_perm_importance
        }).sort_values('Gini', ascending=True)
        
        x = np.arange(len(combined_df))
        width = 0.35
        
        plt.barh(x - width/2, combined_df['Gini'], width, label='Gini')
        plt.barh(x + width/2, combined_df['Permutation'], width, label='Permutation')
        plt.yticks(x, combined_df['Feature'])
        plt.title(f'Combined Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / 'combined_importance.png')
        plt.close()
    
    # Save numerical results to CSV
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Gini_Importance': gini_importance
    })
    if do_permutation:
        results_df['Permutation_Importance'] = mean_perm_importance
        results_df['Permutation_Std'] = std_perm_importance
    results_df.to_csv(save_dir / 'feature_importance_metrics.csv', index=False)
    
    # Create HTML report
    html_content = f"""
    <html>
    <head>
        <title>Feature Importance Analysis - {model_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .visualization {{ margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
            .feature-table {{ overflow-x: auto; }}
            .note {{ 
                background-color: #fff3cd;
                border: 1px solid #ffeeba;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Feature Importance Analysis - {model_name}</h1>
            
            {f'''
            <div class="note">
                <h3>Note: Permutation Importance Analysis</h3>
                <p>Permutation importance analysis has been temporarily disabled due to computational constraints. 
                This analysis would provide additional insights by measuring how much model performance decreases 
                when each feature is randomly shuffled. It's particularly useful for:</p>
                <ul>
                    <li>Understanding feature interactions</li>
                    <li>Measuring actual impact on model performance</li>
                    <li>Identifying redundant features</li>
                </ul>
                <p>This analysis will be implemented in a future update when computational resources allow.</p>
            </div>
            ''' if not do_permutation else ''}
            
            <h2>Feature Importance Metrics</h2>
            <div class="feature-table">
                {results_df.to_html(index=False)}
            </div>
            
            <h2>Visualizations</h2>
            <div class="visualization">
                <h3>Gini Feature Importance</h3>
                <img src="gini_importance.png" alt="Gini Importance">
            </div>
            <div class="visualization">
                <h3>Feature Correlations</h3>
                <img src="feature_correlations.png" alt="Feature Correlations">
            </div>
            {f'''
            <div class="visualization">
                <h3>Permutation Feature Importance</h3>
                <img src="permutation_importance.png" alt="Permutation Importance">
            </div>
            <div class="visualization">
                <h3>Combined Importance Comparison</h3>
                <img src="combined_importance.png" alt="Combined Importance">
            </div>
            ''' if do_permutation else ''}
        </div>
    </body>
    </html>
    """
    
    with open(save_dir / 'feature_importance_report.html', 'w') as f:
        f.write(html_content)
    
    print(f"Feature importance analysis saved to {save_dir}")
    return results 