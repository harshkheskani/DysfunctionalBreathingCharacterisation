import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

def demonstrate_scale_effect():
    """
    Create a simple example showing how scale affects F-scores
    """
    print("\n" + "="*60)
    print("🧪 SCALE EFFECT DEMONSTRATION")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create two features that are equally discriminative
    # Feature 1: small scale (0-1 range)
    feature1_class0 = np.random.normal(0.3, 0.1, n_samples//2)
    feature1_class1 = np.random.normal(0.7, 0.1, n_samples//2)
    feature1 = np.concatenate([feature1_class0, feature1_class1])
    
    # Feature 2: large scale (0-1000 range) - same relative difference
    feature2_class0 = np.random.normal(300, 100, n_samples//2)
    feature2_class1 = np.random.normal(700, 100, n_samples//2)
    feature2 = np.concatenate([feature2_class0, feature2_class1])
    
    # Labels
    y = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Test without standardization
    X_raw = np.column_stack([feature1, feature2])
    f_raw, p_raw = f_classif(X_raw, y)
    
    # Test with standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    f_scaled, p_scaled = f_classif(X_scaled, y)
    
    print("Feature 1 (small scale, 0-1 range):")
    print(f"  Without standardization: F-score = {f_raw[0]:.2f}")
    print(f"  With standardization:    F-score = {f_scaled[0]:.2f}")
    print("\nFeature 2 (large scale, 0-1000 range):")
    print(f"  Without standardization: F-score = {f_raw[1]:.2f}")
    print(f"  With standardization:    F-score = {f_scaled[1]:.2f}")
    print(f"\n💡 Key Insight:")
    print(f"  Without standardization: Feature 2 appears {f_raw[1]/f_raw[0]:.1f}x more important!")
    print(f"  With standardization:    Features are nearly equal ({f_scaled[1]/f_scaled[0]:.2f}x)")
    print("="*60 + "\n")

def analyze_multiclass_features(feature_dir, label_map, top_n=12, p_threshold=0.05, 
                              min_f_score=1.0, standardize=True, show_demo=True):
    """
    Enhanced version that finds and visualizes the top features for distinguishing 
    between all ground truth classes, with statistical significance filtering 
    and standardization options.
    
    Parameters:
    -----------
    feature_dir : str
        Path to directory containing Parquet files
    label_map : dict
        Mapping from integer labels to string names
    top_n : int
        Number of top features to plot
    p_threshold : float
        P-value threshold for statistical significance
    min_f_score : float
        Minimum F-score threshold for effect size
    standardize : bool
        Whether to standardize features before analysis
    show_demo : bool
        Whether to show the scale effect demonstration
    """
    
    if show_demo:
        demonstrate_scale_effect()
    
    print(f"🚀 Loading feature files from: {feature_dir}")
    df = pd.concat((pd.read_parquet(f) for f in glob.glob(os.path.join(feature_dir, '*_features.parquet'))), ignore_index=True)
    print(f"✅ Loaded {len(df)} windows.")

    # --- Step 1: Map Ground Truth Labels for Multi-Class Analysis ---
    print("\n🔬 Performing ENHANCED MULTI-CLASS statistical analysis...")
    
    if 'label' not in df.columns:
        print("Error: 'label' column (ground truth) not found in the Parquet files.")
        return
        
    df['ground_truth_event'] = df['label'].map(label_map)
    df.dropna(subset=['ground_truth_event'], inplace=True)

    print("\n📊 Ground Truth Label Distribution:")
    label_counts = df['ground_truth_event'].value_counts()
    label_props = df['ground_truth_event'].value_counts(normalize=True)
    for event, count in label_counts.items():
        prop = label_props[event]
        print(f"  {event}: {count:,} samples ({prop:.1%})")

    # --- Step 2: Prepare Features ---
    feature_columns = [col for col in df.columns if col not in ['session_id', 'label', 'ground_truth_event']]
    X = df[feature_columns].fillna(0)
    y_multiclass = df['ground_truth_event']
    
    print(f"\n📏 Feature preparation:")
    print(f"  Total features: {len(feature_columns)}")
    print(f"  Samples: {len(X)}")
    
    # Show feature scale statistics before standardization
    scale_stats = pd.DataFrame({
        'feature': feature_columns,
        'mean': X.mean(),
        'std': X.std(),
        'min': X.min(),
        'max': X.max(),
        'range': X.max() - X.min()
    }).sort_values(by='range', ascending=False)
    
    print(f"\n📈 Feature Scale Analysis (Top 10 by range):")
    print(scale_stats.head(10)[['feature', 'mean', 'std', 'range']].to_string(index=False))
    
    # Identify potential scale issues
    large_range_features = scale_stats[scale_stats['range'] > scale_stats['range'].quantile(0.9)]
    print(f"\n⚠️  Features with potentially problematic scales (top 10% by range): {len(large_range_features)}")
    
    # --- Step 3: Run Analysis (with and without standardization if requested) ---
    results = {}
    
    # Analysis without standardization
    print(f"\n🔬 Analysis WITHOUT standardization:")
    f_scores_raw, p_values_raw = f_classif(X, y_multiclass)
    results['raw'] = pd.DataFrame({
        'feature': feature_columns,
        'f_score': f_scores_raw,
        'p_value': p_values_raw,
        'significant': p_values_raw < p_threshold,
        'high_effect': f_scores_raw > min_f_score
    }).sort_values(by='f_score', ascending=False).reset_index(drop=True)
    
    print("Top 10 features (raw scale):")
    display_raw = results['raw'].head(10)[['feature', 'f_score', 'p_value']]
    display_raw['p_formatted'] = display_raw['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    print(display_raw[['feature', 'f_score', 'p_formatted']].to_string(index=False))
    
    # Analysis with standardization
    if standardize:
        print(f"\n🔬 Analysis WITH standardization:")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
        
        # Create standardized version of the full dataframe for plotting
        df_scaled = df.copy()
        df_scaled[feature_columns] = X_scaled
        
        f_scores_scaled, p_values_scaled = f_classif(X_scaled_df, y_multiclass)
        results['standardized'] = pd.DataFrame({
            'feature': feature_columns,
            'f_score': f_scores_scaled,
            'p_value': p_values_scaled,
            'significant': p_values_scaled < p_threshold,
            'high_effect': f_scores_scaled > min_f_score
        }).sort_values(by='f_score', ascending=False).reset_index(drop=True)
        
        print("Top 10 features (standardized):")
        display_std = results['standardized'].head(10)[['feature', 'f_score', 'p_value']]
        display_std['p_formatted'] = display_std['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
        print(display_std[['feature', 'f_score', 'p_formatted']].to_string(index=False))
        
        # Compare rankings
        print(f"\n📊 Ranking Comparison (Raw vs Standardized):")
        raw_ranks = {feat: i+1 for i, feat in enumerate(results['raw']['feature'])}
        std_ranks = {feat: i+1 for i, feat in enumerate(results['standardized']['feature'])}
        
        comparison = pd.DataFrame({
            'feature': feature_columns,
            'raw_rank': [raw_ranks[feat] for feat in feature_columns],
            'std_rank': [std_ranks[feat] for feat in feature_columns]
        })
        comparison['rank_change'] = comparison['raw_rank'] - comparison['std_rank']
        comparison['abs_change'] = comparison['rank_change'].abs()
        
        biggest_changes = comparison.sort_values('abs_change', ascending=False).head(10)
        print("Features with biggest ranking changes:")
        print(biggest_changes[['feature', 'raw_rank', 'std_rank', 'rank_change']].to_string(index=False))
    
    # --- Step 4: Apply Statistical Filters ---
    analysis_key = 'standardized' if standardize else 'raw'
    main_results = results[analysis_key]
    
    # Filter for statistical significance
    significant_features = main_results[main_results['significant']]
    high_effect_features = main_results[main_results['high_effect']]
    both_criteria = main_results[main_results['significant'] & main_results['high_effect']]
    
    # Apply Bonferroni correction
    bonferroni_threshold = p_threshold / len(feature_columns)
    bonferroni_significant = main_results[main_results['p_value'] < bonferroni_threshold]
    
    print(f"\n📊 Statistical Filtering Results:")
    print(f"  • Total features: {len(main_results)}")
    print(f"  • Statistically significant (p < {p_threshold}): {len(significant_features)}")
    print(f"  • High effect size (F > {min_f_score}): {len(high_effect_features)}")
    print(f"  • Both significant AND high effect: {len(both_criteria)}")
    print(f"  • Bonferroni corrected (p < {bonferroni_threshold:.2e}): {len(bonferroni_significant)}")
    
    # Choose features for plotting (prefer both criteria, fall back to significant, then top overall)
    if len(both_criteria) >= top_n:
        features_to_plot = both_criteria
        plot_title_suffix = "Significant & High Effect"
    elif len(significant_features) >= top_n:
        features_to_plot = significant_features
        plot_title_suffix = "Statistically Significant"
    else:
        features_to_plot = main_results
        plot_title_suffix = "Top Features (No Filtering Applied)"
    
    print(f"\n🏆 Features Selected for Plotting ({plot_title_suffix}):")
    display_features = features_to_plot.head(15)[['feature', 'f_score', 'p_value']]
    display_features['p_formatted'] = display_features['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    print(display_features[['feature', 'f_score', 'p_formatted']].to_string(index=False))

    # --- Step 5: Save Results ---
    output_dir = os.path.dirname(feature_dir)
    
    # Save main results
    main_output_path = os.path.join(output_dir, f'feature_importance_multiclass_{"standardized" if standardize else "raw"}.csv')
    main_results.to_csv(main_output_path, index=False)
    print(f"\n✅ Feature importance scores saved to: {main_output_path}")
    
    # Save filtered results
    if len(both_criteria) > 0:
        filtered_output_path = os.path.join(output_dir, f'feature_importance_filtered_{"standardized" if standardize else "raw"}.csv')
        both_criteria.to_csv(filtered_output_path, index=False)
        print(f"✅ Filtered features (significant & high effect) saved to: {filtered_output_path}")

    # --- Step 6: Visualization ---
    top_features_to_plot = features_to_plot['feature'].head(top_n).tolist()
    print(f"\n🎨 Generating plots for the top {len(top_features_to_plot)} features...")
    
    num_plots = len(top_features_to_plot)
    ncols = 3
    nrows = (num_plots + ncols - 1) // ncols
    figsize = (18, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    standardization_text = " (Standardized)" if standardize else " (Raw Scale)"
    fig.suptitle(f'Top {num_plots} Features - {plot_title_suffix}{standardization_text}', 
                fontsize=16, y=1.02)
    axes = axes.flatten()

    # Define consistent order and palette for all classes
    event_order_for_plot = [name for name in label_map.values() if name in y_multiclass.unique()]
    
    for i, feature in enumerate(top_features_to_plot):
        ax = axes[i]
        feature_info = features_to_plot[features_to_plot['feature'] == feature].iloc[0]
        f_score = feature_info['f_score']
        p_value = feature_info['p_value']
        
        # Use appropriate data for plotting
        if standardize:
            plot_data = df_scaled
            ylabel_suffix = ' (Standardized)'
        else:
            plot_data = df
            ylabel_suffix = ''
        
        sns.violinplot(
            data=plot_data,
            x='ground_truth_event',
            y=feature,
            hue='ground_truth_event',
            order=event_order_for_plot,
            ax=ax,
            legend=False
        )
        
        p_text = f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}"
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        ax.set_title(f'{i+1}. {feature.replace("_", " ").title()}\n(F={f_score:.2f}, p={p_text}{significance})')
        ax.set_xlabel('')
        ax.set_ylabel('Feature Value' + ylabel_suffix)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == '__main__':
    # --- Configuration ---
    LOCAL_FEATURE_DIR = './features-multi'

    # Label mapping - UPDATE THIS to match your LabelEncoder
    GROUND_TRUTH_LABEL_MAP = {
        0: 'Normal',
        1: 'Obstructive Apnea',
        2: 'Hypopnea',
        3: 'Central/Mixed Apnea',
        4: 'Desaturation'
    }
    
    # Analysis parameters
    CONFIG = {
        'top_n': 9,              # Number of features to plot
        'p_threshold': 0.05,     # P-value threshold for significance
        'min_f_score': 2.0,      # Minimum F-score for effect size
        'standardize': True,     # Whether to standardize features
        'show_demo': True        # Whether to show scale effect demonstration
    }

    if os.path.isdir(LOCAL_FEATURE_DIR):
        print("🔬 Starting Enhanced Multi-Class Feature Analysis")
        print("="*60)
        
        results = analyze_multiclass_features(
            LOCAL_FEATURE_DIR,
            label_map=GROUND_TRUTH_LABEL_MAP,
            **CONFIG
        )
        
        print("\n✅ Analysis complete!")
        print("📁 Check the output directory for saved CSV files.")
        
    else:
        print(f"❌ Error: Directory not found at '{LOCAL_FEATURE_DIR}'.")
        print("Please update the LOCAL_FEATURE_DIR variable.")