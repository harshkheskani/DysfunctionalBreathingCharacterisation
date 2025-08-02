import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif

def analyze_oa_signature(feature_dir, label_map, top_n=9):
    """
    Finds and visualizes the top features for distinguishing Obstructive Apnea.
    """
    print(f"🚀 Loading feature files from: {feature_dir}")
    df = pd.concat((pd.read_parquet(f) for f in glob.glob(os.path.join(feature_dir, '*_features.parquet'))), ignore_index=True)
    print(f"✅ Loaded {len(df)} windows.")

    # --- Step 1: Create Binary Label and Run Statistical Analysis ---
    print("\n🔬 Performing binary statistical analysis for Obstructive Apnea vs. Not OA...")
    
    # Map integer labels to names
    df['ground_truth_event'] = df['label'].map(label_map)
    df.dropna(subset=['ground_truth_event'], inplace=True)

    # Create the binary target: 1 for OA, 0 for everything else
    df['is_oa'] = (df['ground_truth_event'] == 'Obstructive Apnea').astype(int)
    
    feature_columns = [col for col in df.columns if col not in ['session_id', 'label', 'ground_truth_event', 'is_oa']]
    X = df[feature_columns].fillna(0)
    y_binary = df['is_oa']
    
    if y_binary.sum() == 0:
        print("Error: No 'Obstructive Apnea' samples found. Cannot perform analysis.")
        return

    f_scores, p_values = f_classif(X, y_binary)
    
    oa_feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'f_score': f_scores,
        'p_value': p_values
    }).sort_values(by='f_score', ascending=False).reset_index(drop=True)

    output_csv_path = os.path.join(os.path.dirname(feature_dir), 'feature_importance_binary_oa.csv')
    oa_feature_importance.to_csv(output_csv_path, index=False)
    print(f"✅ Binary OA feature importance scores saved to: {output_csv_path}")

    print("\n🏆 Top Differentiating Features for Obstructive Apnea (vs. All Else):")
    print(oa_feature_importance.head(15))

    # --- Step 2: Focused Visualization ---
    top_features_to_plot = oa_feature_importance['feature'].head(top_n).tolist()
    print(f"\n🎨 Generating focused plots for the top {len(top_features_to_plot)} features...")
    
    # ========================================================== #
    # === FIXES ARE HERE === #
    # ========================================================== #

    # 1. Define the event types you want to see in the plots
    #    Make sure these names EXACTLY match the values in your GROUND_TRUTH_LABEL_MAP
    event_order_for_plot = ['Normal', 'Hypopnea', 'Obstructive Apnea', 'Central/Mixed Apnea', 'Desaturation']
    
    # 2. Filter the DataFrame to include only these event types
    plot_df = df[df['ground_truth_event'].isin(event_order_for_plot)]

    num_plots = len(top_features_to_plot)
    ncols = 3
    nrows = (num_plots + ncols - 1) // ncols
    figsize = (18, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle('Top Feature Signatures for Differentiating Obstructive Apnea (Respeck Data)', fontsize=16, y=1.02)
    axes = axes.flatten()
    
    # 3. Create a palette dictionary that includes ALL the names in event_order_for_plot
    palette = {
        "Normal": "C0", 
        "Hypopnea": "C2", 
        "Obstructive Apnea": "C3",
        "Central/Mixed Apnea": "C4",
        "Desaturation": "C5" # Added the missing key
    }

    for i, feature in enumerate(top_features_to_plot):
        ax = axes[i]
        f_score = oa_feature_importance.loc[i, 'f_score']
        
        # 4. Update the plotting call to be compatible with modern seaborn
        sns.violinplot(
            data=plot_df,
            x='ground_truth_event',
            y=feature,
            hue='ground_truth_event', # <-- Assign x to hue
            order=event_order_for_plot,
            palette=palette,
            ax=ax,
            legend=False # <-- Turn off the automatic legend
        )
        
        ax.set_title(f'{i+1}. {feature.replace("_", " ").title()}\n(Binary F-score: {f_score:.2f})')
        ax.set_xlabel('')
        ax.set_ylabel('Feature Value')
        ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for readability
        ax.grid(True, linestyle='--', alpha=0.5)

    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    LOCAL_FEATURE_DIR = './features-multi'

    # Make sure this map is PERFECTLY CONSISTENT with the plotting code
    GROUND_TRUTH_LABEL_MAP = {
        0: 'Normal',
        1: 'Obstructive Apnea',
        2: 'Hypopnea',
        3: 'Central/Mixed Apnea',
        4: 'Desaturation'
    }

    if os.path.isdir(LOCAL_FEATURE_DIR):
        analyze_oa_signature(LOCAL_FEATURE_DIR, GROUND_TRUTH_LABEL_MAP)
    else:
        print(f"Error: Directory not found at '{LOCAL_FEATURE_DIR}'.")