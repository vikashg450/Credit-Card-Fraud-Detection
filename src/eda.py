import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def generate_eda_plots(csv_path="creditcard.csv", output_dir="static/plots"):
    print("--- Phase 2: Generating EDA Plots ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # 1. Amount Distribution Plot (Log Scale)
    print("Plotting Amount Distribution...")
    plt.figure(figsize=(10, 5))
    # We will log-transform Amount using log1p
    df['log_amount'] = np.log1p(df['Amount'])
    
    sns.kdeplot(data=df[df['Class'] == 0], x='log_amount', label='Legitimate', color='#10b981', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(data=df[df['Class'] == 1], x='log_amount', label='Fraudulent', color='#f43f5e', fill=True, alpha=0.3, linewidth=2)
    
    plt.title("Transaction Amount Distribution (Log-scaled)", fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    plt.xlabel("Log(Amount + 1)", fontsize=12, labelpad=10)
    plt.ylabel("Density", fontsize=12, labelpad=10)
    plt.legend(facecolor='#1a1a24', edgecolor='#2c2c35')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "amount_distribution.png")
    plt.savefig(plot_path, dpi=300, facecolor='#0d0d12')
    plt.close()
    print(f"Saved amount distribution to: {plot_path}")
    
    # 2. Time Distribution Plot (Hour of Day)
    print("Plotting Time Distribution...")
    plt.figure(figsize=(10, 5))
    df['hour_of_day'] = (df['Time'] % 86400) / 3600  # 0 to 24 hours
    
    sns.kdeplot(data=df[df['Class'] == 0], x='hour_of_day', label='Legitimate', color='#10b981', fill=True, alpha=0.1, linewidth=2, common_norm=False)
    sns.kdeplot(data=df[df['Class'] == 1], x='hour_of_day', label='Fraudulent', color='#f43f5e', fill=True, alpha=0.3, linewidth=2, common_norm=False)
    
    plt.title("Transaction Volume over Hour of Day", fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    plt.xlabel("Hour of Day (0 - 24)", fontsize=12, labelpad=10)
    plt.ylabel("Density (Normalized)", fontsize=12, labelpad=10)
    plt.xticks(np.arange(0, 25, 2))
    plt.xlim(0, 24)
    plt.legend(facecolor='#1a1a24', edgecolor='#2c2c35')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "time_distribution.png")
    plt.savefig(plot_path, dpi=300, facecolor='#0d0d12')
    plt.close()
    print(f"Saved time distribution to: {plot_path}")
    
    # 3. Correlation heatmap of top features
    print("Plotting Correlation Heatmap...")
    # Calculate correlation of all features with Class
    corr = df.drop(columns=['log_amount', 'hour_of_day']).corr()
    corr_with_class = corr['Class'].sort_values()
    
    # Select top 7 negatively correlated and top 7 positively correlated variables (excluding Class itself)
    top_neg = corr_with_class.index[:7]
    top_pos = corr_with_class.index[-8:-1]  # exclude Class at the end
    top_corr_features = list(top_neg) + list(top_pos) + ['Class']
    
    plt.figure(figsize=(12, 10))
    sub_corr = df[top_corr_features].corr()
    
    # Create custom colormap matching our palette
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1.0, vmax=1.0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 9, "weight": "bold"})
    
    plt.title("Correlation Matrix of Top Features with Fraud Label", fontsize=14, fontweight='bold', pad=20, color='#f8fafc')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(plot_path, dpi=300, facecolor='#0d0d12')
    plt.close()
    print(f"Saved correlation matrix to: {plot_path}")
    
    # 4. Top Features Scatter Plot (Separability)
    # The two most negatively correlated features are V17 and V14, and V12 is also very high.
    # Let's plot V17 vs V14.
    print("Plotting Separability Scatter Plot...")
    plt.figure(figsize=(10, 6))
    
    # Plot non-fraud as small points first, then fraud as larger points to make sure they are visible on top
    sns.scatterplot(data=df[df['Class'] == 0], x='V17', y='V14', color='#10b981', alpha=0.1, s=5, label='Legitimate')
    sns.scatterplot(data=df[df['Class'] == 1], x='V17', y='V14', color='#f43f5e', alpha=0.8, s=25, label='Fraudulent', edgecolor='white', linewidth=0.5)
    
    plt.title("Separability of Fraudulent Transactions (V17 vs V14)", fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    plt.xlabel("Principal Component V17", fontsize=12, labelpad=10)
    plt.ylabel("Principal Component V14", fontsize=12, labelpad=10)
    plt.legend(facecolor='#1a1a24', edgecolor='#2c2c35')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "top_features_scatter.png")
    plt.savefig(plot_path, dpi=300, facecolor='#0d0d12')
    plt.close()
    print(f"Saved separability scatter to: {plot_path}")
    print("EDA Plot Generation completed successfully!\n")

if __name__ == "__main__":
    generate_eda_plots()
