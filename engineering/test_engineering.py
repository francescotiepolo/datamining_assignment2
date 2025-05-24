import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set consistent style
sns.set(style="whitegrid")

# Define base directory
dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

# Load datasets
df_old = pd.read_csv(os.path.join(dir, "data", "training_set_VU_DM.csv"))
df_new = pd.read_csv(os.path.join(dir, "data", "updated_dataset.csv"))

# Basic inspection
print("Original shape:", df_old.shape)
print("New shape:", df_new.shape)
print(df_new[['purchase_history', 'price_comp', 'any_comp_avail']].isna().sum())
print(df_new[['purchase_history', 'price_comp', 'any_comp_avail']].dtypes)

# Columns to visualize
columns_to_plot = ['purchase_history', 'price_comp', 'any_comp_avail']

# Loop over and plot each
for col in columns_to_plot:
    plt.figure(figsize=(10, 6))  # Bigger plot size
    sns.countplot(x=col, data=df_new)
    plt.title(f"Distribution of '{col.replace('_', ' ').title()}'", fontsize=25)
    plt.xlabel(col.replace('_', ' ').title(), fontsize=23)
    plt.ylabel("Count", fontsize=23)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "images", f"{col}.png"))
    plt.clf()

    print(f"{col} summary:")
    print(df_new[col].describe())
