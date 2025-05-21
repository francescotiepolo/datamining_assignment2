import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df_old = pd.read_csv(dir + "data/training_set_VU_DM.csv")
df_new = pd.read_csv(dir + "data/updated_dataset.csv")

print("Original shape:", df_old.shape)
print("New shape:", df_new.shape)

print(df_new[['purchase_history', 'price_comp', 'any_comp_avail']].isna().sum())
print(df_new[['purchase_history', 'price_comp', 'any_comp_avail']].dtypes)

columns_to_plot = ['purchase_history', 'price_comp', 'any_comp_avail']

for col in columns_to_plot:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df_new)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(dir + f"images/{col}.png")
    plt.clf()
    print(f"{col} summary:")
    print(df_new[col].describe())