import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set seaborn style for consistency
sns.set(style="whitegrid")

# Define base directory
dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

# Load data
df_clean = pd.read_csv(os.path.join(dir, "data", "cleaned_dataset.csv"))
df_dirty = pd.read_csv(os.path.join(dir, "data", "training_set_VU_DM.csv"))

# Basic checks
zero_count = (df_dirty['prop_starrating'] == 0).sum()
print(f'prop_starrating {zero_count}')
print(df_dirty[['prop_review_score']].isna().sum())

zero_count = (df_clean['prop_starrating'] == 0).sum()
print(f'prop_starrating (zero) {zero_count}')
print(df_clean[['prop_starrating', 'prop_review_score', 'has_reviews']].isna().sum())
print(df_clean[['prop_starrating', 'prop_review_score', 'has_reviews']].dtypes)

# Plot: prop_starrating
plt.figure(figsize=(10, 6))
sns.countplot(x='prop_starrating', data=df_clean)
plt.title("Distribution of Property Star Ratings", fontsize=25)
plt.xlabel("Star Rating", fontsize=23)
plt.ylabel("Count", fontsize=23)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tight_layout()
plt.savefig(os.path.join(dir, "images", "prop_starrating.png"))
plt.clf()
print("'prop_starrating' summary:")
print(df_clean['prop_starrating'].describe())

# Plot: prop_review_score
plt.figure(figsize=(10, 6))
sns.histplot(x='prop_review_score', data=df_clean, bins=10)
plt.title("Distribution of Review Scores", fontsize=25)
plt.xlabel("Review Score", fontsize=23)
plt.ylabel("Count", fontsize=23)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tight_layout()
plt.savefig(os.path.join(dir, "images", "prop_review_score.png"))
plt.clf()
print("'prop_review_score' summary:")
print(df_clean['prop_review_score'].describe())

# Plot: has_reviews
plt.figure(figsize=(10, 6))
sns.countplot(x='has_reviews', data=df_clean)
plt.title("Has Reviews (1 = Yes, 0 = No)", fontsize=25)
plt.xlabel("Has Reviews", fontsize=23)
plt.ylabel("Count", fontsize=23)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tight_layout()
plt.savefig(os.path.join(dir, "images", "has_reviews.png"))
plt.clf()
print("'has_reviews' summary:")
print(df_clean['has_reviews'].describe())
