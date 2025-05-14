import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df_clean = pd.read_csv(dir + "data/cleaned_dataset.csv")
df_dirty = pd.read_csv(dir + "data/training_set_VU_DM.csv")

zero_count = (df_dirty['prop_starrating'] == 0).sum()
print(f'prop_starrating {zero_count}')
print(df_dirty[['prop_review_score']].isna().sum())

zero_count = (df_clean['prop_starrating'] == 0).sum()
print(f'prop_starrating (zero) {zero_count}')
print(df_clean[['prop_starrating', 'prop_review_score', 'has_reviews']].isna().sum())
print(df_clean[['prop_starrating', 'prop_review_score', 'has_reviews']].dtypes)

sns.countplot(x='prop_starrating', data=df_clean)
plt.title("Distribution of Property Star Ratings")
plt.xlabel("Star Rating")
plt.ylabel("Count")
plt.savefig(dir + "images/prop_starrating.png")
plt.clf()
print("'prop_starrating' summary:")
print(df_clean['prop_starrating'].describe())

sns.histplot(x='prop_review_score', data=df_clean, bins=10)
plt.title("Distribution of Review Scores")
plt.xlabel("Review Score")
plt.ylabel("Count")
plt.savefig(dir + "images/prop_review_score.png")
plt.clf()
print("'prop_review_score' summary:")
print(df_clean['prop_review_score'].describe())

sns.countplot(x='has_reviews', data=df_clean)
plt.title("Has Reviews (1 = Yes, 0 = No)")
plt.xlabel("Has Reviews")
plt.ylabel("Count")
plt.savefig(dir + "images/has_reviews.png")
print("'has_reviews' summary:")
print(df_clean['has_reviews'].describe())