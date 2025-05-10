import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(dir + "data/training_set_VU_DM.csv")

# Drop

columns_to_drop = [
    'visitor_hist_adr_usd',
    'prop_log_historical_price',
    'srch_query_affinity_score',
    'orig_destination_distance',
    'position',
    'gross_bookings_usd'
]

df.drop(columns=columns_to_drop)

# Fill star rating

df['prop_starrating'] = df['prop_starrating'].replace(0, np.nan)
known = df[df['prop_starrating'].notnull()]
missing = df[df['prop_starrating'].isnull()]

features = [
    'prop_location_score1',
    'prop_location_score2',
    'price_usd']

X_known = known[features]
y_known = known['prop_starrating'].astype(int)

X_missing = missing[features]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_known, y_known)

predicted = knn.predict(X_missing)

df.loc[df['prop_starrating'].isnull(), 'prop_starrating'] = predicted
df['prop_starrating'] = df['prop_starrating'].astype(int)

# Fill review score

df['has_reviews'] = df['prop_review_score'].apply(
    lambda x: 1 if x > 0 else (0 if x == 0 else np.nan)
)

df['prop_review_score'] = df['prop_review_score'].replace(0, np.nan)

knn = KNNImputer(n_neighbors=5)
sub = df[features + ['prop_review_score']]
imputed = knn.fit_transform(sub)

df['prop_review_score'] = imputed[:, -1].round().astype(int)


df.to_csv(dir + 'data/cleaned_dataset.csv', index=False)

# Fill has reviews

known = df[df['has_reviews'].notnull()]
missing = df[df['has_reviews'].isnull()]

X_known = known[features]
y_known = known['has_reviews'].astype(int)

X_missing = missing[features]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_known, y_known)

predicted = knn.predict(X_missing)

df.loc[df['has_reviews'].isnull(), 'has_reviews'] = predicted
df['has_reviews'] = df['has_reviews'].astype(int)