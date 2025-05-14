import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

SAMPLE_SIZE = 1000
RANDOM_STATE = 42
N_ESTIMATORS = 100

# Paths
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
INPUT_CSV = os.path.join(BASE_DIR, "data", "training_set_VU_DM.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "cleaned_dataset.csv")

# Load data
df = pd.read_csv(INPUT_CSV)

# Drop unnecessary columns
df = df.drop(columns=[
    'visitor_hist_adr_usd',
    'prop_log_historical_price',
    'srch_query_affinity_score',
    'orig_destination_distance',
    'position',
    'gross_bookings_usd',
    'prop_location_score2'
])
pct_diff_cols = [col for col in df.columns if col.endswith("_rate_percent_diff")]
df = df.drop(columns=pct_diff_cols)


# Features to use for imputations
features = ['prop_location_score1', 'price_usd']

# Fill prop_starrating
df['prop_starrating'] = df['prop_starrating'].replace(0, np.nan)
mask = df['prop_starrating'].isna()

if mask.any():
    known = df.loc[~mask, features + ['prop_starrating']].dropna()
    if len(known) > SAMPLE_SIZE:
        known = known.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    rf_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf_clf.fit(known[features], known['prop_starrating'].astype(int))

    df.loc[mask, 'prop_starrating'] = rf_clf.predict(df.loc[mask, features])

df['prop_starrating'] = df['prop_starrating'].astype(int)

# Create has_reviews and clean prop_review_score
df['has_reviews'] = np.where(
    df['prop_review_score'] > 0,
    1,
    np.where(df['prop_review_score'] == 0, 0, np.nan)
)

df['prop_review_score'] = df['prop_review_score'].replace(0, np.nan)

# Fill prop_review_score
mask = df['prop_review_score'].isna()

if mask.any():
    known = df.loc[~mask, features + ['prop_review_score']].dropna()
    if len(known) > SAMPLE_SIZE:
        known = known.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    rf_reg = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf_reg.fit(known[features], known['prop_review_score'])

    df.loc[mask, 'prop_review_score'] = rf_reg.predict(df.loc[mask, features])

df['prop_review_score'] = df['prop_review_score'].astype(float)

# Fill has_reviews
mask = df['has_reviews'].isna()

if mask.any():
    known = df.loc[~mask, features + ['has_reviews']].dropna()
    if len(known) > SAMPLE_SIZE:
        known = known.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    rf_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf_clf.fit(known[features], known['has_reviews'].astype(int))

    df.loc[mask, 'has_reviews'] = rf_clf.predict(df.loc[mask, features])

df['has_reviews'] = df['has_reviews'].astype(int)

# Save dataset
df.to_csv(OUTPUT_CSV, index=False)
