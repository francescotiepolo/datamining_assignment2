import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

SAMPLE_SIZE = 1000
RANDOM_STATE = 133
N_ESTIMATORS = 100

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(dir + "data/test_set_VU_DM.csv")

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

# Create purchase history feature
df['purchase_history'] = df['visitor_hist_starrating'].notnull().astype(int)

# Price comparison summary:  
#    +1 if Expedia is cheaper than every competitor with data,  
#     0 if it’s never strictly cheaper nor strictly more expensive,  
#    −1 if it’s more expensive than any competitor.
rate_cols = [f"comp{i}_rate" for i in range(1,9)]
valid = df[rate_cols].notna()
exp_cheapest   =  (df[rate_cols] == 1).where(valid, True).all(axis=1)
exp_not_cheapest   =  (df[rate_cols] == -1).any(axis=1)
df["price_comp"] = (
    +1 * exp_cheapest
    +0 * (~exp_cheapest & ~exp_not_cheapest)
    -1 * exp_not_cheapest
).astype(int)


# Any competitor has availability?   
inv_cols = [f"comp{i}_inv" for i in range(1,9)]
df["any_comp_avail"] = df[inv_cols].eq(0).any(axis=1).astype(int)

df.to_csv(dir + 'data/test_updated_dataset.csv', index=False)