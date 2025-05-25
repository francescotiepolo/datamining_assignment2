import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import ndcg_score
import dalex as dx

# 1. Load your data
dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(dir + "data/updated_dataset.csv")
df = df.drop(columns=["visitor_hist_starrating", "visitor_hist_adr_usd"])

# 2. Create your relevance label
#    5 if booked, 1 if clicked only, 0 otherwise
y = df["booking_bool"] * 5 + df["click_bool"] * 1
df = df.drop(columns=["booking_bool", "click_bool"])

# Drop object-type columns not suitable for LightGBM
df["srch_month"] = pd.to_datetime(df["date_time"]).dt.month
df["srch_weekday"] = pd.to_datetime(df["date_time"]).dt.weekday
df = df.drop(columns=["date_time"])


# 3. Train/validation split by query (srch_id)
#    ensure all hotels of the same srch_id stay together
unique_searches = df["srch_id"].unique()
train_srch, val_srch = train_test_split(unique_searches, test_size=0.1, random_state=133)

is_train = df["srch_id"].isin(train_srch)
X_train, X_val = df[is_train], df[~is_train]
y_train, y_val = y[is_train], y[~is_train]
groups_train = X_train.groupby(df[is_train]["srch_id"]).size().to_list()
groups_val   = X_val.groupby(df[~is_train]["srch_id"]).size().to_list()

# 4. Create LightGBM datasets with group info
lgb_train = lgb.Dataset(X_train, label=y_train, group=groups_train)
lgb_val   = lgb.Dataset(X_val,   label=y_val,   group=groups_val, reference=lgb_train)

# 5. Specify LambdaRank parameters
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],       # evaluate NDCG@5
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbose": 0
}

# 6. Train with early stopping on validation
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_val],
    valid_names=["valid"],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

def compute_ndcg(model, X, y, srch_ids, k=5):
    tmp = X.copy()
    tmp["srch_id"] = srch_ids
    tmp["y_true"]  = y
    tmp["y_pred"]  = model.predict(X, num_iteration=model.best_iteration)
    scores = []
    for _, grp in tmp.groupby("srch_id"):
        true = grp["y_true"].values.reshape(1, -1)
        pred = grp["y_pred"].values.reshape(1, -1)
        if true.shape[1] < 2:
            continue
        scores.append(ndcg_score(true, pred, k=k))
    return np.mean(scores)

final_ndcg = compute_ndcg(model, X_val, y_val, X_val["srch_id"], k=5)
print(final_ndcg)
# exp = dx.Explainer(model, X_train, y_train)

# exp.model_fairness(X_train["srch_children_count"].apply(lambda x: "0" if x == 0 else "1"), "0").fairness_check(epsilon=0.8)
# exp.model_fairness(X_train["srch_adults_count"].apply(lambda x: "1" if x == 1 else ("2" if x == 2 else "3")), "1").fairness_check(epsilon=0.8)
# exp.model_fairness(X_train["srch_adults_count"].apply(lambda x: "1" if x == 1 else ("2" if x == 2 else "3")), "2").fairness_check(epsilon=0.8)
# exp.model_fairness(X_train["srch_booking_window"].apply(lambda x: "short" if x <= 30 else "long"), "short").fairness_check(epsilon=0.8)
# exp.model_fairness(X_train["srch_length_of_stay"].apply(lambda x: "short" if x <= 6 else "long"), "long").fairness_check(epsilon=0.8)
# exp.model_fairness(X_train["srch_saturday_night_bool"].apply(lambda x: "weekday" if x == 1 else "weekend"), "weekday").fairness_check(epsilon=0.8)
# exp.model_fairness(X_train["visitor_hist_starrating"].apply(lambda x: "new" if pd.isna(x) else "old"), "new").fairness_check(epsilon=0.8)
# exp.model_fairness(X_train["visitor_hist_adr_usd"].apply(lambda x: "new" if pd.isna(x) else "old"), "new").fairness_check(epsilon=0.8)