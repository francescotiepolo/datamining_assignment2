import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

# 1. Load full data
base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(base_dir + "data/updated_dataset.csv")

# Create relevance labels and drop originals
y_full = df["booking_bool"] * 5 + df["click_bool"] * 1
df = df.drop(columns=["booking_bool", "click_bool"])

# Extract date
df["srch_month"]   = pd.to_datetime(df["date_time"]).dt.month
df["srch_weekday"] = pd.to_datetime(df["date_time"]).dt.weekday
df = df.drop(columns=["date_time"])

# Split full data into train/val
all_searches = df["srch_id"].unique()
train_srch_full, val_srch_full = train_test_split(all_searches, test_size=0.1, random_state=133)

mask_train_full = df["srch_id"].isin(train_srch_full)
X_train_full = df[mask_train_full].reset_index(drop=True)
X_val_full   = df[~mask_train_full].reset_index(drop=True)
y_train_full = y_full[mask_train_full].reset_index(drop=True)
y_val_full   = y_full[~mask_train_full].reset_index(drop=True)

# 2. Sample a subset of queries for tuning (20%)
np.random.seed(133)
sub_queries = np.random.choice(train_srch_full,
                               size=int(0.5 * len(train_srch_full)),
                               replace=False)

mask_sub = X_train_full["srch_id"].isin(sub_queries)
X_train_sub = X_train_full[mask_sub].reset_index(drop=True)
y_train_sub = y_train_full[mask_sub].reset_index(drop=True)
groups_train_sub = X_train_sub.groupby("srch_id").size().to_list()

# Split subset into sub-train and sub-val for Optuna
sub_srch = X_train_sub["srch_id"].unique()
sub_tr, sub_val = train_test_split(sub_srch, test_size=0.1, random_state=133)
mask_tr_sub = X_train_sub["srch_id"].isin(sub_tr)

X_tr_sub = X_train_sub[mask_tr_sub].reset_index(drop=True)
X_val_sub = X_train_sub[~mask_tr_sub].reset_index(drop=True)
y_tr_sub = y_train_sub[mask_tr_sub].reset_index(drop=True)
y_val_sub = y_train_sub[~mask_tr_sub].reset_index(drop=True)

groups_train_full = X_train_full.groupby("srch_id").size().to_list()
groups_val_full   = X_val_full.groupby("srch_id").size().to_list()

def compute_ndcg(model, X, y, srch_ids, k=5):
    tmp = X.copy()
    tmp["srch_id"] = srch_ids
    tmp["y_true"]  = y
    tmp["y_pred"]  = model.predict(xgb.DMatrix(X))
    scores = []
    for _, grp in tmp.groupby("srch_id"):
        true = grp["y_true"].values.reshape(1, -1)
        pred = grp["y_pred"].values.reshape(1, -1)
        if true.shape[1] < 2:
            continue
        scores.append(ndcg_score(true, pred, k=k))
    return np.mean(scores)

def objective(trial):
    dtrain = xgb.DMatrix(X_train_full, label=y_train_full)
    dtest = xgb.DMatrix(X_val_full, label=y_val_full)

    param = {
        "silent": 1,
        "objective": "rank:ndcg",
        "eval_metric": "ndcg",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-ndcg")
    bst = xgb.train(param, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback])
    return compute_ndcg(bst, X_val_sub, y_val_sub, X_val_sub["srch_id"], k=5)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=10000)

final_params = {
    **study.best_params,
    "objective": "rank:ndcg",
    "eval_metric": "ndcg",
}

dtrain = xgb.DMatrix(X_train_full, label=y_train_full)
dtest = xgb.DMatrix(X_val_full, label=y_val_full)

final_model = xgb.train(
    final_params,
    dtrain,
    valid_sets=[dtest],
)

# Evaluate final
final_ndcg = compute_ndcg(final_model, X_val_full, y_val_full, X_val_full["srch_id"], k=5)

# Create a single LightGBM dataset
X_all = pd.concat([X_train_full, X_val_full], ignore_index=True)
y_all = pd.concat([y_train_full, y_val_full], ignore_index=True)
groups_all = X_all.groupby("srch_id").size().to_list()
dall = xgb.DMatrix(X_all, label=y_all, group=groups_all)

# Train on the entire labeled data, using the best params
cv_results = xgb.cv(
    final_params,
    dall,
    nfold=5,
    stratified=False,
    num_boost_round=5000,
    metrics="ndcg",
    seed=133
)

mean_key = next(k for k in cv_results.keys() if k.endswith("-mean"))
best_rounds = len(cv_results[mean_key])

print(f"Best number of rounds from CV: {best_rounds}")

model_all = xgb.train(
    final_params,
    dall,
    num_boost_round=best_rounds
)

# 6. Load & preprocess test data
df_test = pd.read_csv(base_dir + 'data/test_updated_dataset.csv')

# Extract the same date features
df_test["srch_month"]   = pd.to_datetime(df_test["date_time"]).dt.month
df_test["srch_weekday"] = pd.to_datetime(df_test["date_time"]).dt.weekday
df_test = df_test.drop(columns=["date_time"])

# 7. Predict scores
df_test["pred_score"] = model_all.predict(df_test, num_iteration=final_model.best_iteration)

# 8. Rank within each search (assign rank=1 to highest score, 2 to next, etc.)
df_test["rank"] = (
    df_test
    .groupby("srch_id")["pred_score"]
    .rank(method="first", ascending=False)
    .astype(int)
)

# 9. Write out submission
submission = (
    df_test
    .sort_values(["srch_id", "rank"])
    .loc[:, ["srch_id", "prop_id"]]
    .rename(columns={"srch_id": "SearchId", "prop_id": "PropertyId"})
)

submission.to_csv(base_dir + "data/final_submission.csv", index=False)