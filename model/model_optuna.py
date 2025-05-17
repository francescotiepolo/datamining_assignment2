import os
import pandas as pd
import numpy as np
import lightgbm as lgb
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

groups_train_full = X_train_full.groupby("srch_id").size().to_list()
groups_val_full   = X_val_full.groupby("srch_id").size().to_list()

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

groups_tr_sub = X_tr_sub.groupby("srch_id").size().to_list()
groups_val_sub = X_val_sub.groupby("srch_id").size().to_list()

# Func to compute NDCG@5
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

# 3. Optuna objective on the subset
def objective(trial):
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "verbose": 0,
    }

    dtrain = lgb.Dataset(X_tr_sub, label=y_tr_sub, group=groups_tr_sub)
    dval   = lgb.Dataset(X_val_sub, label=y_val_sub, group=groups_val_sub, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )
    return compute_ndcg(model, X_val_sub, y_val_sub, X_val_sub["srch_id"], k=5)

# 4. Run Optuna tuning
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="lgbm_lambdarank")
    study.enqueue_trial({
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 1
    })
    study.optimize(objective, n_trials=100, timeout=10000)

    print(">>> Best NDCG@5 on subset:", study.best_value)
    print(">>> Best params:", study.best_params)

# 5. Retrain final model on full training data
    final_params = {
        **study.best_params,
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "verbose": 0,
    }

    dtrain_full = lgb.Dataset(X_train_full, label=y_train_full, group=groups_train_full)
    dval_full   = lgb.Dataset(X_val_full,   label=y_val_full,   group=groups_val_full, reference=dtrain_full)

    final_model = lgb.train(
        final_params,
        dtrain_full,
        valid_sets=[dval_full],
        num_boost_round=3000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    # Evaluate final
    final_ndcg = compute_ndcg(final_model, X_val_full, y_val_full, X_val_full["srch_id"], k=5)
    print(f">>> Final model NDCG@5 on full val: {final_ndcg:.4f}")

# Create a single LightGBM dataset
X_all = pd.concat([X_train_full, X_val_full], ignore_index=True)
y_all = pd.concat([y_train_full, y_val_full], ignore_index=True)
groups_all = X_all.groupby("srch_id").size().to_list()
dall = lgb.Dataset(X_all, label=y_all, group=groups_all)

# Train on the entire labeled data, using the best params
cv_results = lgb.cv(
    final_params,
    dall,
    nfold=5,
    stratified=False,
    num_boost_round=5000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=0),    # suppress per‑round logs
    ],
    metrics="ndcg",
    seed=133
)

mean_key = next(k for k in cv_results.keys() if k.endswith("-mean"))
best_rounds = len(cv_results[mean_key])

print(f"Best number of rounds from CV: {best_rounds}")

model_all = lgb.train(
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
)

submission.to_csv(base_dir + "data/VU-DM-2025-Group-129.csv", index=False)