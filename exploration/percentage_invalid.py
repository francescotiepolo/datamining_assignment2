#!/usr/bin/python3
import os
import pandas as pd

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(dir + "data/training_set_VU_DM.csv")

columns = {
    "visitor_hist_starrating": ["NULL"],
    "visitor_hist_adr_usd": ["NULL"],
    "prop_starrating": ["0"],
    "prop_review_score": ["0", "NULL"],
    "prop_log_historical_price": ["0"],
    "srch_query_affinity_score": ["NULL"],
    "orig_destination_distance": ["NULL"],
    "visitor_hist_starrating": ["NULL"],
    "visitor_hist_starrating": ["NULL"],
    "visitor_hist_starrating": ["NULL"],
    "prop_location_score2": ["NULL"]
}

for i in range(1,9):
    columns[f"comp{i}_rate"] = ["NULL"]
    columns[f"comp{i}_inv"] = ["NULL"]
    columns[f"comp{i}_rate_percent_diff"] = ["NULL"]

counts_null = {}
counts_0 = {}

for key in columns.keys():
    if "NULL" in columns[key]:
        invalids = df[key].isnull().sum()
        counts_null[key] = {}
        counts_null[key]["NULL"] = invalids/df[key].size
    if "0" in columns[key]:
        invalids = df[key].isin([0]).sum()
        counts_0[key] = {}
        counts_0[key]["0"] = invalids/df[key].size

pd.DataFrame.from_dict(counts_null).T.to_latex(dir + "exploration/percentage_null.tex")
pd.DataFrame.from_dict(counts_0).T.to_latex(dir + "exploration/percentage_0.tex")