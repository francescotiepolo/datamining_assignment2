import pandas as pd
import numpy as np
import os

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(dir + "data/updated_dataset.csv")

# 1) Price comparison summary:  
#    +1 if Expedia is cheaper than every competitor with data,  
#     0 if it’s never strictly cheaper nor strictly more expensive,  
#    −1 if it’s more expensive than any competitor.
rate_cols = [f"comp{i}_rate" for i in range(1,9)]
valid = df[rate_cols].notna()
exp_cheapest   =  (df[rate_cols] == 1).where(valid, True).all(axis=1)
exp_not_cheapest   =  (df[rate_cols] == -1).any(axis=1)
df["price_cmp"] = (
    +1 * exp_cheapest
    +0 * (~exp_cheapest & ~exp_not_cheapest)
    -1 * exp_not_cheapest
).astype(int)


# 2) Any competitor has availability?  
#    compX_inv == 0 means “competitor X has availability”  
inv_cols = [f"comp{i}_inv" for i in range(1,9)]
df["any_comp_avail"] = df[inv_cols].eq(0).any(axis=1).astype(int)


# 3) Min/Max percent‐diff depending on who’s cheapest:  
#    If Expedia is the cheapest (price_cmp==+1): take the minimum of compX_rate_percent_diff across competitors with data.  
#    Otherwise take the maximum of those diffs.
pct_cols = [f"comp{i}_rate_percent_diff" for i in range(1,9)]
diff = pd.Series(np.nan, index=df.index)
m0 = df["price_cmp"] == +1
m1 = ~m0
diff.loc[m0] = df.loc[m0, pct_cols].min(axis=1)
diff.loc[m1] = df.loc[m1, pct_cols].max(axis=1)
df["pct_diff_rollup"] = diff

df.to_csv(dir + 'data/updated_dataset.csv', index=False)