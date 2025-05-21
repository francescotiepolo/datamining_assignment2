import pandas as pd
import numpy as np
import os

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(dir + "data/cleaned_dataset.csv")

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

df.to_csv(dir + 'data/updated_dataset.csv', index=False)