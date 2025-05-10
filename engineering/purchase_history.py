import os
import pandas as pd

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(dir + "data/cleaned_dataset.csv")

df['purchase_history'] = df['visitor_hist_starrating'].notnull().astype(int)

df.to_csv(dir + 'data/updated_dataset.csv', index=False)