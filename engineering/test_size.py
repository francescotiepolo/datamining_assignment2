import os
import pandas as pd

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df_old = pd.read_csv(dir + "data/training_set_VU_DM.csv")
df_new = pd.read_csv(dir + "data/updated_dataset.csv")

print("Original shape:", df_old.shape)
print("New shape:", df_new.shape)