import os
import pandas as pd

dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df_question = pd.read_csv(dir + "data/test_set_VU_DM.csv")
df_answer = pd.read_csv(dir + "data/VU-DM-2025-Group-129.csv")

print("Original shape:", df_question.shape)
print("New shape:", df_answer.shape)