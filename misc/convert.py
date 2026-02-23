#This file converts the 2s in the training dataset WSB_all_agree into -1 to align with the rest of the code
import pandas as pd
df = pd.read_csv("training.csv")

#print(label)

df["label"] = df["label"].replace({2: -1})
print(df["label"])

df.to_csv("training.csv", index=False)
print(df)
