import numpy as np
import json
import pandas as pd
import csv

target_csv = 'Sentences_AllAgree.csv'
filename = 'Sentences_AllAgree.txt'

# Opens dataset file and organizes sentences/sentiments into a csv
with open(filename, "r", encoding="cp1252") as f:
    data = np.loadtxt(f,delimiter='@',dtype=str)
    df = pd.DataFrame(data, columns=["sentence","sentiment"])
    for index, value in df['sentiment'].items():
        # change sentiment labels to numbers
        if value == 'positive':
            df.iloc[index, 1] = '1'
        if value == 'neutral':
            df.iloc[index, 1] = '0'
        if value == 'negative':
            df.iloc[index, 1] = '-1'
    # write to csv
    df.to_csv(target_csv, index=False)