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
    df.to_csv(target_csv, index=False)