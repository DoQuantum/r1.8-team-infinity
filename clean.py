import numpy as np
import json
import pandas as pd
import csv

target_csv = 'Sentences_AllAgree.csv'
filename = 'Sentences_AllAgree.txt'

with open(filename, "r", encoding="cp1252") as f:
    sentence, sentiment = np.loadtxt(f,delimiter='@',dtype=str,unpack=True)
    data = np.array(sentence, sentiment)
    df = pd.DataFrame(data)
    df.to_csv(target_csv, index=False)