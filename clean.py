import numpy as np
import json
import pandas as pd
import csv

target_csv = 'Sentences_AllAgree.csv'
filename = 'Sentences_AllAgree.txt'

# Get a list of all unique companies that appear in the csv
tickers = pd.read_csv('tickers_unique.csv')
tickers_list = tickers['company'].tolist()
tickers_list = list(set(tickers_list))
tickers_df = pd.DataFrame({'Name': tickers_list})
tickers_df.to_csv("SnP_list_all", index=False)

# Opens dataset file and organizes sentences/sentiments into a csv
# with open(filename, "r", encoding="cp1252") as f:
#     data = np.loadtxt(f,delimiter='@',dtype=str)
#     df = pd.DataFrame(data, columns=["sentence","sentiment"])
#     df['numerical_sentiment'] = np.ones
#     errcount = 0

#     for index, value in df['sentiment'].items():
#         # change sentiment labels to numbers
#         if value == 'positive':
#             df.iloc[index, 2] = '1'
#         elif value == 'neutral':
#             df.iloc[index, 2] = '0'
#         elif value == 'negative':
#             df.iloc[index, 2] = '-1'
#         else:
#             errcount += 1
#     # write to csv
#     #print(errcount, "errors found")
#     df.to_csv(target_csv, index=False)