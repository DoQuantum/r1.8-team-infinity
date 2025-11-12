import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box, Point
from itertools import islice
import pyproj
import json
import re
from datetime import datetime

target_stock = "Amazon"
csv_files = [f'readingArticles_{target_stock}.csv', f'{target_stock}_sentiments.csv']

# Load each CSV file into a DataFrame and store in a list
dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]
dataframes[0].drop('authors') # Temp - remove authors column in news csv
dataframes[1].rename(columns={'created_date': 'date', 'body': 'content'})
# Concatenate all DataFrames into a single DataFrame
csvFile = pd.concat(dataframes, ignore_index=True)
csvFile.to_csv(f"{target_stock}_combined.csv")