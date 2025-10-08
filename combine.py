import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.api import ARDL
from statsmodels.tsa.ardl import ardl_select_order

# Placeholder
stock_prices = pd.Series([1,2,3,4,5])
daily_sentiment = pd.Series([-1,-0.5,0,0.5,1])




# Suppose stock_prices is a pandas Series with the same index as daily_sentiment
ardl_model = ARDL(stock_prices, lags=2, exog=daily_sentiment)
ardl_result = ardl_model.fit()
print(ardl_result.summary())
