import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.filters.hp_filter import hpfilter
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

# Datetime to Int converter
def datetime_to_int(dt_list):
    print(type(dt_list))
    print(dt_list)

    ret = []
    for dt in dt_list:
        ret.append(1*dt.year + 1/12*dt.month +1/365*dt.day)
    return ret

# Read generated .CSV to pandas Dataframe
df_daily_checkouts = pd.read_csv('data/gen/daily_checkouts.csv')
df_daily_checkouts['date'] = pd.to_datetime(df_daily_checkouts['date'], format="%Y-%m-%dT00:00:00.000Z")
df_date = df_daily_checkouts['date']

# Show Daily checkout counts plot
df_daily_checkouts[['date', 'checkout_count']].plot(x='date', y='checkout_count')
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Daily checkout counts')
plt.show()

# Show Linear regression plot
x = np.array(datetime_to_int(df_daily_checkouts['date']))
b = stats.linregress(x, df_daily_checkouts.checkout_count).intercept
m = stats.linregress(x, df_daily_checkouts.checkout_count).slope
stats.linregress(x, df_daily_checkouts.checkout_count)

x_line = np.array(np.linspace(min(x), max(x), 2))
y_line = b + m * x_line

plt.figure(figsize=(12, 7))
plt.scatter(x, df_daily_checkouts.checkout_count)
plt.plot(x_line, y_line, color='red')
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Daily Linear regression')
plt.show()

# Show Seasonal coupon count plot regression plot
df_daily_checkouts.index = pd.to_datetime(df_daily_checkouts['date'])

result = seasonal_decompose(df_daily_checkouts['coupon_count'], model='multiplicative')
result.trend.plot()
plt.xlabel('date')
plt.ylabel('coupon count')
plt.title('Seasonal coupon count')
plt.show()

# Show Non-coupon plots
non_coupon=[]
for i in range(len(df_daily_checkouts.checkout_count)):
    non_coupon.append(df_daily_checkouts.checkout_count[i]-df_daily_checkouts.coupon_count[i])
# Filter
for i in range(len(non_coupon)):
    if non_coupon[i]>2500:
        non_coupon[i]=2500
# Create non-coupon dataframe and merge with date
df_none_coupon=pd.DataFrame(non_coupon, columns=['non'])
df_none_coupon=pd.concat([df_date,df_none_coupon], axis=1)
df_none_coupon=df_none_coupon.set_index('date')
df_none_coupon.index=pd.to_datetime(df_none_coupon.index)
print(df_none_coupon)
# Calculate regression line
b = stats.linregress(x, df_none_coupon.non).intercept
m = stats.linregress(x, df_none_coupon.non).slope
stats.linregress(x, df_none_coupon.non)
x_line = np.array(np.linspace(min(x), max(x), 2))
y_line = b + m*x_line

plt.figure(figsize=(12, 7))
plt.scatter(x, df_none_coupon.non)
plt.xlabel('date')
plt.ylabel('Non-coupon count')
plt.title('Non-coupon regression')
plt.plot(x_line, y_line, color='red')
plt.show()

result = seasonal_decompose(df_none_coupon['non'], model='multiplicative')
result.trend.plot()
plt.xlabel('date')
plt.ylabel('Non-coupon count')
plt.title('Non-coupon trend')
plt.show()
