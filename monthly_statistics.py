import os
import moment
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
from pandas.plotting import autocorrelation_plot as aplt
import datetime as dt
import numpy as np
from numpy import log
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.filters.hp_filter import hpfilter
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import warnings


# Datetime to Int converter
def datetime_to_int(dt_list):
    print(type(dt_list))
    print(dt_list)

    ret = []
    for dt in dt_list:
        ret.append(1*dt.year + 1/12*dt.month)
    return ret

# ARIMA predict conversion
def arima_sales_float(n):
    return float(model_fit.predict(n))

# ARIMA prediction
def arima_sales_predict(n):
    if ((arima_sales_float(n)-int(arima_sales_float(n)))<1/2):
        return int(arima_sales_float(n))
    else:
        return int(arima_sales_float(n))+1

# Create ARIMA model
def evaluate_arima_model(Y, arima_order):

    size = int(len(Y) * 0.66)
    train, test = Y[0:size],Y[size:len(Y)]
    history = [x for x in train]
    predictions = list()
    
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    
    error = mean_squared_error(test, predictions)
    plt.plot(test)
    plt.plot(predictions, color='red')
    return sqrt(error)

# Create correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix', fontsize=15)
    plt.show()

# Read generated .CSV to pandas Dataframe
df_monthly_checkouts = pd.read_csv('./data/gen/monthly_checkouts.csv')
df_monthly_checkouts['date'] = pd.to_datetime(df_monthly_checkouts['date'], format="%Y-%m-%d")
df_date = df_monthly_checkouts['date']

# Show Monthly checkout counts plot
df_monthly_checkouts[['date', 'checkout_count']].plot(x='date', y='checkout_count')
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Monthly checkout counts')
plt.show()

# Show Monthly linear regression plot
plt.figure(figsize=(12, 7))
# Create regression line
x = np.array(datetime_to_int(df_monthly_checkouts['date']))
sns.regplot(x=x, y=df_monthly_checkouts.checkout_count)
b = stats.linregress(x, df_monthly_checkouts.checkout_count).intercept
m = stats.linregress(x, df_monthly_checkouts.checkout_count).slope

x_line = np.array(np.linspace(min(x), max(x), 2))
y_line = b + m*x_line
plt.scatter(x, df_monthly_checkouts.checkout_count)
plt.plot(x_line, y_line, color='red') 
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Monthly Linear regression')
plt.show()

# Monthly checkouts
df_monthly_checkouts.index = pd.to_datetime(df_monthly_checkouts['date'])

result = seasonal_decompose(df_monthly_checkouts['checkout_count'], model='multiplicative')
result.seasonal.plot()
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Seasonal part of monthly checkouts')
plt.show()
result.trend.plot()
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Trend part of monthly checkouts')
plt.show()
series = df_monthly_checkouts['checkout_count']
series.hist()
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Histogram of monthly checkouts')
plt.show()
aplt(df_monthly_checkouts['checkout_count'])
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Correlation of monthly checkouts')
plt.show()

# ARIMA series prediction for total checkouts
Y = series.values
minelem=1000
min=(0,0,0)
for i in range(3):
    for j in range (3):
        for k in range(3):
            if minelem>evaluate_arima_model(Y, (i,j,k)):
                min=(i,j,k)
                minelem=evaluate_arima_model(Y, (i,j,k))
                print((i,j,k), evaluate_arima_model(Y, (i,j,k)))
            warnings.filterwarnings("ignore")
print(min)
plt.xlabel('date')
plt.ylabel('checkout')
plt.title('Optimalization of the ARIMA Model')
plt.show()

model = ARIMA(series, order=(1,0,0))
model_fit = model.fit()
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.xlabel('date')
plt.ylabel('spent money')
plt.title('Optimalization of the ARIMA Model')
plt.show()
residuals.plot(kind='kde')
print(residuals.describe())


"""
res = sm.tsa.ARMA(series, order=(1,0)).fit()
res.plot_predict()
plt.xlabel('date')
plt.ylabel('???')
plt.title('???')
plt.show()


print('A 2023. januári eladások várható értéke:', arima_sales_predict(30))
"""



## Non-coupon checkouts-spent money plots

# Calculate count difference non-coupon difference for each checkout
non_coupon=[]
for i in range(len(df_monthly_checkouts.checkout_count)):
    non_coupon.append(df_monthly_checkouts.checkout_count[i]-df_monthly_checkouts.coupon_count[i])
# Create Dataframe
df_none_coupon=pd.DataFrame(non_coupon, columns=['non'])

# Merge this Dataframe with dates
df_none_coupon=pd.concat([df_date,df_none_coupon], axis=1)
df_none_coupon=df_none_coupon.set_index('date')
df_none_coupon.index=pd.to_datetime(df_date)
df_none_coupon.plot()
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Non-coupon checkouts')
plt.show()

# Calculate spent difference non-coupon difference for each checkout
non_coupon_spent=[]
for i in range(len(df_date)):
    non_coupon_spent.append(df_monthly_checkouts.spent[i]-df_monthly_checkouts.coupon_spent[i])

# Create Dataframe
df_non_spent=pd.DataFrame(non_coupon, columns=['non_spent'])

# Merge this Dataframe with dates
df_non_spent=pd.concat([df_date,df_non_spent], axis=1)
df_non_spent=df_non_spent.set_index('date')
df_non_spent.index=pd.to_datetime(df_date)


X = np.array(datetime_to_int(df_non_spent.index))
plt.figure(figsize=(12, 7))
X = np.array(X)
sns.regplot(x=X, y=df_non_spent.non_spent)
plt.xlabel('date')
plt.ylabel('non spent money')
plt.title('Non-coupon spent linear regression')
plt.show()

# Calculate spent difference non-coupon difference for each checkout
ns=[]
for i in range(len(df_date)):
    ns.append(df_monthly_checkouts.spent[i]-df_monthly_checkouts.coupon_spent[i])
# Create Dataframe
df_ns=pd.DataFrame(ns, columns=['non_spent'])
# Merge this Dataframe with dates
df_ns=pd.concat([df_date,df_ns], axis=1)
df_ns=df_ns.set_index('date')
df_ns.index=pd.to_datetime(df_date)

plt.figure(figsize=(12, 7))
X = np.array(X)
sns.regplot(x=X, y=df_ns.non_spent)
plt.xlabel('date')
plt.ylabel('spent money')
plt.title('Non-coupon spent money')
plt.show()

result = seasonal_decompose(df_ns['non_spent'], model='multiplicative')
result.seasonal.plot()
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Seasonal part of monthly non-coupon checkouts')
plt.show()
result.trend.plot()
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Trend part of monthly non-coupon checkouts')
plt.show()
series = df_ns
series.hist()
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Histogram of monthly non-coupon checkouts')
plt.show()
aplt(series)
plt.xlabel('date')
plt.ylabel('checkout count')
plt.title('Correlation of monthly non-coupon checkouts')
plt.show()

# ARIMA series prediction for non-coupon checkouts
Y = series.values
minelem=1000
min=(0,0,0)
for i in range(3):
    for j in range (3):
        for k in range(3):
            if minelem>evaluate_arima_model(Y, (i,j,k)):
                min=(i,j,k)
                minelem=evaluate_arima_model(Y, (i,j,k))
                print((i,j,k), evaluate_arima_model(Y, (i,j,k)))
            warnings.filterwarnings("ignore")
print(min)

model = ARIMA(series, order=(1,0,0))
model_fit = model.fit()

print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.xlabel('date')
plt.ylabel('???')
plt.title('ARIMA')
plt.show()

residuals.plot(kind='kde')
plt.xlabel('date')
plt.ylabel('???')
plt.title('ARIMA')
plt.show()

print(residuals.describe())

res = sm.tsa.ARMA(series, order=(1,0)).fit()
res.plot_predict()
plt.xlabel('date')
plt.ylabel('???')
plt.title('ARIMA')
plt.show()
