#%%
import numpy as np # linear algebra
import pandas as pd 

from subprocess import check_output
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle


df = pd.read_csv("GlobalLandTemperaturesByCountry.csv", index_col=0, parse_dates=True)
df = df.loc[df["Country"]=="India"]
df = df.dropna(subset = ["AverageTemperature"])


temp = shuffle(df["AverageTemperature"])

# Calculating the number of days since the first day i.e 1796-01-01
days_since = pd.Series((temp.keys().year * 365 + temp.keys().month * 30 + temp.keys().day) - (1796*365 + 1*30 + 1),index=temp.keys(),name="DaysSince")
# Concatenating the series temp and days_sice to make the dataframe 'data'
data = pd.concat([temp, days_since],axis=1)
# Deleting the dataframe 'df'
del df

# Creating the polynomial features of degree 2
poly = PolynomialFeatures(degree=2)
data = pd.DataFrame(poly.fit_transform(data))


train = data.iloc[:-int(len(data)*.2)]
test = data.iloc[-int(len(data)*.2):]


regr = linear_model.LinearRegression()
regr.fit(train[[0,2,5]],train[1].to_frame())

# Printing the calculated Mean squared error on the test set and variance score
print('Coefficients: \n', regr.coef_) # Coefficient 
print("Mean squared error: %.2f" % pd.DataFrame.mean((regr.predict(test[[0,2,5]]) - test[1].to_frame()) ** 2))
print('Variance score: %.2f' % regr.score(test[[0,2,5]],test[1].to_frame()))

# Visualizing the data
test = test.sort_values(by=2)
plt.figure(figsize=(15, 10))
plt.scatter(test[2], test[1],  color='purple', alpha=0.5)
plt.plot(test[2], regr.predict(test[[0,2,5]]), color='blue',
         linewidth=2)
plt.savefig("Polynominal Regression Scatterplot.png")
# %%
