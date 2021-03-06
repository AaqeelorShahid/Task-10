#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.regression.linear_model as smf
import statsmodels.api as sm
from matplotlib import pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('GlobalLandTemperaturesByCountry.csv')
#selecting data only for United States
data=data[data['Country']=='United States']

#Calculating the missing values
null=data.isnull().sum()
print("Null values in the data ")
print(null)
data=data.dropna()

#Data processing step 
null_After=data.isnull().sum()
print("Null values in the data ")
print(null_After)
data_type=data.dtypes
print("Data type of the dataframe columns are ")
print(data_type)

data[['dt']]=pd.to_datetime(data['dt'])
# data before 1870 is not good so ...
data_new=data[data['dt']>'1870']

#grouping the data in terms of year and finding the mean of the average temp
mean_temp_inc_over_year=data_new.groupby(data_new.dt.dt.year).mean()
print("Average temperature per year in United States")
print(mean_temp_inc_over_year)

mean_temp_inc_over_year.reset_index(level=0, inplace=True)

x = mean_temp_inc_over_year['dt']
y = mean_temp_inc_over_year['AverageTemperature']
# x1 =  sm.add_constant(x)

X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


#Training and predicting usign Statsmodel OLS



model = smf.OLS(endog=y_train,exog=X_train).fit()
predict = model.predict(X_test)
print("Predicted values ")
print(predict)
print("Actual values")
print(y_test)
#RMSE value
rms = sqrt(mean_squared_error(y_test, predict))
print(rms)

plt.figure(figsize=(12,8))
plt.scatter(X_test,y_test)
plt.plot(X_test,predict)
plt.savefig("Linear Regression Scatterplot.png")
plt.show()

# Regression Summary
print("Summary of the regrssion model")
print(model.summary())
# %%
