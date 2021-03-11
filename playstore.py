import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib


import os
print(os.listdir("../GooglePlayStore"))

df = pd.read_csv('../GooglePlayStore/googleplaystore.csv')

#Cleaning the dataset
#Looking dataset Column
df.info()

#Checking out the nul value and dealing with them
df.isnull().sum()

# 'Rating' column has most null value and it is our the dependent variable.
# The best way to fill missing values might be using the median instead of mean.
df['Rating'] = df['Rating'].fillna(df['Rating'].median())

# convert reviews to numeric
df['Reviews'] = pd.to_numeric(df.Reviews, errors = 'coerce')

#Let's look at the apps in the data 
print(df.App.value_counts().head(20))

# Let's check out the App categories
print(df.Category.value_counts())

#Now remove the catagories 1.9 which is irrelevant for our model
df[df['Category'] == '1.9']
df = df.drop([10472])

#Drops other duplicate entries keeping the App with the highest reviews
df.drop_duplicates('App', keep = 'last', inplace = True)
print(df.App.value_counts())

#lets Deal with the Size of Apps
print(df.Size.value_counts())

#Now Convert non nemurice value to 'NaN' value
df['Size'][df['Size' ] == 'Varies with devices'] = np.nan

#Now Convert M with Million and K with Thousand
df['Size'] = df.Size.str.replace('M', 'e6')
df['Size'] = df.Size.str.replace('K', 'e3')

#Now Convert to the nemuric value
df['Size'] = pd.to_numeric(df['Size'], errors = 'coerce')

#Replace the "NaN' Value with Mean 
df['Size'] = df['Size'].fillna(df['Size'].mean())

#Now lets Check the Install
df.Installs.value_counts()

df['Installs'] = df.Installs.str.replace('+', '')
df['Installs'] = df.Installs.str.replace(',', '')

#Now Convert to the nemuric value
df['Installs'] = pd.to_numeric(df['Installs'], errors = 'coerce')

#Now Build the Machine Learning Model
#Now Lets take matrix of features
x = df.iloc[:,3:6].values
y = df.iloc[:,2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting Random Forest Regression to tranning set
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
regressor.fit(x_train, y_train)
model = 'playstore_model.sav'
joblib.dump(regressor, model)

'''
#Visualising the Predicted Result
import matplotlib.pyplot as plt
plt.plot(y_test, color = 'red', label = 'Actual')
plt.plot(y_pred, color = 'blue', label = 'Predicted')
plt.xlabel('Number of Reviews')
plt.ylabel('Rating')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
'''
