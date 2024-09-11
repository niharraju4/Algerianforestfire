# Algerian Forest Fire Prediction

## Table of Contents
1. [Introduction](#introduction)
2. [Load Libraries and Dataset](#load-libraries-and-dataset)
3. [Data Cleaning](#data-cleaning)
    - [Missing Values](#missing-values)
    - [Add Region Column](#add-region-column)
    - [Convert Data Types](#convert-data-types)
    - [Remove Null Values](#remove-null-values)
    - [Fix Column Names](#fix-column-names)
    - [Convert Columns to Integer](#convert-columns-to-integer)
    - [Convert Object Types to Float](#convert-object-types-to-float)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Drop Unnecessary Columns](#drop-unnecessary-columns)
    - [Encoding Categorical Variables](#encoding-categorical-variables)
    - [Visualization](#visualization)
    - [Correlation](#correlation)
    - [Box Plots](#box-plots)
    - [Monthly Fire Analysis](#monthly-fire-analysis)
5. [Save Cleaned Dataset](#save-cleaned-dataset)
6. [Feature Selection Based on Correlation](#feature-selection-based-on-correlation)
    - [Calculate Correlation Matrix](#calculate-correlation-matrix)
    - [Check for Multicollinearity](#check-for-multicollinearity)
    - [Drop Highly Correlated Features](#drop-highly-correlated-features)
7. [Feature Scaling](#feature-scaling)
    - [Visualize Effect of Scaling](#visualize-effect-of-scaling)
8. [Linear Regression Model](#linear-regression-model)
9. [Lasso Regression](#lasso-regression)
10. [Hyperparameter Tuning for Lasso Regression](#hyperparameter-tuning-for-lasso-regression)
11. [Ridge Regression](#ridge-regression)
12. [Hyperparameter Tuning for Ridge Regression](#hyperparameter-tuning-for-ridge-regression)
13. [Elastic Net](#elastic-net)
14. [Hyperparameter Tuning for Elastic Net](#hyperparameter-tuning-for-elastic-net)
15. [Save Models and Scaler](#save-models-and-scaler)
16. [Results](#results)
17. [Conclusion](#conclusion)
18. [Future Work](#future-work)

## Introduction
This project aims to predict the Fire Weather Index (FWI) using various regression models. The dataset used is the Algerian Forest Fires dataset, which has been cleaned and preprocessed.
Algerian Forest Fires Dataset
Data Set Information:

The dataset includes 244 instances that regroup a data of two regions of Algeria,namely the Bejaia region located in the northeast of Algeria and the Sidi Bel-abbes region located in the northwest of Algeria.

122 instances for each region.
The period from June 2012 to September 2012. The dataset includes 11 attribues and 1 output attribue (class) The 244 instances have been classified into fire(138 classes) and not fire (106 classes) classes.

Attribute Information:
1. Date : (DD/MM/YYYY) Day, month ('june' to 'september'), year (2012) Weather data observations
2. Temp : temperature noon (temperature max) in Celsius degrees: 22 to 42
3. RH : Relative Humidity in %: 21 to 90
4. Ws :Wind speed in km/h: 6 to 29
5. Rain: total day in mm: 0 to 16.8 FWI Components
6. Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
7. Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
8. Drought Code (DC) index from the FWI system: 7 to 220.4
9. Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
10. Buildup Index (BUI) index from the FWI system: 1.1 to 68
11. Fire Weather Index (FWI) Index: 0 to 31.1
12. Classes: two classes, namely Fire and not Fire''
## Load Libraries and Dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataset = pd.read_csv(r'N:\Personal_Projects\Machine-Learning\Algerianforestfire\Algerian_forest_fires_dataset_UPDATE.csv', header=1)
dataset.head()
dataset.tail()
dataset.info()
```

## Data Cleaning
### Missing Values
```python
dataset.isnull().sum()
dataset[dataset.isnull().any(axis=1)]
```

### Add Region Column
```python
dataset.loc[:122, "Region"] = 0
dataset.loc[122:, "Region"] = 1
df = dataset
df.info()
```

### Convert Data Types
```python
df[['Region']] = df[['Region']].astype(int)
df.head()
df.isnull().sum()
```

### Remove Null Values
```python
df = df.dropna().reset_index(drop=True)
df.isnull().sum()
df.iloc[122]
df.iloc[[122]]
df.drop(122).reset_index(drop=True)
df.iloc[[122]]
df.columns
```

### Fix Column Names
```python
df.columns = df.columns.str.strip()
df.columns
df.info()
```

### Convert Columns to Integer
```python
columns_to_convert = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws']
for column in columns_to_convert:
    df = df[pd.to_numeric(df[column], errors='coerce').notnull()]

df[columns_to_convert] = df[columns_to_convert].astype(int)
df.info()
```

### Convert Object Types to Float
```python
objects = [features for features in df.columns if df[features].dtypes == 'O']
for i in objects:
    if i != 'Classes':
        df[i] = df[i].astype(float)
df.info()
objects
df.describe()
df.info()
df.head()
```

## Save Cleaned Dataset
```python
df.to_csv('Algerian_forest_fires_cleaned_dataset.csv', index=False)
```

## Exploratory Data Analysis
### Drop Unnecessary Columns
```python
df1 = df.drop(['day', 'month', 'year'], axis=1)
df1.info()
df1.head()
df1['Classes'].value_counts()
```

### Encoding Categorical Variables
```python
df1['Classes'] = np.where(df1['Classes'].str.contains('not fire'), 0, 1)
df1.head()
df1.tail()
df1['Classes'].value_counts()
print(df1['Classes'].unique())
```

### Visualization
```python
import matplotlib.pyplot as plt
print(plt.style.available)

# List available styles
print(plt.style.available)

# Choose a valid style, e.g., 'ggplot'
plt.style.use('ggplot')

# Plot density plot for all features
df1.hist(bins=50, figsize=(20, 15))
plt.show()

## Percentage for Pie Chart
percentage = df1['Classes'].value_counts(normalize=True) * 100

# plotting piechart
classlabels = ["Fire", "Not Fire"]
plt.figure(figsize=(12, 7))
plt.pie(percentage, labels=classlabels, autopct='%1.1f%%')
plt.title("Pie Chart of Classes")
plt.show()
```

### Correlation
```python
df1.corr()
sns.heatmap(df1.corr())
```

### Box Plots
```python
sns.boxplot(df['FWI'], color='green')
df.head()
df['Classes'] = np.where(df['Classes'].str.contains('not fire'), 'not fire', 'fire')
```

### Monthly Fire Analysis
```python
dftemp = df.loc[df['Region'] == 1]
plt.subplots(figsize=(13, 6))
sns.set_style('whitegrid')
sns.countplot(x='month', hue='Classes', data=df)
plt.ylabel('Number of Fires', weight='bold')
plt.xlabel('Months', weight='bold')
plt.title("Fire Analysis of Sidi-Bel Regions", weight='bold')

dftemp = df.loc[df['Region'] == 0]
plt.subplots(figsize=(13, 6))
sns.set_style('whitegrid')
sns.countplot(x='month', hue='Classes', data=df)
plt.ylabel('Number of Fires', weight='bold')
plt.xlabel('Months', weight='bold')
plt.title("Fire Analysis of Bejaia Regions", weight='bold')
```

## Feature Selection Based on Correlation
### Calculate Correlation Matrix
```python
X_train.corr()
```

### Check for Multicollinearity
```python
plt.figure(figsize=(12,10))
corr = X_train.corr()
sns.heatmap(corr, annot=True)
```

### Drop Highly Correlated Features
```python
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.85)
X_train.drop(corr_features, axis=1, inplace=True)
X_test.drop(corr_features, axis=1, inplace=True)
X_train.shape, X_test.shape
```

## Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled
X_test_scaled
```

### Visualize Effect of Scaling
```python
plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=X_train)
plt.title('X_train Before Scaling')
plt.subplot(1, 2, 2)
sns.boxplot(data=X_train_scaled)
plt.title('X_train After Scaling')
```

## Linear Regression Model
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

linearreg = LinearRegression()
linearreg.fit(X_train_scaled, y_train)
y_pred = linearreg.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean absolute error", mae)
print("R2 score", score)
plt.scatter(y_test, y_pred)
```

## Lasso Regression
```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test, y_pred)
```

## Hyperparameter Tuning for Lasso Regression
```python
from sklearn.linear_model import LassoCV

lass = LassoCV(cv=5)
lass.fit(X_train_scaled, y_train)
y_pred = lass.predict(X_test_scaled)
plt.scatter(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
```

## Ridge Regression
```python
from sklearn.linear_model import Ridge

rid = Ridge()
rid.fit(X_train_scaled, y_train)
y_pred = rid.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test, y_pred)
```

## Hyperparameter Tuning for Ridge Regression
```python
from sklearn.linear_model import RidgeCV

ridgecv = RidgeCV(cv=5)
ridgecv.fit(X_train_scaled, y_train)
y_pred = ridgecv.predict(X_test_scaled)
plt.scatter(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
```

## Elastic Net
```python
from sklearn.linear_model import ElasticNet

elnet = ElasticNet()
elnet.fit(X_train_scaled, y_train)
y_pred = elnet.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test, y_pred)
```

## Hyperparameter Tuning for Elastic Net
```python
from sklearn.linear_model import ElasticNetCV

elasticcv = ElasticNetCV(cv=5)
elasticcv.fit(X_train_scaled, y_train)
y_pred = elasticcv.predict(X_test_scaled)
plt.scatter(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
```

## Save Models and Scaler
```python
import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(rid, open('rid.pkl', 'wb'))
```

## Results

### Linear Regression
- Mean Absolute Error: [Value]
- R2 Score: [Value]

### Lasso Regression
- Mean Absolute Error: [Value]
- R2 Score: [Value]

### LassoCV
- Mean Absolute Error: [Value]
- R2 Score: [Value]

### Ridge Regression
- Mean Absolute Error: [Value]
- R2 Score: [Value]

### RidgeCV
- Mean Absolute Error: [Value]
- R2 Score: [Value]

### Elastic Net
- Mean Absolute Error: [Value]
- R2 Score: [Value]

### ElasticNetCV
- Mean Absolute Error: [Value]
- R2 Score: [Value]

## Conclusion
This project demonstrates the application of various regression models to predict the Fire Weather Index (FWI). The models were evaluated based on Mean Absolute Error and R2 Score. Hyperparameter tuning was performed to improve model performance. The best model can be selected based on the evaluation metrics.

## Future Work
- Explore more advanced models such as Random Forest, Gradient Boosting, or Neural Networks.
- Perform feature engineering to create new features that might improve model performance.
- Conduct a more thorough hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

---

This documentation provides a comprehensive overview of your project, including the steps taken, the code used, and the results obtained. Make sure to replace `[Value]` with the actual values obtained from your analysis.
