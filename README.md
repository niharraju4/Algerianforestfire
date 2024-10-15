# Algerian Forest Fire Prediction

## Table of Contents
1. [Introduction](#introduction)
2. [Import Libraries and Load Data](#import-libraries-and-load-data)
3. [Data Preprocessing](#data-preprocessing)
    - [Drop Unnecessary Columns](#drop-unnecessary-columns)
    - [Encoding Categorical Variables](#encoding-categorical-variables)
4. [Define Independent and Dependent Features](#define-independent-and-dependent-features)
5. [Train-Test Split](#train-test-split)
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

## Import Libraries and Load Data
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv(r'N:\Personal_Projects\Machine-Learning\Algerianforestfire\Algerian_forest_fires_cleaned_dataset.csv')
df.head()
df.tail()
df.info()
df.dtypes
df.describe()
df.info()
df.columns
```

## Data Preprocessing
### Drop Unnecessary Columns
```python
df.drop(['day', 'month', 'year'], axis=1, inplace=True)
df.head()
```

### Encoding Categorical Variables
```python
df['Classes'] = np.where(df['Classes'].str.contains("not fire"), 0, 1)
df.head()
df['Classes'].value_counts()
```

## Define Independent and Dependent Features
```python
X = df.drop('FWI', axis=1)
y = df['FWI']

X.head()
y
```

## Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.shape, X_test.shape
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
- Mean Absolute Error: 0.5468236465249978
- R2 Score:            0.9847657384266951

### Lasso Regression
- Mean Absolute Error: 1.1331759949144085
- R2 Score:            0.9492020263112388

### LassoCV
- Mean Absolute Error: 0.6199701158263433
- R2 Score:            0.9820946715928275

### Ridge Regression
- Mean Absolute Error:  0.5642305340105693
- R2 Score:             0.9842993364555513

### RidgeCV
- Mean Absolute Error: 0.5642305340105693
- R2 Score:            0.9842993364555513

### Elastic Net
- Mean Absolute Error:  1.8822353634896
- R2 Score:             0.8753460589519703

### ElasticNetCV
- Mean Absolute Error: 0.6575946731430904
- R2 Score:            0.9814217587854941

## Conclusion
This project demonstrates the application of various regression models to predict the Fire Weather Index (FWI). The models were evaluated based on Mean Absolute Error and R2 Score. Hyperparameter tuning was performed to improve model performance. The best model can be selected based on the evaluation metrics.

## Future Work
- Explore more advanced models such as Random Forest, Gradient Boosting, or Neural Networks.
- Perform feature engineering to create new features that might improve model performance.
- Conduct a more thorough hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

---

