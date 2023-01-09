# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:24:17 2023

@author: Emre
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import statsmodels.formula.api as smf


trainf = pd.read_csv('Car_features_train.csv')
trainp = pd.read_csv('Car_prices_train.csv')

train = pd.merge(trainf, trainp)

# # Create a stat model - OLS for ordinary Least Squares
    # Dataframe is the second input
    # X(engineSize) and Y(price) go to the formula input
# Notation for SLR is 'Y~X'
ols_object = smf.ols(formula='price~engineSize', data=train)

# Training the model - .fit() method
model = ols_object.fit()

# The 'model' variable is the trained SLR model

# Returning numerical/statistical analysis - .summary() method
print(model.summary())


# Visualization

sns.regplot(x='engineSize', y='price', data=train)
plt.show()

# Prediction/Testing

testf = pd.read_csv('Car_features_test.csv')
testp = pd.read_csv('Car_prices_test.csv')

# Getting the predictions - .predict()

pred_price = model.predict(testf)

# Numeric assessment - RMSE
print(np.sqrt(((testp['price']-pred_price)**2).mean()))


# Visualization
# Comparison of the real responses and the predictions
    # Either scatterplot and compare against y=x
    # or get (real-predicted) compare against the y=0






















