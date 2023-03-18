import pandas as pd
import numpy as np
import collect_new_data
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, max_error, r2_score

from sklearn import linear_model
def create_lagged_data(target:np.array, n_lag=7):
    '''
    target: np.ndarray containing water level values
    It can be 1D or 2D.
    '''
    target = np.array(target).flatten()
    y_lag = target[n_lag:].copy()
    y_lag = np.array(y_lag).flatten()
    X_lag = []
    for i in range (n_lag, target.shape[0]):
        X_lag.append(target[i-n_lag:i])
    X_lag = np.array(X_lag)
    return X_lag, y_lag

def model_training(dataset):
    #Structuring dataset to feed an autoregressive model
    n_lag = 7
    array_water_levels = np.array(dataset['belgrade_water_level_cm']).reshape(-1,1)
    X, y = create_lagged_data(array_water_levels, n_lag)

    #Splitting dataset into training and test sets.
    #Test set has a length of 1/3 of the amount of data.
    test_size = round(1/3 * y.shape[0])
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    X_train = X[:-test_size]
    X_test = X[-test_size:]

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    return model
