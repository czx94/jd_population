from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

def MSE(x, y):
    return mean_squared_error(x, y)

def RMSE(x, y):
    return MSE(x, y) ** 0.5

def MAE(x, y):
    return mean_absolute_error(x, y)

def MSLE(x, y):
    return mean_squared_log_error(x, y)

def RMSLE(x, y):
    return MSLE(x, y) ** 0.5
