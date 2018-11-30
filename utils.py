from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

def MSE(y1, y2):
    return mean_squared_error(y1, y2)

def RMSE(y1, y2):
    return MSE(y1, y2) ** 0.5

def MAE(y1, y2):
    return mean_absolute_error(y1, y2)

def MSLE(y1, y2):
    return mean_squared_log_error(y1, y2)

def RMSLE(y1, y2):
    return MSLE(y1, y2) ** 0.5

def eval(pred, groundtruth):
    assert len(pred) == len(groundtruth)
    predict = []
    real = []
    for i in range(len(pred.index)):
        predict.append(pred.loc[i, ['dwell', 'flow_in', 'flow_out']].tolist())
        real.append(groundtruth.loc[i, ['dwell', 'flow_in', 'flow_out']].tolist())

    result = RMSLE(predict, real)
    print('RMSLE error:', result)
    return result

def predict_by_ARIMA(data, type, param=(1, 1, 6), offset = 1):
    trend, seasonal, residual = decomp(data[type])

    trend.dropna(inplace=True)

    trend_model = ARIMA(trend, order=param).fit(disp=-1, method='css')
    n = 15
    trend_pred = trend_model.forecast(n)[0]
    season_part = seasonal[offset:n+offset]
    predict = pd.Series(trend_pred, index=season_part.index, name='predict')

    return predict + season_part

def decomp(data):
    decomposition = seasonal_decompose(data, freq=7, two_sided=False)

    trend = decomposition.trend

    seasonal = decomposition.seasonal

    residual = decomposition.resid

    return trend, seasonal, residual