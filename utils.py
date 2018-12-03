from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import json

from preprocessing import *

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

def eval(pred, groundtruth, channels):
    assert len(pred) == len(groundtruth)
    predict = []
    real = []
    for i in range(len(pred.index)):
        predict.append(pred.loc[i, channels].tolist())
        real.append(groundtruth.loc[i, channels].tolist())

    result = RMSLE(predict, real)
    print('RMSLE error: ',result)
    print('Channels ', channels)
    return result

def predict_by_ARIMA(data, type, param=(1, 1, 5), offset = 1, freq=7):
    trend, seasonal, residual = decomp(data[type], freq=freq)

    trend.dropna(inplace=True)

    trend_model = ARIMA(trend, order=param).fit(disp=-1, method='css')
    n = 15
    trend_pred = trend_model.forecast(n)[0]
    season_part = seasonal[offset:n+offset]
    predict = pd.Series(trend_pred, index=season_part.index, name='predict')

    return predict + season_part

def decomp(data, freq=7):
    decomposition = seasonal_decompose(data, freq=freq, two_sided=False)

    trend = decomposition.trend

    seasonal = decomposition.seasonal

    residual = decomposition.resid

    return trend, seasonal, residual

def stat_mod_n(n, df, ds_type = 'flow'):
    group_by_mod_n = df.groupby(df.index % n)
    result_by_mod_n = {}
    for mod, v in group_by_mod_n:
        if ds_type == 'flow':
            result_by_mod_n[mod] = v.drop(['city_code', 'district_code', 'date_dt'], axis=1).mean().to_dict()
        else:
            result_by_mod_n[mod] = v.drop(['o_city_code', 'o_district_code', 'date_dt'], axis=1).mean().to_dict()

    return result_by_mod_n

def param_json_reader(city, district, channel_type):
    json_path = '../arima_param_search_each_channel_'+channel_type+'_each_sample/loss_tables/'+channel_type+'_loss_table_each_sample.json'
    with open(json_path, 'r') as load_f:
        loss_dict = json.load(load_f)

    param = loss_dict[city+'_'+district+'_'+channel_type][0][0].split('_')
    param = tuple(map(lambda a: int(a), param))

    return param