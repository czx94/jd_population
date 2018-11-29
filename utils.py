from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

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
    assert len(pred.index) == len(groundtruth.index)
    predict = []
    real = []
    for i in range(len(pred.index)):
        predict.append(pred[i]['dwell', 'flow_in', 'flow_out'].tolist())
        real.append(groundtruth[i]['dwell', 'flow_in', 'flow_out'].tolist())
        
    result = RMSLE(predict, real)
    print('RMSLE error:', result)