import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from preprocessing import train_val_split

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

from utils import *
import logging
import time
if __name__ == '__main__':
    # setting log
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    fg = str(int(time.time()))
    log_name = 'log_' + fg + '.txt'
    handler = logging.FileHandler("./logs/" + log_name)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("Start print log")

    # read data
    flow_train = pd.read_csv('../data/flow_train.csv')

    # read sample data paths
    sample_data_path = '../data/flow/'
    all_sample = os.listdir(sample_data_path)
    gt_for_each_sample = []
    result_for_each_sample = []
    for sample in tqdm(all_sample):
        city, district = sample[:-4].split('_')

        flow_sample = pd.read_csv(sample_data_path + sample)

        sample_train = flow_sample

        # first condider dwell
        try:
            dwell_param = param_json_reader(city, district, 'dwell')
            dwell_predict = predict_by_ARIMA(sample_train, 'dwell', param=dwell_param)
        except:
            dwell_param = (1, 1, 5)
            dwell_predict = predict_by_ARIMA(sample_train, 'dwell', param=dwell_param)
        logger.info(district + '_dwell:' + '_'.join(list(map(lambda x: str(x), dwell_param))))

        # flow_in
        try:
            flow_in_param = param_json_reader(city, district, 'flow_in')
            flow_in_predict = predict_by_ARIMA(sample_train, 'flow_in', param=flow_in_param)
        except:
            flow_in_param = (1, 1, 5)
            flow_in_predict = predict_by_ARIMA(sample_train, 'flow_in', param=flow_in_param)
        logger.info(district + '_flow_in:' + '_'.join(list(map(lambda x: str(x), flow_in_param))))

        # flow_out
        try:
            flow_out_param = param_json_reader(city, district, 'flow_out')
            flow_out_predict = predict_by_ARIMA(sample_train, 'flow_out', param=flow_out_param)
        except:
            flow_out_param = (1, 1, 5)
            flow_out_predict = predict_by_ARIMA(sample_train, 'flow_out', param=flow_out_param)
        logger.info(district + '_flow_out:' + '_'.join(list(map(lambda x: str(x), flow_out_param))))

        columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        flow_sample_prediction = pd.DataFrame(columns=columns)
        
        for d in range(15):
            day = 20180302 + d
            dwell = dwell_predict[d+1]
            flow_in = flow_in_predict[d+1]
            flow_out = flow_out_predict[d+1]
            flow_sample_prediction.loc[d] = {columns[0]: day,
                                             columns[1]: city,
                                             columns[2]: district,
                                             columns[3]: dwell,
                                             columns[4]: flow_in,
                                             columns[5]: flow_out}

        # gt_for_each_sample.append(sample_val)
        result_for_each_sample.append(flow_sample_prediction)

    result = pd.concat(result_for_each_sample)
    result.to_csv('./result/prediction.csv', index=False, header=None)
    print('csv generated')