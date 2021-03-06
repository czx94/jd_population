# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:46:31 2018

@author: wyq
"""

import sys, os

project_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
sys.path.insert(0, project_root_dir)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from preprocessing import *
from utils import *

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

import logging
import time

def decomp(data):
    decomposition = seasonal_decompose(data, freq=7, two_sided=False)

    trend = decomposition.trend

    seasonal = decomposition.seasonal

    residual = decomposition.resid

    return trend, seasonal, residual


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
    flow_train = pd.read_csv('../../data/flow_train.csv')

    # read sample data paths
    sample_data_path = '../../data/flow/'
    all_sample = os.listdir(sample_data_path)

    for sample in tqdm(all_sample):

        city, district = sample[:-4].split('_')

        logger.info(sample)

        flow_sample = pd.read_csv(sample_data_path + sample)

        sample_train, sample_val = train_val_split(flow_sample)

        # first condider dwell
        dwell_predict = predict_by_ARIMA(sample_train, 'dwell', param=(1, 1, 6), offset=0)

        # flow_in
        flow_in_predict = predict_by_ARIMA(sample_train, 'flow_in', param=(1, 1, 6), offset=0)

        # flow_out
        flow_out_predict = predict_by_ARIMA(sample_train, 'flow_out', param=(1, 1, 6), offset=0)

        columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        flow_sample_prediction = pd.DataFrame(columns=columns)
        for d in range(15):
            day = 20180215 + d
            dwell = dwell_predict[d]
            flow_in = flow_in_predict[d]
            flow_out = flow_out_predict[d]
            flow_sample_prediction.loc[d] = {columns[0]: day,
                                             columns[1]: city,
                                             columns[2]: district,
                                             columns[3]: dwell,
                                             columns[4]: flow_in,
                                             columns[5]: flow_out}

        for channel in ['dwell', 'flow_in', 'flow_out']:
            loss = eval(sample_val, flow_sample_prediction, channels=[channel])
            logger.info(channel+':'+str(loss))


