# -*- coding: utf-8 -*-

import sys, os

project_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
sys.path.insert(0, project_root_dir)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from preprocessing import train_val_split
from utils import *
import json

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
    if not os.path.isdir('loss_tables'):
        os.mkdir('loss_tables')
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

    channels = ['flow_out']

    top5_loss_param_each_sample = {}
    # search for channel
    for channel in channels:
        for sample in tqdm(all_sample):
            loss_table = {}

            city, district = sample[:-4].split('_')

            flow_sample = pd.read_csv(sample_data_path + sample)

            sample_train, sample_val = train_val_split(flow_sample)
            # grid search
            for a in range(12):
                for b in range(3):
                    for c in range(12):
                        loss = 100

                        try:
                            # first condider flow_out
                            channel_predict = predict_by_ARIMA(sample_train, channel, param=(a, b, c), offset=0)

                            columns = ['date_dt', 'city_code', 'district_code', channel]
                            flow_sample_prediction = pd.DataFrame(columns=columns)
                            for d in range(15):
                                day = 20180215 + d
                                channel_used = channel_predict[d]
                                flow_sample_prediction.loc[d] = {columns[0]: day,
                                                                 columns[1]: city,
                                                                 columns[2]: district,
                                                                 columns[3]: channel_used}

                            loss = eval(flow_sample_prediction, sample_val, [channel])

                        except:
                            pass

                        if loss != 100:
                            logger.info(district + '_' + channel + ':' + str(a) + '_' + str(b) + '_' + str(c))
                            logger.info(loss)

                            loss_table[str(a) + '_' + str(b) + '_' + str(c)] = loss

            loss_table = sorted(loss_table.items(), key=lambda item: item[1])
            loss_table = loss_table[:5]

            top5_loss_param_each_sample[city + '_' + district + '_' + channel] = loss_table

        with open('./loss_tables/' + channel + '_loss_table_each_sample.json', 'w') as f:
            json.dump(loss_table, f)

