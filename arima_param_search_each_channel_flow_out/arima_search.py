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
    #setting log
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


    loss_table = {}
    channels = ['flow_out']

    #search for channel
    for channel in channels:
        # grid search
        for a in range(12):
            for b in range(3):
                for c in range(12):
                    logger.info(channel+':'+str(a) + '_' + str(b) + '_' + str(c))
                    loss = 100
                    gt_for_each_sample = []
                    result_for_each_sample = []

                    try:
                        for sample in tqdm(all_sample):
                            city, district = sample[:-4].split('_')

                            flow_sample = pd.read_csv(sample_data_path + sample)

                            sample_train, sample_val = train_val_split(flow_sample)

                            # first condider dwell
                            channel_predict = predict_by_ARIMA(sample_train, channel, param=(a, b, c), offset=0)

                            columns = ['date_dt', 'city_code', 'district_code', channel]
                            flow_sample_prediction = pd.DataFrame(columns=columns)
                            for d in range(15):
                                day = 20180215 + d
                                channel_used = channel_predict[d]
                                flow_sample_prediction.loc[d] = {columns[0]: day,
                                                                 columns[1]: city,
                                                                 columns[2]: district,
                                                                 columns[3]: channel}

                            gt_for_each_sample.append(sample_val)
                            result_for_each_sample.append(flow_sample_prediction)

                        result = pd.concat(result_for_each_sample).reset_index(drop=True)
                        gt = pd.concat(gt_for_each_sample).reset_index(drop=True)

                        loss = eval(result, gt, [channel])

                    except:
                        pass

                    logger.info(loss)
                    loss_table[str(a) + '_' + str(b) + '_' + str(c)] = loss

        loss_table = sorted(loss_table.items(), key=lambda item: item[1])
        with open(channel + 'loss_table.json', 'w') as f:
            json.dump(loss_table, f)