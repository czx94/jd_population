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


def decomp(data):
    decomposition = seasonal_decompose(data, freq=7, two_sided=False)

    trend = decomposition.trend

    seasonal = decomposition.seasonal

    residual = decomposition.resid

    return trend, seasonal, residual


if __name__ == '__main__':
    # read data
    flow_train = pd.read_csv('../../data/flow_train.csv')

    # read sample data paths
    sample_data_path = '../../data/flow/'
    all_sample = os.listdir(sample_data_path)
    gt_for_each_sample = []
    result_for_each_sample = []
    for sample in tqdm(all_sample):

        city, district = sample[:-4].split('_')

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

        gt_for_each_sample.append(sample_val)
        result_for_each_sample.append(flow_sample_prediction)

    result = pd.concat(result_for_each_sample).reset_index(drop=True)
    gt = pd.concat(gt_for_each_sample).reset_index(drop=True)

    for channel in ['dwell', 'flow_in', 'flow_out']:
        eval(result, gt, channels = [channel])