import sys,os
project_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
sys.path.insert(0,project_root_dir)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from preprocessing import train_val_split
from utils import *

def stat_mod_n(n, df, ds_type = 'flow'):
    group_by_mod_n = df.groupby(df.index % n)
    result_by_mod_n = {}
    for mod, v in group_by_mod_n:
        if ds_type == 'flow':
            result_by_mod_n[mod] = v.drop(['city_code', 'district_code', 'date_dt'], axis=1).mean().to_dict()
        else:
            result_by_mod_n[mod] = v.drop(['o_city_code', 'o_district_code', 'date_dt'], axis=1).mean().to_dict()

    return result_by_mod_n

if __name__ == '__main__':
    #read data
    flow_train = pd.read_csv('../../data/flow_train.csv')
    total_flow_train, total_flow_val = train_val_split(flow_train, val_len=15*97)

    # transition_train = pd.read_csv('../data/transition_train.csv')

    #read all sample path
    sample_data_path = '../../data/flow/'
    all_sample = os.listdir(sample_data_path)

    gt_for_each_sample = []
    result_for_each_sample = []
    for sample in tqdm(all_sample):
        city, district = sample[:-4].split('_')
        #We'll start by playing the flow, then the transition
        ###statistic with mod7, week
        flow_sample = pd.read_csv(sample_data_path + sample)
        sample_train, sample_val = train_val_split(flow_sample)
        # print(sample_train.head(), sample_val.head())

        ##sample
        #group by mod7
        sample_result_mod_7 = stat_mod_n(7, sample_train)
        # print(result_sample_by_mod7)

        # #group by mod30, month
        sample_result_mod_30 = stat_mod_n(30, sample_train)

        ##total
        #group by mod7
        total_result_mod_7 = stat_mod_n(7, flow_train)

        # #group by mod30, month
        # total_result_mod_30 = stat_mod_n(30, flow_train)

        # #prediction of the coming 15 days based on mod7 stat
        columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        flow_sample_prediction = pd.DataFrame(columns = columns)
        for d in range(15):
            day = 20180215 + d
            dwell = sample_result_mod_7[(259 + d) % 7]['dwell'] * 0.5 + total_result_mod_7[(259 + d) % 7]['dwell'] * 0.1 + sample_result_mod_30[(259 + d) % 30]['dwell'] * 0.4
            flow_in = sample_result_mod_7[(259 + d) % 7]['flow_in'] * 0.5 + total_result_mod_7[(259 + d) % 7]['flow_in'] * 0.1 + sample_result_mod_30[(259 + d) % 30]['flow_in'] * 0.4
            flow_out = sample_result_mod_7[(259 + d) % 7]['flow_out'] * 0.5 + total_result_mod_7[(259 + d) % 7]['flow_out'] * 0.1 + sample_result_mod_30[(259 + d) % 30]['flow_out'] * 0.4

            flow_sample_prediction.loc[d] = {columns[0]:day,
                                            columns[1]:city,
                                            columns[2]:district,
                                            columns[3]:dwell,
                                            columns[4]:flow_in,
                                            columns[5]:flow_out}


        gt_for_each_sample.append(sample_val)
        result_for_each_sample.append(flow_sample_prediction)

    result = pd.concat(result_for_each_sample).reset_index(drop=True)
    gt = pd.concat(gt_for_each_sample).reset_index(drop=True)

    eval(result, gt)


