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
    total_flow_train, total_flow_val = train_val_split(flow_train)

    # transition_train = pd.read_csv('../data/transition_train.csv')

    #read all sample path
    sample_data_path = '../../data/flow/'
    all_sample = os.listdir(sample_data_path)

    #grid search for n
    for i in range(3, 16):
        gt_for_each_sample = []
        result_for_each_sample = []
        for sample in tqdm(all_sample):
            city, district = sample[:-4].split('_')
            # We'll start by playing the flow, then the transition
            ###statistic with mod7, week
            flow_sample = pd.read_csv(sample_data_path + sample)
            sample_train, sample_val = train_val_split(flow_sample)
            # print(sample_train.head(), sample_val.head())

            ##sample
            # group by modn
            sample_result_mod_n = stat_mod_n(i, sample_train)
            # print(result_sample_by_mod7)

            # #prediction of the coming 15 days based on mod7 stat
            columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
            flow_sample_prediction = pd.DataFrame(columns=columns)
            for d in range(15):
                day = 20180215 + d
                dwell = sample_result_mod_n[(259 + d) % i]['dwell']
                flow_in = sample_result_mod_n[(259 + d) % i]['flow_in']
                flow_out = sample_result_mod_n[(259 + d) % i]['flow_out']

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

        eval(result, gt)
        print(i)


