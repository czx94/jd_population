import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from .preprocessing import train_val_split

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
    flow_train = pd.read_csv('../data/flow_train.csv')
    # transition_train = pd.read_csv('../data/transition_train.csv')

    #read all sample path
    sample_data_path = '../data/flow/'
    all_sample = os.listdir(sample_data_path)

    result_for_each_sample = []
    for sample in tqdm(all_sample):
        city, district = sample[:-4].split('_')
        #We'll start by playing the flow, then the transition
        ###statistic with mod7, week
        flow_sample = pd.read_csv(sample_data_path + sample)

        ##sample
        #group by mod7
        sample_result_mod_7 = stat_mod_n(7, flow_sample)
        # print(result_sample_by_mod7)

        # #group by mod30, month
        sample_result_mod_30 = stat_mod_n(30, flow_sample)

        ##total
        #group by mod7
        total_result_mod_7 = stat_mod_n(7, flow_train)

        # #group by mod30, month
        # total_result_mod_30 = stat_mod_n(30, flow_train)

        # #prediction of the coming 15 days based on mod7 stat
        columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
        flow_sample_prediction = pd.DataFrame(columns = columns)
        for d in range(15):
            day = 20180302 + d
            dwell = sample_result_mod_7[(274 + d) % 7]['dwell'] * 0.5 + total_result_mod_7[(274 + d) % 7]['dwell'] * 0.1 + sample_result_mod_30[(274 + d) % 30]['dwell'] * 0.4
            flow_in = sample_result_mod_7[(274 + d) % 7]['flow_in'] * 0.5 + total_result_mod_7[(274 + d) % 7]['flow_in'] * 0.1 + sample_result_mod_30[(274 + d) % 30]['flow_in'] * 0.4
            flow_out = sample_result_mod_7[(274 + d) % 7]['flow_out'] * 0.5 + total_result_mod_7[(274 + d) % 7]['flow_out'] * 0.1 + sample_result_mod_30[(274 + d) % 30]['flow_out'] * 0.4
            flow_sample_prediction.loc[d] = {columns[0]:day,
                                            columns[1]:city,
                                            columns[2]:district,
                                            columns[3]:dwell,
                                            columns[4]:flow_in,
                                            columns[5]:flow_out}

        result_for_each_sample.append(flow_sample_prediction)

    result = pd.concat(result_for_each_sample)
    result.to_csv('./result/prediction.csv', index=False, header=None)