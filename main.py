import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

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
    transition_train = pd.read_csv('../data/transition_train.csv')

    #We'll start by playing the flow, then the transition
    ###statistic with mod7, week
    flow_sample = pd.read_csv('../data/flow_train_sample.csv')
    print(flow_sample.head())
    # flow_sample.info()

    ##sample
    #group by mod7
    sample_result_mod_7 = stat_mod_n(7, flow_sample)
    # print(result_sample_by_mod7)

    # #group by mod30, month
    # sample_result_mod_30 = stat_mod_n(30, flow_sample)

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
        dwell = sample_result_mod_7[(274 + d) % 7]['dwell'] * 0.9 + total_result_mod_7[(274 + d) % 7]['dwell'] * 0.1
        flow_in = sample_result_mod_7[(274 + d) % 7]['flow_in'] * 0.9 + total_result_mod_7[(274 + d) % 7]['flow_in'] * 0.1
        flow_out = sample_result_mod_7[(274 + d) % 7]['flow_out'] * 0.9 + total_result_mod_7[(274 + d) % 7]['flow_out'] * 0.1
        flow_sample_prediction.loc[d] = {columns[0]:str(day),
                                         columns[1]:'06d86ef037e4bd311b94467c3320ff38',
                                         columns[2]:'85792b2278de59316d1158f6a97537ec',
                                         columns[3]:dwell,
                                         columns[4]:flow_in,
                                         columns[5]:flow_out}

    print(flow_sample_prediction)