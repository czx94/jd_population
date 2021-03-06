import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

if __name__ == '__main__':
################################################data analysis############################################
    # #read data
    # flow_train = pd.read_csv('../data/flow_train.csv')
    # transition_train = pd.read_csv('../data/transition_train.csv')

    # flow_train.info()
    # transition_train.info()
    #
    # #show info
    # print(flow_train.head())
    # print(transition_train.head())
    #
    # #7 cities in total
    # print(flow_train['city_code'].value_counts())
    # print(transition_train['o_city_code'].value_counts())
    #
    # #98 districts in total
    # print(flow_train['district_code'].value_counts())
    # print(transition_train['o_district_code'].value_counts())
    #
    # # #274 days in total
    # print(flow_train['date_dt'].value_counts())
    # print(flow_train['date_dt'].value_counts())
    #
    # #from 20170601 to 20180301
    # print(flow_train['date_dt'].min())
    # print(flow_train['date_dt'].max())
    # print(transition_train['date_dt'].min())
    # print(transition_train['date_dt'].max())
    #
    # #visualize group by day
    # dwell_by_day = flow_train.groupby('date_dt')['dwell'].sum()
    # # flow in == flow out cause the total is invariable
    # flow_in_by_day = flow_train.groupby('date_dt')['flow_in'].sum()
    # flow_out_by_day = flow_train.groupby('date_dt')['flow_out'].sum()
    # flow_by_day = pd.merge(dwell_by_day.to_frame(), flow_in_by_day.to_frame(), how='left', on='date_dt')
    # flow_by_day = pd.merge(flow_by_day, flow_out_by_day.to_frame(), how='right', on='date_dt')
    #
    # print(flow_by_day)
    # flow_by_day_plot = flow_by_day.reset_index(drop=True)
    # flow_by_day_plot.plot()
    #
    # plt.show()



    #analysis for transition
    transition_train = pd.read_csv('../data/transition_train.csv')
    cnt = transition_train['cnt']
    figure = cnt.hist(range=[0,5], alpha=0.5, bins=10).get_figure()
    plt.show()
    print(cnt.describe())