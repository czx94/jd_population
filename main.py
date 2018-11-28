import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
################################################data analysis############################################
    #read data
    flow_train = pd.read_csv('../data/flow_train.csv')
    transition_train = pd.read_csv('../data/transition_train.csv')

    # flow_train.info()
    # transition_train.info()

    # #show info
    # print(flow_train.head())
    # print(transition_train.head())

    # #7 cities in total
    # print(flow_train['city_code'].value_counts())
    # print(transition_train['o_city_code'].value_counts())

    # #98 districts in total
    # print(flow_train['district_code'].value_counts())
    # print(transition_train['o_district_code'].value_counts())

    # # #274 days in total
    # print(flow_train['date_dt'].value_count())
    # print(flow_train['date_dt'].value_count())
    #
    # #from 20170601 to 20180301
    # print(flow_train['date_dt'].min())
    # print(flow_train['date_dt'].max())
    # print(transition_train['date_dt'].min())
    # print(transition_train['date_dt'].max())

    # #visualize group by day
    # dwell_by_day = flow_train.groupby('date_dt')['dwell'].sum()
    # # flow in == flow out cause the total is invariable
    # flow_in_by_day = flow_train.groupby('date_dt')['flow_in'].sum()
    # flow_out_by_day = flow_train.groupby('date_dt')['flow_out'].sum()
    # flow_by_day = pd.merge(dwell_by_day.to_frame(), flow_in_by_day.to_frame(), how='left', on='date_dt')
    # flow_by_day = pd.merge(flow_by_day, flow_out_by_day.to_frame(), how='right', on='date_dt')

    # print(flow_by_day)
    # flow_by_day_plot = flow_by_day.reset_index(drop=True)
    # flow_by_day_plot.plot()
    #
    # plt.show()


###########################################################data processing##########################################################
    #construct city-district name group
    cities = flow_train.drop_duplicates(['city_code','district_code'], keep='first').groupby('city_code')['district_code']
    city_district_group = {}
    for name, group in cities:
        city_district_group[name] = group.tolist()

    # # verification
    # for city, districts in city_district_group.items():
    #     print(city, len(districts))

    #construct city-district data group
    flow_train_city_district = {}
    transition_train_city_district = {}
    for city, districts in tqdm(city_district_group.items()):
        flow_dict = {}
        transition_dict = {}
        flow_train_city = flow_train[flow_train['city_code'] == city]
        transition_train_city = transition_train[transition_train['o_city_code'] == city]
        for district in tqdm(districts):
            flow_train_district = flow_train_city[flow_train_city['district_code'] == district]
            # flow_dict[district] = flow_train_district.reset_index(drop=True)

            transition_train_district = transition_train_city[transition_train_city['o_district_code'] == district]
            # transition_dict[district] = transition_train_district.reset_index(drop=True)

            #generate csv for each district
            flow_train_district.to_csv('../data/flow/' + city + '_' + district + '.csv', index=False)
            transition_train_district.to_csv('../data/transition/' + city + '_' + district + '.csv', index=False)

        # flow_train_city_district[city] = flow_dict
        # transition_train_city_district[city] = transition_dict

    # # #verifying
    # flow_sample = flow_train_city_district['06d86ef037e4bd311b94467c3320ff38']['85792b2278de59316d1158f6a97537ec']
    # transition_sample = transition_train_city_district['06d86ef037e4bd311b94467c3320ff38']['85792b2278de59316d1158f6a97537ec']
    # # print(flow_sample)
    #
    # flow_sample.to_csv('../data/flow_train_sample.csv', index=False)
    # transition_sample.to_csv('../data/transition_train_sample.csv', index=False)



#####################################################################modeling#############################################################
    # #We'll start by playing the flow, then the transition
    # ###statistic with mod7, week
    # flow_sample = pd.read_csv('../data/flow_train_sample.csv')
    # print(flow_sample.head())
    # # flow_sample.info()
    #
    # ##sample
    # #group by mod7
    # flow_sample_group_by_mod7 = flow_sample.groupby(flow_sample.index % 7)
    # result_sample_by_mod7 = {}
    # for mod, v in flow_sample_group_by_mod7:
    #     result_sample_by_mod7[mod] = v.drop(['city_code', 'district_code', 'date_dt'], axis=1).mean().to_dict()
    # # print(result_sample_by_mod7)
    #
    # # #group by mod30, month
    # # flow_sample_group_by_mod30 = flow_sample.groupby(flow_sample.index % 30)
    # # for mod, v in flow_sample_group_by_mod30:
    # #     print(mod, v.drop(['city_code', 'district_code', 'date_dt'], axis=1).mean())
    #
    #
    # ##total
    # #group by mod7
    # flow_group_by_mod7 = flow_train.groupby(flow_train.index % 7)
    # result_by_mod7 = {}
    # for mod, v in flow_group_by_mod7:
    #     result_by_mod7[mod] = v.drop(['city_code', 'district_code', 'date_dt'], axis=1).mean().to_dict()
    #
    # # #group by mod30, month
    # # flow_group_by_mod30 = flow_train.groupby(flow_train.index % 30)
    # # for mod, v in flow_group_by_mod30:
    # #     print(mod, v.drop(['city_code', 'district_code', 'date_dt'], axis=1).mean())
    #
    # # #prediction of the coming 15 days based on mod7 stat
    # columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
    # flow_sample_prediction = pd.DataFrame(columns = columns)
    # for d in range(15):
    #     day = 20180302 + d
    #     dwell = result_sample_by_mod7[(274 + d) % 7]['dwell'] * 0.9 + result_by_mod7[(274 + d) % 7]['dwell'] * 0.1
    #     flow_in = result_sample_by_mod7[(274 + d) % 7]['flow_in'] * 0.9 + result_by_mod7[(274 + d) % 7]['flow_in'] * 0.1
    #     flow_out = result_sample_by_mod7[(274 + d) % 7]['flow_out'] * 0.9 + result_by_mod7[(274 + d) % 7]['flow_out'] * 0.1
    #     flow_sample_prediction.loc[d] = {columns[0]:str(day),
    #                                      columns[1]:'06d86ef037e4bd311b94467c3320ff38',
    #                                      columns[2]:'85792b2278de59316d1158f6a97537ec',
    #                                      columns[3]:dwell,
    #                                      columns[4]:flow_in,
    #                                      columns[5]:flow_out}
    #
    # print(flow_sample_prediction)
