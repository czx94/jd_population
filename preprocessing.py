import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

#指数平滑公式
def exponential_smoothing(s, alpha=0.47):
    s_out = np.zeros(len(s))
#    s_result[0] = float(s[1]+s[2]+s[3])/3
    s_out[0] = s[0]
    for i in range(1, len(s_out)):
        s_out[i] = alpha*s[i]+(1-alpha)*s_out[i-1]

    for i in range(len(s)):
        s[i] = s_out[i]
    # return s_out

def generate_csv(city, districts, df, type):
    if type == 'flow':
        train_city = df[df['city_code'] == city]
    else:
        train_city = df[df['o_city_code'] == city]
    for district in tqdm(districts):
        if type == 'flow':
            train_district = train_city[train_city['district_code'] == district]
        else:
            train_district = train_city[train_city['o_district_code'] == district]
        # #generate csv for each district
        train_district.to_csv('../data/' + type + '/' + city + '_' + district + '.csv', index=False)

def train_val_split(raw_ds, val_len = 15):
    return raw_ds[:-val_len].reset_index(drop=True), raw_ds[-15:].reset_index(drop=True)

if __name__ == '__main__':
    #read data
    flow_train = pd.read_csv('../data/flow_train.csv')
    transition_train = pd.read_csv('../data/transition_train.csv')

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
            flow_dict[district] = flow_train_district.reset_index(drop=True)

            transition_train_district = transition_train_city[transition_train_city['o_district_code'] == district]
            transition_dict[district] = transition_train_district.reset_index(drop=True)

        flow_train_city_district[city] = flow_dict
        transition_train_city_district[city] = transition_dict

    # # #verifying
    # flow_sample = flow_train_city_district['06d86ef037e4bd311b94467c3320ff38']['85792b2278de59316d1158f6a97537ec']
    # transition_sample = transition_train_city_district['06d86ef037e4bd311b94467c3320ff38']['85792b2278de59316d1158f6a97537ec']
    # # print(flow_sample)
    #
    # flow_sample.to_csv('../data/flow_train_sample.csv', index=False)
    # transition_sample.to_csv('../data/transition_train_sample.csv', index=False)

    #generate csv for each district
    for city, districts in tqdm(city_district_group.items()):
        generate_csv(city, districts, flow_train, 'flow')
        generate_csv(city, districts, transition_train, 'transition')


    # #train_val_split test
    # flow_sample=pd.read_csv('../data/flow_train_sample.csv')
    # train, val = train_val_split(flow_sample)
    # train.info()
    # val.info()