# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle
import math
import tqdm


import numpy as np
import pandas as pd
import torch
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# from param import args
# from src.utils import load_obj_h5py

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.

# The path to data and image features.


ROOT = '/local/shanxiu/ubert/covid/'
ARCHIVED_COVID_DATA = os.path.join(ROOT, 'archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv')
COUNTRY_DATA = os.path.join(ROOT, 'country_covid.csv')


SPLIT2NAME = {
    'train': 'train.json',
    'dev': 'dev.json',
    'test': 'test.json',
}

time = "1/22/20,1/23/20,1/24/20,1/25/20,1/26/20,1/27/20,1/28/20,1/29/20,1/30/20,1/31/20,2/1/20,2/2/20,2/3/20,2/4/20,2/5/20,2/6/20,2/7/20,2/8/20,2/9/20,2/10/20,2/11/20,2/12/20,2/13/20,2/14/20,2/15/20,2/16/20,2/17/20,2/18/20,2/19/20,2/20/20,2/21/20,2/22/20,2/23/20,2/24/20,2/25/20,2/26/20,2/27/20,2/28/20,2/29/20,3/1/20,3/2/20,3/3/20,3/4/20,3/5/20,3/6/20,3/7/20,3/8/20,3/9/20,3/10/20,3/11/20,3/12/20,3/13/20,3/14/20,3/15/20,3/16/20,3/17/20,3/18/20,3/19/20,3/20/20,3/21/20,3/22/20,3/23/20"


"""
time_series_column = [Province/State,Country/Region,Lat,Long,1/22/20,1/23/20,1/24/20,1/25/20,1/26/20,
1/27/20,1/28/20,1/29/20,1/30/20,1/31/20,2/1/20,2/2/20,2/3/20,2/4/20,2/5/20,2/6/20,2/7/20,2/8/20,
2/9/20,2/10/20,2/11/20,2/12/20,2/13/20,2/14/20,2/15/20,2/16/20,2/17/20,2/18/20,2/19/20,2/20/20,
2/21/20,2/22/20,2/23/20,2/24/20,2/25/20,2/26/20,2/27/20,2/28/20,2/29/20,3/1/20,3/2/20,3/3/20,
3/4/20,3/5/20,3/6/20,3/7/20,3/8/20,3/9/20,3/10/20,3/11/20,3/12/20,3/13/20,3/14/20,3/15/20,3/16/20,
3/17/20,3/18/20,3/19/20,3/20/20,3/21/20,3/22/20,3/23/20]

covid19_country_column = Country,Region,Population,Area (sq. mi.),Population Density,GDP ($per capita),Literacy (%),1/22/20,


"""

DAY_OF_OBSERVATION = 15

#county, province, country, 1/22/20
def new_confirmed(County, Province, Country, time):
    #return an array of previous days
    date_list = []
    date = time.split('/') #month/day/year
    month = int(date[0])
    day = int(date[1])
    year = int(date[2])


    #assume date is valid, edit later

    if (day- DAY_OF_OBSERVATION > 0):
        for i in range(1, DAY_OF_OBSERVATION+1, 1):
            curr_day = str(day - i)
            date_string = str(month)+'/' + curr_day + '/' + str(year)
            date_list.append(date_string)
    else:
        left = DAY_OF_OBSERVATION - day + 1
        for i in range(1, day, 1):
            curr_day = str(day - i)
            date_string = str(month)+'/' + curr_day + '/' + str(year)
            date_list.append(date_string)
        if month == 3:
            for i in range(0, left, 1):
                curr_day = str(29 - i)
                date_string = str(month-1) + '/' + curr_day + '/' + str(year)
                date_list.append(date_string)
        if month == 2:
            for i in range(0, left, 1):
                curr_day = str(31 - i)
                date_string = str(month-1) + '/' + curr_day + '/' + str(year)
                date_list.append(date_string)

    date_list.reverse()



    df = pd.read_csv(COUNTRY_DATA, index_col='Country')

    cumulative = []
    increase = []

    increase.append(int(df[date_list[0]][Country]))
    for i, date in enumerate(date_list):
        cumulative.append(float(df[date][Country]))
        if i > 0:
            increase.append(cumulative[i] - cumulative[i-1])

    # start_num, day1 increase, ..., day 14 increase
    #
    # print(torch.FloatTensor(increase))
    return increase



def city_info(Province, Country):

    df = pd.read_csv(COUNTRY_DATA, index_col='Country')

    info = []
    population = math.log(float(df["Population"][Country]), 10)
    area = math.log(float(df["Area (sq. mi.)"][Country]), 10)
    density = math.log(float(df["Population Density"][Country]), 10)
    gdp = math.log(float(df["GDP ($per capita)"][Country]), 10)
    literacy = math.log(float(df["Literacy (%)"][Country]), 10)

    info.append(population)
    info.append(area)
    info.append(density)
    info.append(gdp)
    info.append(literacy)

    # print(torch.FloatTensor(info))
    return info


class Covid19Dataset(Dataset):
    def __init__(self, splits: str):
        super().__init__()

        self.splits = splits

        # entire_country = pd.read_csv(COUNTRY_DATA)['Country'].values.tolist()
        #
        # train_set, test_set = train_test_split(entire_country, test_size = 0.1, random_state = 42)
        # train_set, dev_set = train_test_split(train_set, test_size=0.1, random_state=42)
        #
        if splits == "train":
            self.id_to_output = json.load(open(os.path.join(ROOT+"train/", "id_to_output.json")))
        if splits == "dev":
            self.id_to_output = json.load(open(os.path.join(ROOT+"dev/", "id_to_output.json")))
        if splits == "test":
            self.id_to_output = json.load(open(os.path.join(ROOT+"test/", "id_to_output.json")))

        # self.date = time.split(",")[DAY_OF_OBSERVATION:]
        #
        # # {"location, date", id}
        # # {id: (location, population, policy), (day1, day2, ..., day14)}
        # self.input_to_id = {}
        # self.id_to_output = {}
        #
        # for i, curr_location in enumerate(self.location):
        #     for j, date in enumerate(self.date):
        #         self.input_to_id[(curr_location, date)] = len(self.input_to_id)
        #         _id = self.input_to_id[(curr_location, date)]
        #
        #         city = city_info("", curr_location)
        #         cumulative_array = new_confirmed("", "", curr_location, date)
        #         self.id_to_output[_id] = (city, cumulative_array)


        print("Use %d data in dataset" % (len(self.id_to_output)))
        print()

    def __len__(self):
        return len(self.id_to_output)

    def __getitem__(self, item: int):

        # Get image info
        city, covid = self.id_to_output[str(item+1)]
        return torch.FloatTensor(city), torch.FloatTensor(covid)

# train.json, dev.json, test.json {city : id}
if __name__ == "__main__":
    # Build Class
    time_array = time.split(",")
    city_info("", "Japan")
    new_confirmed("", "Alabama", "Japan", '03/22/20')


    train = Covid19Dataset("train")
    evaluator = None
    data_loader = DataLoader(train)

    for i, value in enumerate(data_loader):
        city, covid_info = value
        with torch.no_grad():
            print("city")
            print(city)
            print("covid_info")
            print(covid_info)

