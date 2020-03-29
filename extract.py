
import json
import os
import pickle
import math

import numpy as np
import pandas as pd
import torch
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from covid19_data import new_confirmed, city_info, time, target

ROOT = '/local/shanxiu/ubert/covid/'
ARCHIVED_COVID_DATA = os.path.join(ROOT, 'archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv')
COUNTRY_DATA = os.path.join(ROOT, 'country_covid.csv')
REGION_DATA = os.path.join(ROOT, 'region_covid.csv')
DAY_OF_OBSERVATION = 15


if __name__ == "__main__":

    entire_country = pd.read_csv(COUNTRY_DATA)['Country'].values.tolist()
    entire_region = pd.read_csv(REGION_DATA)['ID'].values.tolist()

    country_train, country_test = train_test_split(entire_country, test_size=0.1, random_state=42)
    country_train, country_dev = train_test_split(country_train,  test_size=0.1, random_state=42)

    region_train, region_test = train_test_split(entire_region, test_size=0.1, random_state=42)
    region_train, region_dev = train_test_split(region_train, test_size=0.1, random_state=42)

    list = [country_train, country_dev, country_test, region_train, region_dev, region_test]
    name = ["train", "dev", "test"]

    date_list = time.split(",")[DAY_OF_OBSERVATION:]
    input_to_id = {}
    id_to_output = {}
    id_to_target = {}

    for index, split in enumerate(name):
        country_data = list[index]
        county_data = list[index+3]
        for i, curr_location in enumerate(country_data):
            country_all_zero = False
            for j, date in enumerate(date_list):

                loc_date = curr_location + '!' + date
                city = city_info("Country","", curr_location)
                cumulative_array = new_confirmed("Country", "", "", curr_location, date)
                goal = target("Country", curr_location, date)

                if cumulative_array[-1] != 0 and goal > 0:
                    _id = len(input_to_id)
                    input_to_id[loc_date] = _id
                    id_to_target[_id] = str(goal)
                    id_to_output[_id] = (city, cumulative_array)
                    country_all_zero = True
                if j == len(date_list) - 1 and country_all_zero is True:
                    _id = len(input_to_id)
                    input_to_id[loc_date] = _id
                    id_to_target[_id] = "0"
                    id_to_output[_id] = (city, [0 for i in range(15)])


        for i, curr_location in enumerate(county_data):
            city_all_zero = False
            for j, date in enumerate(date_list):
                loc_date = curr_location + '!' + date

                city = city_info("Region","", curr_location)
                cumulative_array = new_confirmed("Region", "", "", curr_location, date)
                goal = target("Region", curr_location, date)

                if cumulative_array[-1] != 0 and goal > 0:
                    _id = len(input_to_id)
                    input_to_id[loc_date] = _id
                    id_to_target[_id] = str(goal)
                    id_to_output[_id] = (city, cumulative_array)
                    city_all_zero = True

                if j == len(date_list) - 1 and city_all_zero is True:
                    _id = len(input_to_id)
                    input_to_id[loc_date] = _id
                    id_to_target[_id] = "0"
                    id_to_output[_id] = (city, [0 for i in range(15)])


        outfile = os.path.join(ROOT+'trimmed_data/', (name[index]))
        json.dump(input_to_id, open(os.path.join(outfile, "input_to_id.json"), "w"))
        json.dump(id_to_output, open(os.path.join(outfile, "id_to_output.json"), "w"))
        json.dump(id_to_target, open(os.path.join(outfile, "id_to_target.json"), "w"))
