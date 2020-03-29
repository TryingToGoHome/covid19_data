
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
from covid19_data import new_confirmed, city_info, time

ROOT = '/local/shanxiu/ubert/covid/'
ARCHIVED_COVID_DATA = os.path.join(ROOT, 'archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv')
COUNTRY_DATA = os.path.join(ROOT, 'country_covid.csv')
DAY_OF_OBSERVATION = 15


if __name__ == "__main__":

    entire_country = pd.read_csv(COUNTRY_DATA)['Country'].values.tolist()

    train_set, test_set = train_test_split(entire_country, test_size=0.1, random_state=42)
    train_set, dev_set = train_test_split(train_set, test_size=0.1, random_state=42)

    list = [train_set, dev_set, test_set]
    name = ["train", "dev", "test"]

    date_list = time.split(",")[DAY_OF_OBSERVATION:]
    input_to_id = {}
    id_to_output = {}

    for index, split in enumerate(list):
        for i, curr_location in enumerate(split):
            for j, date in enumerate(date_list):
                loc_date = curr_location + '!' + date
                input_to_id[loc_date] = len(input_to_id)


                city = city_info("", curr_location)
                cumulative_array = new_confirmed("", "", curr_location, date)
                id_to_output[len(input_to_id)] = (city, cumulative_array)

        outfile = os.path.join(ROOT, (name[index]))
        json.dump(input_to_id, open(os.path.join(outfile, "input_to_id.json"), "w"))
        json.dump(id_to_output, open(os.path.join(outfile, "id_to_output.json"), "w"))
