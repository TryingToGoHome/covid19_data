# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
# import pandas as pd
import torch
import csv
from torch.utils.data import Dataset

# from param import args
# from src.utils import load_obj_h5py

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.

# The path to data and image features.

ARCHIVED_COVID_DATA = '/Users/nanyamashuu/PycharmProjects/COVID-19/archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv'

SPLIT2NAME = {
    'train': 'train',
    'dev': 'dev',
    'test': 'test',
}

"""
An example in obj36 tsv:
csv_column = [Province/State,Country/Region,Lat,Long,1/22/20,1/23/20,1/24/20,1/25/20,1/26/20,
1/27/20,1/28/20,1/29/20,1/30/20,1/31/20,2/1/20,2/2/20,2/3/20,2/4/20,2/5/20,2/6/20,2/7/20,2/8/20,
2/9/20,2/10/20,2/11/20,2/12/20,2/13/20,2/14/20,2/15/20,2/16/20,2/17/20,2/18/20,2/19/20,2/20/20,
2/21/20,2/22/20,2/23/20,2/24/20,2/25/20,2/26/20,2/27/20,2/28/20,2/29/20,3/1/20,3/2/20,3/3/20,
3/4/20,3/5/20,3/6/20,3/7/20,3/8/20,3/9/20,3/10/20,3/11/20,3/12/20,3/13/20,3/14/20,3/15/20,3/16/20,
3/17/20,3/18/20,3/19/20,3/20/20,3/21/20,3/22/20,3/23/20]


FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""


#county, province, country, 1/22/20
def new_confirmed(County, Province, Country, time):
    #return an array of previous 14 days
    date_list = []
    date = time.split('/') #month/day/year
    month = int(date[0])
    day = int(date[1])
    year = int(date[2])


    #assume date is valid, edit later

    if (day-14 > 0):
        for i in range(1,15,1):
            curr_day = str(day - i)
            date_string = str(month)+'/' +curr_day + '/' + str(year)
            date_list.append(date_string)
    else:
        left = 14 - day + 1
        for i in range(1, day, 1):
            curr_day = str(day - i)
            date_string = str(month)+'/' +curr_day + '/' + str(year)
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







    with open(ARCHIVED_COVID_DATA, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            print(
                f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
            line_count += 1
    #     print(f'Processed {line_count} lines.')



class Covid19Dataset(Dataset):
    def __init__(self, splits: str):
        super().__init__()

        self.splits = splits

        # Loading detection features to img_data
        self.img_data = []
        # for split in self.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
        load_topk = SPLIT2NUM[splits]
        # self.img_data.extend(load_obj_tsv(
        #     os.path.join(VISUALSRL_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[self.splits])), topk=load_topk))

        self.img_data.extend(load_obj_h5py(
            os.path.join(VISUALSRL_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[self.splits])), topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in self.img_data:
            self.imgid2img[img_datum['img_id']] = img_datum


        print("Use %d data in torch dataset" % (len(self.img_data)))
        print()

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, item: int):

        # Get image info
        img_info = self.img_data[item]
        img_name = img_info['img_id']
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        return img_name, feats, boxes


new_confirmed("", "", "", '03/10/20')