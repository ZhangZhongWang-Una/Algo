# coding: utf-8
import os
import re
import math
import time
from tqdm import tqdm
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__file__)
import numpy as np
import pandas as pd

ROOT_PATH = "../../data/ydzx/"
USER_INFO = os.path.join(ROOT_PATH, "origin/user_info.txt")
DOC_INFO = os.path.join(ROOT_PATH, "origin/doc_info.txt")
TRAIN_DATA = os.path.join(ROOT_PATH, "origin/train_data.txt")
TEST_DATA = os.path.join(ROOT_PATH, "origin/test_data.txt")


def process_user_info():
    print('Start process user_info')
    user_info = pd.read_csv(USER_INFO, sep='\t', header=None, names=['userId', 'device', 'os', 'province', 'city', 'age', 'sex'])
    user_info = user_info.fillna('nan')

    # process device
    device_set = set(user_info['device'].tolist())
    device2id = dict((id, i) for (i, id) in enumerate(device_set))
    new_device = []
    for row in tqdm(user_info['device'], desc='device', total=len(user_info['device']), leave=True, unit='row'):
        new_device.append(device2id[row])
    user_info['device'] = new_device

    # process os
    os_set = set(user_info['os'].tolist())
    os2id = dict((id, i) for (i, id) in enumerate(os_set))
    new_os = []
    for row in tqdm(user_info['os'], desc='os', total=len(user_info['os']), leave=True, unit='row'):
        new_os.append(os2id[row])
    user_info['os'] = new_os

    # process province
    province_set = set(user_info['province'].tolist())
    province2id = dict((id, i) for (i, id) in enumerate(province_set))
    new_province = []
    for row in tqdm(user_info['province'], desc='province', total=len(user_info['province']), leave=True, unit='row'):
        new_province.append(province2id[row])
    user_info['province'] = new_province

    # process city
    city_set = set(user_info['city'].tolist())
    city2id = dict((id, i) for (i, id) in enumerate(city_set))
    new_city = []
    for row in tqdm(user_info['city'], desc='city', total=len(user_info['city']), leave=True, unit='row'):
        new_city.append(city2id[row])
    user_info['city'] = new_province

    # process age
    new_age = []
    age = [0, 1, 2, 3, 4]  # 0=A_0_24, 1=A_25_29, 2=A_30_39, 3=A_40+, 4=nan
    age1 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    for row in tqdm(user_info['age'], desc='age', total=len(user_info['age']), leave=True, unit='row'):
        if 'nan' == row:
            new_age.append(4)
            continue
        elif len(row) > 100:
            pro = list(map(float, re.split(r'[:,]', row)[1::2]))
            new_age.append(age1[np.nanargmax(pro)])
            continue
        # age = re.split(r'[:,]', row)[::2]
        pro = list(map(float, re.split(r'[:,]', row)[1::2]))
        new_age.append(age[np.nanargmax(pro)])
    user_info['age'] = new_age

    # process sex
    new_sex = []
    sex = [0, 1, 2]  # 0=female, 1=male, 2=nan
    for row in tqdm(user_info['sex'], desc='sex', total=len(user_info['sex']), leave=True, unit='row'):
        if 'nan' == row:
            new_sex.append(2)
            continue
        # sex = re.split(r'[:,]', row)[::2]
        pro = list(map(float, re.split(r'[:,]', row)[1::2]))
        new_sex.append(sex[np.nanargmax(pro)])
    user_info['sex'] = new_sex

    user_info.to_csv(os.path.join(ROOT_PATH, 'v1/user_info.csv'), index=False)
    print('User_info process success \n')


def process_doc_info():
    print('Start process doc_info')
    doc_info = pd.read_csv(DOC_INFO, sep='\t', header=None, names=['docId', 'title', 'pushTime', 'photoNum', 'c1', 'c2', 'keyword'])
    doc_info = doc_info.fillna('nan')[['docId', 'pushTime', 'photoNum', 'c1', 'c2', 'keyword']]

    # process c2
    new_c2 = []
    for row in tqdm(doc_info['c2'], desc='c2_set', total=len(doc_info['c2']), leave=True, unit='row'):
        if 'nan' == row:
            new_c2.append(row)
            continue
        new_c2.append(row.split('/')[1])
    c2_set = set(new_c2)
    c22id = dict((id, i) for (i, id) in enumerate(c2_set))
    new_c2_2 = []
    for row in tqdm(new_c2, desc='c2_map', total=len(new_c2), leave=True, unit='row'):
        new_c2_2.append(c22id[row])
    doc_info['c2'] = new_c2_2

    # process c1
    c1_set = set(doc_info['c1'].tolist())
    c12id = dict((id, i) for (i, id) in enumerate(c1_set))
    new_c1 = []
    for row in tqdm(doc_info['c1'], desc='c1', total=len(doc_info['c1']), leave=True, unit='row'):
        new_c1.append(c12id[row])
    doc_info['c1'] = new_c1

    # process keyword
    keyword_set = ['nan']
    for row in tqdm(doc_info['keyword'], desc='keyword_set', total=len(doc_info['keyword']), leave=True, unit='row'):
        if 'nan' == row:
            continue
        keyword_set.extend(re.split(r'[:,]', row)[::2])
    keyword_set = set(keyword_set)
    keyword2id = dict((id, i) for (i, id) in enumerate(keyword_set))
    new_keyword = []
    for row in tqdm(doc_info['keyword'], desc='keyword_map', total=len(doc_info['keyword']), leave=True, unit='row'):
        if 'nan' == row:
            new_keyword.append(keyword2id[row])
            continue
        keyword_list = re.split(r'[:,]', row)[::2]
        tmp1 = list((keyword2id[keyword] for keyword in keyword_list))
        new_keyword.append(tmp1)
    doc_info['keyword'] = new_keyword

    doc_info.to_csv(os.path.join(ROOT_PATH, 'v1/doc_info.csv'), index=False)
    print('Doc_info process success \n')


def process_test_sample():
    print('Start create test sample')
    train_data = pd.read_csv(TRAIN_DATA, sep='\t', header=None,
                             names=['userId', 'docId', 'showTime', 'network', 'refreshNum', 'loc', 'isClick','consumeTime'])
    num_samples = len(train_data)
    samples_random = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
    samples_idx = np.zeros(num_samples, dtype=bool)
    samples_idx[samples_random] = True
    samples = train_data[samples_idx]
    samples.to_csv(os.path.join(ROOT_PATH, 'v1/train_data_sample.csv'), index=False)
    print('Sample create success \n')


if __name__ == '__main__':
    # process_user_info()
    # process_doc_info()
    process_test_sample()
    # doc_info = pd.read_csv(DOC_INFO, sep='\t', header=None, names=['docId', 'title', 'pushTime', 'photoNum', 'c1', 'c2', 'keyword'])
    # train_data = pd.read_csv(TRAIN_DATA, sep='\t', header=None, names=['userId', 'docId', 'showTime', 'network', 'refreshNum', 'loc', 'isClick', 'consumeTime'])
    # test_data = pd.read_csv(TEST_DATA, sep='\t', header=None, names=['id', 'userId', 'docId', 'showTime', 'network', 'refreshNum'])
    # print(1)