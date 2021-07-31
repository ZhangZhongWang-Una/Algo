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
USER_INFO = os.path.join(ROOT_PATH, "origin/user_info_origin.txt")
DOC_INFO = os.path.join(ROOT_PATH, "origin/doc_info.txt")
TRAIN_DATA = os.path.join(ROOT_PATH, "origin/train_data.txt")
TEST_DATA = os.path.join(ROOT_PATH, "origin/test_data.txt")


def process_user_info():
    print('Start process user_info')
    user_info = pd.read_csv(USER_INFO, sep='\t', header=None, names=['userId', 'device', 'os', 'province', 'city', 'age', 'sex'])
    user_info = user_info.fillna('nan')

    # process userId
    user2Id = np.load(os.path.join(ROOT_PATH, 'v1/user2id.npy'), allow_pickle=True).item()
    new_userId = []
    for row in tqdm(user_info['userId'], desc='userId', total=len(user_info['userId']), leave=True, unit='row'):
        new_userId.append(user2Id[row])
    user_info['userId'] = new_userId

    # process device
    device_set = set(user_info['device'].tolist())
    device_set.remove('nan')
    device2id = {'nan': 0}
    device2id.update(dict((id, i + 1) for (i, id) in enumerate(device_set)))
    new_device = []
    for row in tqdm(user_info['device'], desc='device', total=len(user_info['device']), leave=True, unit='row'):
        new_device.append(device2id[row])
    user_info['device'] = new_device

    # process os
    os_set = set(user_info['os'].tolist())
    os_set.remove('nan')
    os2id = {'nan': 0}
    os2id.update(dict((id, i + 1) for (i, id) in enumerate(os_set)))
    new_os = []
    for row in tqdm(user_info['os'], desc='os', total=len(user_info['os']), leave=True, unit='row'):
        new_os.append(os2id[row])
    user_info['os'] = new_os

    # process province
    province_set = set(user_info['province'].tolist())
    province_set.remove('nan')
    province2id = {'nan': 0}
    province2id.update(dict((id, i + 1) for (i, id) in enumerate(province_set)))
    new_province = []
    for row in tqdm(user_info['province'], desc='province', total=len(user_info['province']), leave=True, unit='row'):
        new_province.append(province2id[row])
    user_info['province'] = new_province

    # process city
    city_set = set(user_info['city'].tolist())
    city_set.remove('nan')
    city2id = {'nan': 0}
    city2id.update(dict((id, i + 1) for (i, id) in enumerate(city_set)))
    new_city = []
    for row in tqdm(user_info['city'], desc='city', total=len(user_info['city']), leave=True, unit='row'):
        new_city.append(city2id[row])
    user_info['city'] = new_province

    # process age
    new_age = []
    age = [1, 2, 3, 4, 0]  # 1=A_0_24, 2=A_25_29, 3=A_30_39, 4=A_40+, 0=nan
    age1 = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    for row in tqdm(user_info['age'], desc='age', total=len(user_info['age']), leave=True, unit='row'):
        if 'nan' == row:
            new_age.append(0)
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
    sex = [1, 2, 0]  # 1=female, 2=male, 0=nan
    for row in tqdm(user_info['sex'], desc='sex', total=len(user_info['sex']), leave=True, unit='row'):
        if 'nan' == row:
            new_sex.append(0)
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

    # process docId
    doc2Id = np.load(os.path.join(ROOT_PATH, 'v1/doc2id.npy'), allow_pickle=True).item()
    new_docId = []
    for row in tqdm(doc_info['docId'], desc='docId', total=len(doc_info['docId']), leave=True, unit='row'):
        new_docId.append(doc2Id[row])
    doc_info['docId'] = new_docId

    # process c2
    new_c2 = []
    for row in tqdm(doc_info['c2'], desc='c2_set', total=len(doc_info['c2']), leave=True, unit='row'):
        if 'nan' == row:
            new_c2.append(row)
            continue
        new_c2.append(row.split('/')[1])
    c2_set = set(new_c2)
    c2_set.remove('nan')
    c22id = {'nan': 0}
    c22id.update(dict((id, i + 1) for (i, id) in enumerate(c2_set)))
    new_c2_2 = []
    for row in tqdm(new_c2, desc='c2_map', total=len(new_c2), leave=True, unit='row'):
        new_c2_2.append(c22id[row])
    doc_info['c2'] = new_c2_2

    # process c1
    c1_set = set(doc_info['c1'].tolist())
    c1_set.remove('nan')
    c12id = {'nan': 0}
    c12id.update(dict((id, i + 1) for (i, id) in enumerate(c1_set)))
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
    keyword_set.remove('nan')

    keyword2id = {'nan': 0}
    keyword2id.update(dict((id, i + 1) for (i, id) in enumerate(keyword_set)))
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
    data_path = os.path.join(ROOT_PATH, 'v1/train_data.csv')
    train_data = pd.read_csv(data_path)
    # train_data = pd.read_csv(data_path, sep='\t', header=None,
    #                          names=['userId', 'docId', 'showTime', 'network', 'refreshNum', 'loc', 'isClick','consumeTime'])
    num_samples = len(train_data)
    samples_random = np.random.choice(num_samples, size=int(0.005 * num_samples), replace=False)
    samples_idx = np.zeros(num_samples, dtype=bool)
    samples_idx[samples_random] = True
    samples = train_data[samples_idx]
    samples.to_csv(os.path.join(ROOT_PATH, 'v1/train_data_sample.csv'), index=False)
    print('Sample create success \n')


def id2map():
    print('Start id to map')
    user_info = pd.read_csv(USER_INFO, sep='\t', header=None, names=['userId', 'device', 'os', 'province', 'city', 'age', 'sex'])
    doc_info = pd.read_csv(DOC_INFO, sep='\t', header=None, names=['docId', 'title', 'pushTime', 'photoNum', 'c1', 'c2', 'keyword'])
    train_data = pd.read_csv(TRAIN_DATA, sep='\t', header=None, names=['userId', 'docId', 'showTime', 'network', 'refreshNum', 'loc', 'isClick', 'consumeTime'])
    test_data = pd.read_csv(TEST_DATA, sep='\t', header=None, names=['id', 'userId', 'docId', 'showTime', 'network', 'refreshNum'])
    user_list = list(train_data['userId'].unique()) + list(test_data['userId'].unique()) + list(user_info['userId'].unique())
    user_set = set(user_list)
    doc_list = list(train_data['docId'].unique()) + list(test_data['docId'].unique()) + list(doc_info['docId'].unique())
    doc_set = set(doc_list)
    user2id = dict((id, i) for (i, id) in enumerate(user_set))
    doc2id = dict((id, i) for (i, id) in enumerate(doc_set))
    np.save(os.path.join(ROOT_PATH, 'v1/user2id.npy'), user2id)
    np.save(os.path.join(ROOT_PATH, 'v1/doc2id.npy'), doc2id)

    # process test userId
    new_userId = []
    for row in tqdm(test_data['userId'], desc='test_userId', total=len(test_data['userId']), leave=True, unit='row'):
        new_userId.append(user2id[row])
    test_data['userId'] = new_userId

    # process test docId
    new_docId = []
    for row in tqdm(test_data['docId'], desc='test_docId', total=len(test_data['docId']), leave=True, unit='row'):
        new_docId.append(doc2id[row])
    test_data['docId'] = new_docId

    # process train userId
    new_userId = []
    for row in tqdm(train_data['userId'], desc='train_userId', total=len(train_data['userId']), leave=True, unit='row'):
        new_userId.append(user2id[row])
    train_data['userId'] = new_userId

    # process train docId
    new_docId = []
    for row in tqdm(train_data['docId'], desc='train_docId', total=len(train_data['docId']), leave=True, unit='row'):
        new_docId.append(doc2id[row])
    train_data['docId'] = new_docId

    train_data.to_csv(os.path.join(ROOT_PATH, 'v1/train_data.csv'), index=False)
    test_data.to_csv(os.path.join(ROOT_PATH, 'v1/test_data.csv'), index=False)
    print('Map success \n')


def csv2hdf5():
    print('Train data csv to hdf5')
    train_data = pd.read_csv(os.path.join(ROOT_PATH, 'v1/train_data.csv'), dtype=int)
    train_data.to_hdf(os.path.join(ROOT_PATH, 'v1/train_data.h5'), key='traindata')
    print('Change success \n')


if __name__ == '__main__':
    # process_user_info()
    # process_doc_info()
    # process_test_sample()
    # id2map()
    csv2hdf5()
    # print(1)