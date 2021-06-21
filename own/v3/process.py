import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

USER_ACTION = '../../data/v1/origin/user_action.csv'
FEED_INFO = '../../data/v1/origin/feed_info.csv'
TEST_A = '../../data/v1/origin/test_a.csv'
ROOT_PATH = '../../data/v3/'
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward"]
SPACE_PARA = ['description', 'ocr', 'asr', 'description_char', 'ocr_char', 'asr_char']
SEMICOLON_PARA = ['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']



def del_file(path):
    '''
    删除path目录下的所有内容
    '''
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)


def process_train_data():
    # user_action信息表
    data = pd.read_csv(USER_ACTION)
    # data = data.set_index('userid')
    # feed信息表
    feed_info = pd.read_csv(FEED_INFO)[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']]
    feed_info = feed_info.set_index('feedid')
    # 基于userid统计的历史行为的次数
    user_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "{}_feature.csv".format('userid')))
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    # 基于feedid统计的历史行为的次数
    feed_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "{}_feature.csv".format('feedid')))
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])

    data = data.join(feed_info, on="feedid", how="left", rsuffix="_feed")
    data = data.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
    data = data.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")

    feed_feature_col = [b + "sum" for b in FEA_COLUMN_LIST]
    user_feature_col = [b + "sum_user" for b in FEA_COLUMN_LIST]
    data[feed_feature_col] = data[feed_feature_col].fillna(0.0)
    data[user_feature_col] = data[user_feature_col].fillna(0.0)
    data[feed_feature_col] = np.log(data[feed_feature_col] + 1.0)
    data[user_feature_col] = np.log(data[user_feature_col] + 1.0)

    data[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1
    data[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        data[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    data[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
        data[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)

    train = data[data['date_'] < 14]
    val = data[data['date_'] == 14]
    train.to_csv(os.path.join(ROOT_PATH, "train.csv"), index=False)
    val.to_csv(os.path.join(ROOT_PATH, "val.csv"), index=False)
    print('save train data done')


def process_test_data():
    # test_a信息表
    data = pd.read_csv(TEST_A)
    data["date_"] = 15
    # feed信息表
    feed_info = pd.read_csv(FEED_INFO)[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']]
    feed_info = feed_info.set_index('feedid')
    # 基于userid统计的历史行为的次数
    user_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "{}_feature.csv".format('userid')))
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    # 基于feedid统计的历史行为的次数
    feed_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "{}_feature.csv".format('feedid')))
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])

    data = data.join(feed_info, on="feedid", how="left", rsuffix="_feed")
    data = data.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
    data = data.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")

    feed_feature_col = [b + "sum" for b in FEA_COLUMN_LIST]
    user_feature_col = [b + "sum_user" for b in FEA_COLUMN_LIST]
    data[feed_feature_col] = data[feed_feature_col].fillna(0.0)
    data[user_feature_col] = data[user_feature_col].fillna(0.0)
    data[feed_feature_col] = np.log(data[feed_feature_col] + 1.0)
    data[user_feature_col] = np.log(data[user_feature_col] + 1.0)

    data[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1
    data[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        data[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    data[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
        data[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)

    data = data.drop(columns=['date_'])
    data.to_csv(os.path.join(ROOT_PATH, "test.csv"), index=False)
    print('save test data done')


def statis_feature(start_day=1, end_day=15, agg='sum'):
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    for dim in ["userid", "feedid"]:
        print('statis {} feature'.format(dim))
        user_data = history_data[[dim, "date_"] + FEA_COLUMN_LIST]
        res_arr = []
        for start in range(start_day, end_day + 1):
            temp = user_data[((user_data["date_"]) < start)]
            temp = temp.drop(columns=['date_'])
            temp = temp.groupby([dim]).agg([agg]).reset_index()
            temp.columns = list(map(''.join, temp.columns.values))
            temp["date_"] = start
            res_arr.append(temp)
        dim_feature = pd.concat(res_arr)
        feature_path = os.path.join(ROOT_PATH, "{}_feature.csv".format(dim))
        print('Save to: %s' % feature_path)
        dim_feature.to_csv(feature_path, index=False)


def process_train_data_all():
    # user_action信息表
    data = pd.read_csv(USER_ACTION)
    # feed信息表
    feed_info = pd.read_csv(FEED_INFO)
    feed_info = feed_info.set_index('feedid')
    # 基于userid统计的历史行为的次数
    user_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "{}_feature.csv".format('userid')))
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    # 基于feedid统计的历史行为的次数
    feed_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "{}_feature.csv".format('feedid')))
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])

    data = data.join(feed_info, on="feedid", how="left", rsuffix="_feed")
    data = data.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
    data = data.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")

    feed_feature_col = [b + "sum" for b in FEA_COLUMN_LIST]
    user_feature_col = [b + "sum_user" for b in FEA_COLUMN_LIST]
    data[feed_feature_col] = data[feed_feature_col].fillna(0.0)
    data[user_feature_col] = data[user_feature_col].fillna(0.0)
    data[feed_feature_col] = np.log(data[feed_feature_col] + 1.0)
    data[user_feature_col] = np.log(data[user_feature_col] + 1.0)

    data[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1
    data[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        data[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    data[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
        data[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)

    train = data[data['date_'] < 14]
    val = data[data['date_'] == 14]
    train.to_csv(os.path.join(ROOT_PATH, "train.csv"), index=False)
    val.to_csv(os.path.join(ROOT_PATH, "val.csv"), index=False)
    print('save train data done')


def process_user_action_table():
    print('process_user_action_table')
    # user_action信息表
    data = pd.read_csv(USER_ACTION)
    # 基于userid统计的历史行为的次数
    user_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "{}_feature.csv".format('userid')))
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    # 基于feedid统计的历史行为的次数
    feed_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "{}_feature.csv".format('feedid')))
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])

    data = data.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
    data = data.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")

    feed_feature_col = [b + "sum" for b in FEA_COLUMN_LIST]
    user_feature_col = [b + "sum_user" for b in FEA_COLUMN_LIST]
    data[feed_feature_col] = data[feed_feature_col].fillna(0)
    data[user_feature_col] = data[user_feature_col].fillna(0)
    data[user_feature_col] = data[user_feature_col].astype(int)

    data.to_csv(os.path.join(ROOT_PATH, "user_action.bin"), index=False)
    print('save to {} \n'.format(os.path.join(ROOT_PATH, "user_action.bin")))


def process_feed_table():
    print('process_feed_table')
    # feed信息表
    feed_info = pd.read_csv(FEED_INFO)
    for para in SPACE_PARA:
        data = feed_info[para].fillna('0')
        data_list = list()
        data_id_dict = load_data_dict(os.path.join(ROOT_PATH, para+'_id.bin'))
        for row in tqdm(data, desc=para, total=len(data), leave=True, unit='row'):
            row_list = [str(data_id_dict.get(i)) for i in row.split(' ')]
            data_list.append(' '.join(row_list))
        feed_info[para + '_map'] = data_list

    for para in SEMICOLON_PARA:
        data = feed_info[para].fillna('0')
        data_list = list()
        data_id_dict = load_data_dict(os.path.join(ROOT_PATH, para+'_id.bin'))
        for row in tqdm(data, desc=para, total=len(data), leave=True, unit='row'):
            row_list = [str(data_id_dict.get(i)) for i in row.split(';')]
            data_list.append(' '.join(row_list))
        feed_info[para + '_map'] = data_list

    feed_info[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1
    feed_info[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed_info[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    feed_info[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
        feed_info[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)

    feed_info.to_csv(os.path.join(ROOT_PATH, 'feed_info_all.bin'), index=False)
    feed_info[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id',
               'description_map', 'ocr_map', 'asr_map', 'description_char_map', 'ocr_char_map', 'asr_char_map',
               'manual_keyword_list_map', 'machine_keyword_list_map', 'manual_tag_list_map', 'machine_tag_list_map'
               ]].to_csv(os.path.join(ROOT_PATH, 'feed_info_part.bin'), index=False)

    print('save to {} \n'.format(os.path.join(ROOT_PATH, "feed_info_all.bin")))


def process_test_table():
    print('process_test_table')
    # test_a信息表
    data = pd.read_csv(TEST_A)
    data["date_"] = 15
    # feed信息表
    feed_info = pd.read_csv(FEED_INFO)[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']]
    feed_info = feed_info.set_index('feedid')
    # 基于userid统计的历史行为的次数
    user_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "{}_feature.csv".format('userid')))
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    # 基于feedid统计的历史行为的次数
    feed_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "{}_feature.csv".format('feedid')))
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])

    data = data.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
    data = data.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")

    feed_feature_col = [b + "sum" for b in FEA_COLUMN_LIST]
    user_feature_col = [b + "sum_user" for b in FEA_COLUMN_LIST]
    data[feed_feature_col] = data[feed_feature_col].fillna(0)
    data[user_feature_col] = data[user_feature_col].fillna(0)

    data = data.drop(columns=['date_'])
    data.to_csv(os.path.join(ROOT_PATH, "test_a.bin"), index=False)
    print('save to {}'.format(os.path.join(ROOT_PATH, 'test_a.bin')))


def build_data_id_table():
    print('build_data_id_table')
    # feed信息表
    feed_info = pd.read_csv(FEED_INFO)
    for para in SPACE_PARA:
        data = feed_info[para].fillna('0')
        data_set = set()
        for row in tqdm(data, desc=para, total=len(data), leave=True, unit='row'):
            row_set = set(row.split(' '))
            # row_set = set(map(int, row_set))
            data_set = data_set | row_set
        data_set = list(sorted(data_set))
        data_map = {x: i for i, x in enumerate(data_set)}
        dump_file(data_map, para+'_id')

    for para in SEMICOLON_PARA:
        data = feed_info[para].fillna('0')
        data_set = set()
        for row in tqdm(data, desc=para, total=len(data), leave=True, unit='row'):
            row_set = set(row.split(';'))
            data_set = data_set | row_set
        data_set = list(sorted(data_set))
        data_map = {x: i for i, x in enumerate(data_set)}
        dump_file(data_map, para+'_id')


def dump_file(data, name):
    with open(os.path.join(ROOT_PATH, name+'.bin'), 'w') as f:
        for key in data.keys():
            row = key + '\t' + str(data[key]) + '\n'
            f.write(row)
        f.close()
    print('save to {}'.format(os.path.join(ROOT_PATH, name+'.bin')))


def load_data_dict(path):
    wordId_dict = {}

    with open(path, "r") as f:
        line = f.readline().replace("\n", "")
        while line != None and line != "":
            arr = line.split("\t")
            wordId_dict[arr[0]] = int(arr[1])
            line = f.readline().replace("\n", "")
        f.close()

    return wordId_dict


def build_data_statis_table():
    print('build_data_statis_table')
    statis = {}

    feed_info = pd.read_csv(os.path.join(ROOT_PATH, 'feed_info_part.bin'))
    for para in SPACE_PARA + SEMICOLON_PARA:

        data_id_dict = load_data_dict(os.path.join(ROOT_PATH, para+'_id.bin'))
        statis[para+'_map_len'] = str(len(data_id_dict))

        data = feed_info[para+'_map']
        len_list = list()
        for row in tqdm(data, desc=para, total=len(data), leave=True, unit='row'):
            len_list.append(len(row.split(' ')))
        mean = int(np.nanmean(len_list))
        statis[para + '_map_avg'] = str(mean)
        u = np.array(len_list)
        x = np.sort(u)
        u_len = x[int(0.85 * len(u)) - 1]
        statis[para + '_map_85'] = str(u_len)

    j = json.dumps(statis)
    with open(os.path.join(ROOT_PATH, 'statis.json'), 'w') as f:
        json.dump(j, f)
        f.close()
    print('save to {}'.format(os.path.join(ROOT_PATH, 'statis.json')))



if __name__ == '__main__':
    # statis_feature()
    # process_test_data()
    # process_train_data()

    # build_data_id_table()
    # process_feed_table()
    # process_user_action_table()
    process_test_table()
    # build_data_statis_table()
    print(1)