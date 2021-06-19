import os
import pandas as pd
import numpy as np

USER_ACTION = '../../data/v1/origin/user_action.csv'
FEED_INFO = '../../data/v1/origin/feed_info.csv'
TEST_A = '../../data/v1/origin/test_a.csv'
ROOT_PATH = '../../data/v3/'
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward"]


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


if __name__ == '__main__':
    statis_feature()
    process_test_data()
    process_train_data()
    print(1)