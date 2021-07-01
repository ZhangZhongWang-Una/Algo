import pandas as pd
import json
import os
import numpy as np
USER_ACTION = '../../data/v1/origin/user_action.csv'
FEED_INFO = '../../data/v1/origin/feed_info.csv'
TEST_A = '../../data/v1/origin/test_a.csv'
ROOT_PATH = '../../data/v4/'
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward"]
# user_action = pd.read_csv(USER_ACTION)
# feed_info = pd.read_csv(FEED_INFO)
# manual_keyword_list = feed_info['manual_keyword_list']
# machine_tag_list = feed_info['machine_tag_list']

# statis = json.loads(json.load(open(os.path.join(ROOT_PATH, 'statis.json'))))
# tmp = int(statis['description_map_len'])
# print(1)
# feed_emb = pd.read_csv(os.path.join(ROOT_PATH, 'feed_embeddings.csv'))
# maxfeed = feed_emb['feedid'].max() + 1
# feedids = list(feed_emb['feedid'])
# feed_emb = feed_emb.set_index('feedid')
# embeddings = np.random.randn(1, 512)
#
# for i in range(1, maxfeed):
#     if i in feedids:
#         tmp = np.array(feed_emb.loc[i].values[0].split(' ')[:-1], dtype=np.float64).reshape([1, 512])
#         embeddings = np.concatenate((embeddings, tmp), axis=0)
#     else:
#         embeddings = np.concatenate((embeddings, np.random.randn(1, 512)), axis=0)
# print(1)
# embeddings

# df = pd.read_csv(r'../../data/v4/feed_embeddings.csv', delimiter=',| ', skiprows=[0], header=None, engine='python')
# data = df.values
# X = np.array(data[:, 1:], dtype=float)
# tensor_a=tf.convert_to_tensor(a)

# user_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "userid_feature.csv"))
# tmp = user_date_feature.columns.values
# print(1)
# import time
# t = time.time()
# tmp = pd.read_csv('../../data/v2/train.csv')
# tmp1 = tmp.head(3)
# tmp2 = tmp.columns.values
# print('{}'.format(time.time() - t))
# print(2)

# print(1)
# import time
# t = time.time()
# train = pd.read_csv('../../data/v2/train.csv')
# print('{}'.format(time.time() - t))
# print(2)
#
#
# TARGET = ["read_comment", "like", "click_avatar", "forward"]
# SPARE_FEATURES = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'device']
# dense_cols = [f for f in train.columns if f not in ['date_', 'is_finish', 'play_times', 'play', 'stay'] + SPARE_FEATURES + TARGET]

# ['click_avatar' 'comment' 'date_' 'device' 'favorite' 'feedid' 'follow', 'forward' 'like' 'play' 'read_comment' 'stay' 'userid' 'authorid', 'videoplayseconds' 'is_finish' 'play_times' 'userid_5day_count', 'userid_5day_finish_rate' 'userid_5day_play_times_max', 'userid_5day_play_times_mean' 'userid_5day_play_max', 'userid_5day_play_mean' 'userid_5day_stay_max' 'userid_5day_stay_mean', 'userid_5day_read_comment_sum' 'userid_5day_read_comment_mean', 'userid_5day_like_sum' 'userid_5day_like_mean', 'userid_5day_click_avatar_sum' 'userid_5day_click_avatar_mean', 'userid_5day_forward_sum' 'userid_5day_forward_mean' 'feedid_5day_count', 'feedid_5day_finish_rate' 'feedid_5day_play_times_max', 'feedid_5day_play_times_mean' 'feedid_5day_play_max', 'feedid_5day_play_mean' 'feedid_5day_stay_max' 'feedid_5day_stay_mean', 'feedid_5day_read_comment_sum' 'feedid_5day_read_comment_mean', 'feedid_5day_like_sum' 'feedid_5day_like_mean', 'feedid_5day_click_avatar_sum' 'feedid_5day_click_avatar_mean', 'feed...
# userid staynansum staynanstd stayamin stayamax staymedian staynanmean playnansum playnanstd playamin playamax playmedian playnanmean read_commentnansum read_commentnanstd read_commentamin read_commentamax read_commentmedian read_commentnanmean likenansum l
import pandas as pd

# SUM_FEATURES = ['staynansum', 'staynanstd', 'stayamin', 'stayamax', 'staymedian', 'staynanmean',
#                  'playnansum', 'playnanstd', 'playamin', 'playamax', 'playmedian', 'playnanmean',
#                  'read_commentnansum', 'read_commentnanstd', 'read_commentamin', 'read_commentamax', 'read_commentmedian', 'read_commentnanmean',
#                  'likenansum', 'likenanstd', 'likeamin', 'likeamax', 'likemedian', 'likenanmean',
#                  'click_avatarnansum', 'click_avatarnanstd', 'click_avataramin', 'click_avataramax', 'click_avatarmedian', 'click_avatarnanmean',
#                  'forwardnansum', 'forwardnanstd', 'forwardamin', 'forwardamax', 'forwardmedian', 'forwardnanmean',
#
#                  'staynansum_user', 'staynanstd_user', 'stayamin_user', 'stayamax_user', 'staymedian_user', 'staynanmean_user',
#                  'playnansum_user', 'playnanstd_user', 'playamin_user', 'playamax_user', 'playmedian_user', 'playnanmean_user',
#                  'read_commentnansum_user', 'read_commentnanstd_user', 'read_commentamin_user', 'read_commentamax_user', 'read_commentmedian_user', 'read_commentnanmean_user',
#                  'likenansum_user', 'likenanstd_user', 'likeamin_user', 'likeamax_user', 'likemedian_user', 'likenanmean_user',
#                  'click_avatarnansum_user', 'click_avatarnanstd_user', 'click_avataramin_user', 'click_avataramax_user', 'click_avatarmedian_user', 'click_avatarnanmean_user',
#                  'forwardnansum_user', 'forwardnanstd_user', 'forwardamin_user', 'forwardamax_user', 'forwardmedian_user', 'forwardnanmean_user']

# from sklearn import decomposition
#
# feed_embeddings = np.load(os.path.join(ROOT_PATH, 'feed_embeddings.npy'))
#
# pca = decomposition.PCA(n_components=None,  #指定降到那多少维，0 < n_components < 1 或者 mle
#                         svd_solver="auto", #矩阵分解方法，{‘auto’, ‘full’, ‘arpack’, ‘randomized’}, default=’auto’
#                         whiten=False) #归一化处理，大数据下太耗时，默认 False
# Z = pca.fit_transform(feed_embeddings) #数据格式 (n_components，n_features)
# print(Z.shape)
# print(Z)
# print(pca.explained_variance_ratio_) #各个特征的贡献率，从大到小
#
# Ratios = pca.explained_variance_ratio_
# RatioSum = 0
# theta = 0.9
# dim = 0
# for i in range(len(Ratios)):
#     RatioSum +=Ratios[i]
#     dim += 1
#     if RatioSum >= theta: break
# print(dim)
# print(sum(pca.explained_variance_ratio_[0:dim]) )
# Z = Z[:, :dim]
# print(Z.shape)
# print(Z)


test1 = pd.read_csv('./submit/submit_1624887693.csv')
test2 = pd.read_csv('./submit/lgb_1624887693.csv')

import numpy as np

data1 = np.loadtxt(r"./submit/submit_1624887693.csv", comments='#', delimiter=",", skiprows=1,)
data2 = np.loadtxt(r"./submit/lgb_1624887693.csv", comments='#', delimiter=",", skiprows=1,)

assert (data1[:, 0] == data2[:, 0]).all()
assert (data1[:, 1] == data2[:, 1]).all()

data3 = (0.6*data1[:, 2:] + 0.4*data2[:, 2:])
data4 = np.concatenate((data1[:, :2], data3), axis=1)
data5 = pd.DataFrame(data4)
data5.to_csv('./submit/submit_2.csv', index=False, header=['userid',"feedid","read_comment","like","click_avatar",'forward'])
# print((data1[:, 2:] + data2[:, 2:]) / 2)
print(1)
# userid,feedid,read_comment,like,click_avatar,forward