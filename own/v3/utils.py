import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score

STR_PARA = ['description', 'ocr', 'asr', 'description_char', 'ocr_char', 'asr_char',
            'manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']
LOG_COL = ['read_commentsum', 'read_commentsum_user', 'likesum', 'likesum_user',
           'click_avatarsum', 'click_avatarsum_user', 'forwardsum', 'forwardsum_user']

def load_data(root_path):
    train = pd.read_csv(os.path.join(root_path, 'train.csv'))
    val = pd.read_csv(os.path.join(root_path, 'val.csv'))
    test = pd.read_csv(os.path.join(root_path, 'test.csv'))
    return train, val, test


def load_tvt_data(root_path, PARA, log=False):
    user_action = pd.read_csv(os.path.join(root_path, 'user_action.bin'))
    feed_info = pd.read_csv(os.path.join(root_path, 'feed_info_part.bin'))
    feed_info = feed_info.set_index('feedid')
    feed_info["videoplayseconds"] = np.log(feed_info["videoplayseconds"] + 1.0)
    statis = json.loads(json.load(open(os.path.join(root_path, 'statis.json'))))
    test = pd.read_csv(os.path.join(root_path, 'test_a.bin'))

    # for para in PARA:
    #     tmp = feed_info[para]
    #     tmp_list = list()
    #     for row in tmp:
    #         tmp_list.append(row.split(' '))
    #
    #     feed_info[para] = tmp_list
    data = user_action.join(feed_info, on="feedid", how="left", rsuffix="_feed")
    test = test.join(feed_info, on="feedid", how="left", rsuffix="_feed")

    if log:
        data[LOG_COL] = np.log(data[LOG_COL] + 1.0)
        test[LOG_COL] = np.log(data[LOG_COL] + 1.0)

    train = data[data['date_'] < 14]
    val = data[data['date_'] == 14]
    return train, val, test, statis


def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc


def compute_weighted_score(score_dict, weight_dict):
    '''基于多个行为的uAUC值，计算加权uAUC
    Input:
        scores_dict: 多个行为的uAUC值映射字典, dict
        weights_dict: 多个行为的权重映射字典, dict
    Output:
        score: 加权uAUC值, float
    '''
    score = 0.0
    weight_sum = 0.0
    for action in score_dict:
        weight = float(weight_dict[action])
        score += weight*score_dict[action]
        weight_sum += weight
    score /= float(weight_sum)
    score = round(score, 6)
    return score


def evaluate_deepctr(val_labels,val_pred_ans,userid_list,target):
    eval_dict = {}
    for i, action in enumerate(target):
        eval_dict[action] = uAUC(val_labels[i], val_pred_ans[i], userid_list)
    print('\033[32;1m[EVAL]\033[0m {}'.format(eval_dict))
    # weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
    #                "comment": 1, "follow": 1}
    weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "forward": 1}
    weight_auc = compute_weighted_score(eval_dict, weight_dict)
    print('\033[32;1m[uAUC]\033[0m weighted uAUC: \033[31;4m{}\033[0m \n'.format(weight_auc))
    return weight_auc


def print_end(FLAGS):
    print('\033[32;1m[PARA]\033[0m' +
          ' epochs: {}'.format(FLAGS.epochs) +
          ' batch_size: {}'.format(FLAGS.batch_size) +
          ' emb_dim: {}'.format(FLAGS.emb_dim) +
          ' expert_dim: {}'.format(FLAGS.expert_dim) +
          ' dnn1: {}'.format(FLAGS.dnn1) +
          ' dnn2: {}'.format(FLAGS.dnn2) +
          ' dropout: {}'.format(FLAGS.dropout) +
          ' lr: {}'.format(FLAGS.lr) +
          ' l2: {}'.format(FLAGS.l2) +
          ' day: {}'.format(FLAGS.day) +
          ' copy: {}'.format(FLAGS.copy) +
          ' model: {}'.format(FLAGS.model) +
          ' expert_num: {}'.format(FLAGS.expert_num) +
          ' mem_size: {}'.format(FLAGS.mem_size) +
          ' conv_dim: {}'.format(FLAGS.conv_dim) +
          # ' : {}'.format(FLAGS.) +
          ' \n')
    print('\033[32;1m' + '=' * 86 + '\033[0m')
