import os
import time
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras import backend as K
import tensorflow as tf


def load_data(root_path, keyword_length, val_features):
    data = pd.read_csv(os.path.join(root_path, 'train_data_sample.csv'))
    # data = pd.read_csv(os.path.join(root_path, 'train_data.csv'))
    # data = pd.read_csv(os.path.join(root_path, 'train_data_part.csv'))
    # data = pd.read_hdf(os.path.join(root_path, 'train_data.h5'), key='traindata', dtype=int)

    # 用户信息
    user_info = pd.read_csv(os.path.join(root_path, 'user_info.csv'))
    user_info = user_info.set_index('userId')
    data = data.join(user_info, on='userId', how='left', rsuffix='_user')

    # 文章信息
    doc_info = pd.read_csv(os.path.join(root_path, 'doc_info.csv'))
    doc_info = doc_info.set_index('docId')

    if 'keyword' in val_features:
        genres_list = []
        for row in tqdm(doc_info['keyword'], desc='keyword', total=len(doc_info['keyword']), leave=True, unit='row'):
            if '0' == row:
                genres_list.append([])
            else:
                genres_list.append(eval(row))

        doc_info['keyword'] = list(
            pad_sequences(list(genres_list), maxlen=keyword_length, padding='post', truncating='post'))

    if 'title' in val_features:
        genres_list = []
        for row in tqdm(doc_info['title'], desc='title', total=len(doc_info['title']), leave=True, unit='row'):
                genres_list.append(eval(row))

        doc_info['title'] = list(
            pad_sequences(list(genres_list), maxlen=keyword_length, padding='post', truncating='post'))

    data = data.join(doc_info, on='docId', how='left', rsuffix='_doc')
    pad = pad_sequences([[]], maxlen=keyword_length, padding='post', truncating='post')[0]
    L = sum(data["keyword"].isnull())
    data["keyword"][data["keyword"].isnull()] = [pd.Series([pad])] * L
    L = sum(data["title"].isnull())
    data["title"][data["title"].isnull()] = [pd.Series([pad])] * L

    data = data.fillna(0)
    np.random.seed(2021)
    # 划分百分之20作为验证集
    num_samples = len(data)
    valid = np.random.choice(num_samples, size=int(0.20 * num_samples), replace=False)
    valid_idx = np.zeros(num_samples, dtype=bool)
    valid_idx[valid] = True
    val = data[valid_idx]
    train = data[~valid_idx]

    test = pd.read_csv(os.path.join(root_path, 'test_data.csv'))
    test = test.join(user_info, on='userId', how='left', rsuffix='_user')
    test = test.join(doc_info, on='docId', how='left', rsuffix='_doc')

    return train, val, test



def evaluate(labels, preds, user_id_list):
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


def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


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
          ' word_fea_len: {}'.format(FLAGS.word_fea_len) +
          ' word_fea_dim: {}'.format(FLAGS.word_fea_dim) +
          ' tag_fea_len: {}'.format(FLAGS.tag_fea_len) +
          ' tag_fea_dim: {}'.format(FLAGS.tag_fea_dim) +
          # ' : {}'.format(FLAGS.) +
          ' \n')
    print('\033[32;1m' + '=' * 86 + '\033[0m')