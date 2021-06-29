import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import time
pd.set_option('display.max_columns', None)

import argparse
parser = argparse.ArgumentParser(description='Model args')
parser.add_argument('--learning_rate', default=0.05, type=float)
parser.add_argument('--num_leaves', default=63, type=int)
parser.add_argument('--subsample', default=0.8, type=float)
parser.add_argument('--colsample_bytree', default=0.8, type=float)


args = parser.parse_args()


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


y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']


if __name__ == '__main__':
    print('\033[32;1m[DATA]\033[0m start load data, please wait')
    T = time.time()
    train = pd.read_csv('../../data/v2/train.csv')
    test = pd.read_csv('../../data/v2/test.csv')
    cols = [f for f in train.columns if f not in ['date_', 'is_finish', 'play_times', 'play', 'stay']+ y_list]
    print('\033[32;1m[DATA]\033[0m load data done, total cost {}'.format(
        time.strftime("%H:%M:%S", time.gmtime(float(time.time() - T)))))
    print('\033[32;1m[DATA]\033[0m data.shape {}\n'.format(train[cols].shape))
    trn_x = train[train['date_'] < 14].reset_index(drop=True)
    val_x = train[train['date_'] == 14].reset_index(drop=True)
    ##################### 线下验证 #####################
    uauc_list = []
    r_list = []
    for y in y_list[:4]:
        print('=========', y, '=========')
        t = time.time()
        clf = LGBMClassifier(
            learning_rate=args.learning_rate,
            n_estimators=5000,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=2021,
            metric='None'
        )
        clf.fit(
            trn_x[cols], trn_x[y],
            eval_set=[(val_x[cols], val_x[y])],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=50
        )
        val_x[y + '_score'] = clf.predict_proba(val_x[cols])[:, 1]
        val_uauc = uAUC(val_x[y], val_x[y + '_score'], val_x['userid'])
        uauc_list.append(val_uauc)
        print(val_uauc)
        r_list.append(clf.best_iteration_)
        print('runtime: {}\n'.format(time.time() - t))

    weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]
    print(uauc_list)
    print(weighted_uauc)
    ##################### 全量训练 #####################
    r_dict = dict(zip(y_list[:4], r_list))
    for y in y_list[:4]:
        print('=========', y, '=========')
        t = time.time()
        clf = LGBMClassifier(
            learning_rate=args.learning_rate,
            n_estimators=r_dict[y],
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=2021
        )
        clf.fit(
            train[cols], train[y],
            eval_set=[(train[cols], train[y])],
            early_stopping_rounds=r_dict[y],
            verbose=100
        )
        test[y] = clf.predict_proba(test[cols])[:, 1]
        print('runtime: {}\n'.format(time.time() - t))

    file_name = "./submit/tree_%.6f_%.6f_%.6f_%.6f_%.6f.csv" % (
        weighted_uauc, uauc_list[0], uauc_list[1], uauc_list[2], uauc_list[3])
    test[['userid', 'feedid'] + y_list[:4]].to_csv(file_name, index=False)











