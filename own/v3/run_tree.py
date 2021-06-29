import os
import sys
from collections import defaultdict

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

path1 = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path1)
path2 = os.path.split(path1)[0]
sys.path.append(path2)
path3 = os.path.split(path2)[0]
sys.path.append(path3)
import warnings
import logging
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')
import tensorflow as tf
from tqdm import tqdm
from sklearn import decomposition
from keras_preprocessing.sequence import pad_sequences
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from own.v3.utils import evaluate_deepctr, pd, time, print_end, json, np
from own.v3.model.modules import MatrixInit


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 2, 'epochs')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_integer('emb_dim', 32, 'embeddings dim')
flags.DEFINE_integer('expert_dim', 16, 'MMOE expert dim')
flags.DEFINE_integer('dnn1', 128, 'dnn_hidden_units in layer 1')
flags.DEFINE_integer('dnn2', 128, 'dnn_hidden_units in layer 2')
flags.DEFINE_integer('conv_dim', 64, 'conv layer dim')
flags.DEFINE_integer('expert_num', 4, 'MMOE expert num')
flags.DEFINE_integer('mem_size', 8, 'memory layer mem size')
flags.DEFINE_integer('day', 1, 'train dataset day select from ? to 14')
flags.DEFINE_integer('model', 4, 'which model to select')
flags.DEFINE_integer('word_fea_len', 32, 'length of word features ')
flags.DEFINE_integer('word_fea_dim', 32, 'emb dim of word features ')
flags.DEFINE_integer('tag_fea_len', 16, 'length of tag features ')
flags.DEFINE_integer('tag_fea_dim', 32, 'emb dim of tag features ')
flags.DEFINE_integer('seed', 2021, 'seed')
flags.DEFINE_float('dropout', 0.0, 'dnn_dropout')
flags.DEFINE_float('l2', 0.00, 'l2 reg')
flags.DEFINE_float('lr', 0.001, 'learning_rate')
flags.DEFINE_string('cuda', '1', 'CUDA_VISIBLE_DEVICES')
flags.DEFINE_string('root_path', '../../data/v4/', 'data dir')
flags.DEFINE_boolean('submit', True, 'Submit or not')
flags.DEFINE_boolean('copy', False, 'Concat train and val or not')
# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.cuda)
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if 1 == FLAGS.model:
    from own.v3.model.model_clear import Model
elif 2 == FLAGS.model:
    from own.v3.model.model_dcn import Model
elif 3 == FLAGS.model:
    from own.v3.model.model_mem import Model
elif 4 == FLAGS.model:
    from own.v3.model.model_conv import Model
elif 5 == FLAGS.model:
    from own.v3.model.model_trm import Model
elif 6 == FLAGS.model:
    from own.v3.model.model_ple import Model
else:
    raise Exception('Unknown model:', FLAGS.model)

TARGET = ["read_comment", "like", "click_avatar", "forward"]
SPARE_FEATURES = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'device']
DENSE_FEATURES = ['videoplayseconds']
SUM_FEATURES = ['read_commentsum', 'read_commentsum_user', 'likesum', 'likesum_user',
                  'click_avatarsum', 'click_avatarsum_user', 'forwardsum', 'forwardsum_user']
WORD_FEATURES = []
TAG_FEATURES = []


def interface():
    print('\033[32;1m' + '=' * 86 + '\033[0m')
    print('\033[32;1mSelect word features \033[0m ')
    print('-' * 86)
    print('1. description         2. ocr                 3. asr')
    print('4. description_char    5. ocr_char            6. asr_char')
    print('7. all                 8.none')
    # orderA = input('please enter the num of word features to choice it:')
    orderA = '8'
    if '7' == orderA:
        WORD_FEATURES.extend(['description_map', 'ocr_map', 'asr_map', 'description_char_map', 'ocr_char_map', 'asr_char_map'])
    elif '8' == orderA:
        una = 857
    else:
        order = orderA.split(',')
        word_dic = {'1': 'description_map', '2': 'ocr_map', '3': 'asr_map', '4': 'description_char_map',
                      '5': 'ocr_char_map', '6': 'asr_char_map'}
        for f in order:
            WORD_FEATURES.append(word_dic[f])

    print('\033[32;1mSelect tag features \033[0m ')
    print('-' * 86)
    print('1. manual_keyword      2. machine_keyword')
    print('3. manual_tag          4. machine_tag')
    print('5. all                 6.none')
    # orderB = input('please enter the num of tag features to choice it:')
    orderB = '6'
    print('\n')
    if '5' == orderB:
        TAG_FEATURES.extend(['manual_keyword_list_map', 'machine_keyword_list_map', 'manual_tag_list_map', 'machine_tag_list_map'])
    elif '6' == orderB:
        una = 857
    else:
        order = orderB.split(',')
        tag_dic = {'1': 'manual_keyword_list_map', '2': 'machine_keyword_list_map', '3': 'manual_tag_list_map', '4': 'machine_tag_list_map'}
        for f in order:
            TAG_FEATURES.append(tag_dic[f])


def load_data(flags):
    feed_info = pd.read_csv(os.path.join(flags.root_path, 'feed_info_map.csv'))
    feed_info = feed_info.set_index('feedid')

    for feat in WORD_FEATURES:
        genres_list = list()
        for row in tqdm(feed_info[feat], desc=feat, total=len(feed_info[feat]), leave=True, unit='row'):
            genres_list.append(list(map(int, row.split(' '))))
        feed_info[feat] = list(pad_sequences(genres_list, maxlen=flags.word_fea_len, padding='post', ))

    for feat in TAG_FEATURES:
        genres_list = list()
        for row in tqdm(feed_info[feat], desc=feat, total=len(feed_info[feat]), leave=True, unit='row'):
            genres_list.append(list(map(int, row.split(' '))))
        feed_info[feat] = list(pad_sequences(genres_list, maxlen=flags.tag_fea_len, padding='post', ))

    data = pd.read_csv(os.path.join(flags.root_path, 'user_action.csv'))
    data = data[data['date_'] >= flags.day]
    test = pd.read_csv(os.path.join(flags.root_path, 'test_a.csv'))
    user_date_feature = pd.read_csv(os.path.join(flags.root_path, "userid_feature.csv"))
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    feed_date_feature = pd.read_csv(os.path.join(flags.root_path, "feedid_feature.csv"))
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])

    # join action sum
    data = data.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
    data = data.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")
    test["date_"] = 15
    test = test.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
    test = test.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")
    test = test.drop(columns=['date_'])

    # join feed info
    data = data.join(feed_info, on="feedid", how="left", rsuffix="_feed")
    test = test.join(feed_info, on="feedid", how="left", rsuffix="_feed")

    data[SPARE_FEATURES] = data[SPARE_FEATURES].fillna(0)
    data[SPARE_FEATURES] = data[SPARE_FEATURES].astype(int)
    test[SPARE_FEATURES] = test[SPARE_FEATURES].fillna(0)
    test[SPARE_FEATURES] = test[SPARE_FEATURES].astype(int)

    data[DENSE_FEATURES] = data[DENSE_FEATURES].fillna(0.)
    data[DENSE_FEATURES] = data[DENSE_FEATURES].astype(float)
    test[DENSE_FEATURES] = test[DENSE_FEATURES].fillna(0.)
    test[DENSE_FEATURES] = test[DENSE_FEATURES].astype(float)
    data[DENSE_FEATURES] = np.log(data[DENSE_FEATURES] + 1.0)
    test[DENSE_FEATURES] = np.log(test[DENSE_FEATURES] + 1.0)

    train = data[data['date_'] < 14]
    val = data[data['date_'] == 14]

    return train, val, test


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


if __name__ == "__main__":
    interface()
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    DENSE_FEATURES.extend(SUM_FEATURES)
    statis = json.loads(json.load(open(os.path.join(FLAGS.root_path, 'statis.json'))))

    print('\033[32;1m[DATA]\033[0m start load data, please wait')
    T = time.time()
    train, val, test = load_data(FLAGS)
    data = pd.concat([train, val])

    feature_names = SPARE_FEATURES + DENSE_FEATURES

    train_labels = [train[y].values for y in TARGET]
    val_labels = [val[y].values for y in TARGET]

    print('\033[32;1m[DATA]\033[0m load data done, total cost {}'.format(time.strftime("%H:%M:%S", time.gmtime(float(time.time() - T)))))
    print('\033[32;1m[DATA]\033[0m data.shape {}, train.shape {}, val.shape {}, test.shape {}, \n'.format(data.shape, train.shape, val.shape, test.shape))
    #
    # input_feature_columns = sparse_feature_columns + dense_feature_columns + word_feature_columns + tag_feature_columns
    uauc_list = []
    r_list = []
    # tree
    for y in TARGET:
        print('=========', y, '=========')
        t = time.time()
        clf = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=5000,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=2021,
            metric='None'
        )
        clf.fit(
            train[feature_names], train[y],
            eval_set=[(val[feature_names], val[y])],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=50
        )

        val[y + '_score'] = clf.predict_proba(val[feature_names])[:, 1]
        userid_list = val['userid'].astype(str).tolist()
        val_uauc = uAUC(val[y].tolist(), val[y + '_score'].tolist(), userid_list)
        uauc_list.append(val_uauc)
        print(val_uauc)
        r_list.append(clf.best_iteration_)
        print('runtime: {}\n'.format(time.time() - t))

    weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]
    print(uauc_list)
    print(weighted_uauc)
    ##################### 全量训练 #####################
    r_dict = dict(zip(TARGET, r_list))
    for y in TARGET:
        print('=========', y, '=========')
        t = time.time()
        clf = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=r_dict[y],
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=2021
        )
        clf.fit(
            data[feature_names], data[y],
            eval_set=[(data[feature_names], data[y])],
            early_stopping_rounds=r_dict[y],
            verbose=100
        )

        test[y] = clf.predict_proba(test[feature_names])[:, 1]

        print('runtime: {}\n'.format(time.time() - t))

    test[['userid', 'feedid'] + TARGET[:4]].to_csv(
        'sub_%.6f_%.6f_%.6f_%.6f_%.6f.csv' % (weighted_uauc, uauc_list[0], uauc_list[1], uauc_list[2], uauc_list[3]),index=False)

    print_end(FLAGS)
