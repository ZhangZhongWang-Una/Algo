import os
import sys

from keras_preprocessing.sequence import pad_sequences

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
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from own.v3.utils import evaluate_deepctr, pd, time, print_end, json, np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 2, 'epochs')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_integer('emb_dim', 16, 'embeddings dim')
flags.DEFINE_integer('expert_dim', 8, 'MMOE expert dim')
flags.DEFINE_integer('dnn1', 128, 'dnn_hidden_units in layer 1')
flags.DEFINE_integer('dnn2', 128, 'dnn_hidden_units in layer 2')
flags.DEFINE_integer('conv_dim', 64, 'conv layer dim')
flags.DEFINE_integer('expert_num', 4, 'MMOE expert num')
flags.DEFINE_integer('mem_size', 8, 'memory layer mem size')
flags.DEFINE_integer('day', 1, 'train dataset day select from ? to 14')
flags.DEFINE_integer('model', 4, 'which model to select')
flags.DEFINE_integer('var_len', 10, 'length of var len features ')
flags.DEFINE_integer('var_emb_dim', 16, 'emb dim of var len features ')
flags.DEFINE_integer('seed', 2021, 'seed')
flags.DEFINE_float('dropout', 0.0, 'dnn_dropout')
flags.DEFINE_float('l2', 0.00, 'l2 reg')
flags.DEFINE_float('lr', 0.001, 'learning_rate')
flags.DEFINE_string('cuda', '0', 'CUDA_VISIBLE_DEVICES')
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
    from own.v3.model.model_att import Model
elif 3 == FLAGS.model:
    from own.v3.model.model_mem import Model
elif 4 == FLAGS.model:
    from own.v3.model.model_conv import Model
else:
    raise Exception('Unknown model:', FLAGS.model)

TARGET = ["read_comment", "like", "click_avatar", "forward"]
SPARE_FEATURES = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'device']
DENSE_FEATURES = ['videoplayseconds', 'read_commentsum', 'read_commentsum_user', 'likesum', 'likesum_user',
                  'click_avatarsum', 'click_avatarsum_user', 'forwardsum', 'forwardsum_user']
# VAR_FEATURES = ['description_map', 'ocr_map', 'asr_map', 'description_char_map', 'ocr_char_map', 'asr_char_map',
#                 'manual_keyword_list_map', 'machine_keyword_list_map', 'manual_tag_list_map', 'machine_tag_list_map']
VAR_FEATURES = []


def interface():
    print('\033[32;1m' + '=' * 86 + '\033[0m')
    print('\033[32;1mSelect varlen features \033[0m ')
    print('-' * 86)
    print('1. description     2. ocr               3. asr                4.ocr_char      5.description_char')
    print('6. asr_char        7. manual_keyword    8. machine_keyword    9.manual_tag    10.machine_tag')
    print('11. all            12.none')
    order = input('please enter the num of the features to choice it:')
    print('\n')
    if '11' == order:
        VAR_FEATURES.extend(['description_map', 'ocr_map', 'asr_map', 'description_char_map', 'ocr_char_map', 'asr_char_map',
                'manual_keyword_list_map', 'machine_keyword_list_map', 'manual_tag_list_map', 'machine_tag_list_map'])
    elif '12' == order:
        VAR_FEATURES.append([])
    else:
        order = order.split(',')
        varlen_dic = {'1': 'description_map', '2': 'ocr_map', '3': 'asr_map', '4': 'ocr_char_map',
                      '5': 'description_char_map', '6': 'asr_char_map', '7': 'manual_keyword_list_map',
                      '8': 'machine_keyword_list_map', '9': 'manual_tag_list_map', '10': 'machine_tag_list_map'
                      }
        for f in order:
            VAR_FEATURES.append(varlen_dic[f])


def load_data(flags):
    feed_info = pd.read_csv(os.path.join(flags.root_path, 'feed_info_map.csv'))
    feed_info = feed_info.set_index('feedid')

    for feat in VAR_FEATURES:
        genres_list = list()
        for row in tqdm(feed_info[feat], desc=feat, total=len(feed_info[feat]), leave=True, unit='row'):
            genres_list.append(list(map(int, row.split(' '))))
        feed_info[feat] = list(pad_sequences(genres_list, maxlen=FLAGS.var_len, padding='post', ))

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

    date_feature_col = [b + "sum" for b in TARGET] + [b + "sum_user" for b in TARGET] + ['authorid', 'bgm_song_id', 'bgm_singer_id']
    data[date_feature_col] = data[date_feature_col].fillna(0)
    data[date_feature_col] = data[date_feature_col].astype(int)
    test[date_feature_col] = test[date_feature_col].fillna(0)
    test[date_feature_col] = test[date_feature_col].astype(int)

    data[DENSE_FEATURES] = np.log(data[DENSE_FEATURES] + 1.0)
    test[DENSE_FEATURES] = np.log(test[DENSE_FEATURES] + 1.0)

    train = data[data['date_'] < 14]
    val = data[data['date_'] == 14]
    return train, val, test


if __name__ == "__main__":
    interface()
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    statis = json.loads(json.load(open(os.path.join(FLAGS.root_path, 'statis.json'))))

    print('\033[32;1m[DATA]\033[0m start load data, please wait')
    T = time.time()
    train, val, test = load_data(FLAGS)
    data = pd.concat([train, val])
    if FLAGS.copy:
        train = data.copy()

    feature_names = SPARE_FEATURES + DENSE_FEATURES
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=FLAGS.emb_dim) for
                              feat in SPARE_FEATURES]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in DENSE_FEATURES]
    var_feature_columns = [
        VarLenSparseFeat(SparseFeat(feat, vocabulary_size=int(statis[feat + '_len']), embedding_dim=FLAGS.var_emb_dim),
                         maxlen=FLAGS.var_len) for feat in VAR_FEATURES]
    # 为模型生成输入数据
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test[name] for name in feature_names}

    for feat in VAR_FEATURES:
        train_model_input[feat] = np.array(list(train[feat].values))
        val_model_input[feat] = np.array(list(val[feat].values))
        test_model_input[feat] = np.array(list(test[feat].values))

    train_labels = [train[y].values for y in TARGET]
    val_labels = [val[y].values for y in TARGET]

    print('\033[32;1m[DATA]\033[0m load data done, total cost {}'.format(time.strftime("%H:%M:%S", time.gmtime(float(time.time() - T)))))
    print('\033[32;1m[DATA]\033[0m data.shape {}, train.shape {}, val.shape {}, test.shape {}, \n'.format(data.shape, train.shape, val.shape, test.shape))

    input_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns
    # 定义模型并训练
    train_model = Model(input_feature_columns, num_tasks=4, tasks=['binary', 'binary', 'binary', 'binary'], flags=FLAGS)
    train_model.compile("adagrad", loss='binary_crossentropy')
    highest_auc = 0.0
    for epoch in range(epochs):
        print('\033[32;1m[EPOCH]\033[0m {}/{}'.format(epoch+1, epochs))
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1)

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        val_auc = evaluate_deepctr(val_labels, val_pred_ans, userid_list, TARGET)
        if val_auc > highest_auc:
            highest_auc = val_auc

    print('\033[32;1m[EVAL]\033[0m 验证集最高AUC: \033[31;4m{}\033[0m '.format(highest_auc))
    t1 = time.time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    t2 = time.time()
    print('\033[32;1m[Time]\033[0m 4个目标行为{}条样本预测耗时（毫秒）：{:.3f}'.format(len(test), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
    print('\033[32;1m[Time]\033[0m 4个目标行为2000条样本平均预测耗时（毫秒）：{:.3f}'.format(ts))

    # 生成提交文件
    if FLAGS.submit:
        for i, action in enumerate(TARGET):
            test[action] = pred_ans[i]
        file_name = "./submit/submit_" + str(int(time.time())) + ".csv"
        test[['userid', 'feedid'] + TARGET].to_csv(file_name, index=None, float_format='%.6f')
        print('\033[32;1m[FILE]\033[0m save to {}'.format(file_name))

    print_end(FLAGS)
