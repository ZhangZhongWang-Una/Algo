import os
import sys
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
    orderA = input('please enter the num of word features to choice it:')
    # orderA = '8'
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
    orderB = input('please enter the num of tag features to choice it:')
    # orderB = '2,3,4'
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

    # feed_embeddings = np.load(os.path.join(FLAGS.root_path, 'feed_embeddings.npy'))
    # pca = decomposition.PCA(n_components=flags.emb_dim,  # 指定降到那多少维，0 < n_components < 1 或者 mle
    #                         svd_solver="auto",  # 矩阵分解方法，{‘auto’, ‘full’, ‘arpack’, ‘randomized’}, default=’auto’
    #                         whiten=False)
    # feed_embeddings_t = pca.fit_transform(feed_embeddings)
    # return train, val, test, feed_embeddings_t
    return train, val, test


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
    if FLAGS.copy:
        train = data.copy()

    feature_names = SPARE_FEATURES + DENSE_FEATURES
    # SPARE_FEATURES.remove('feedid')
    # sparse_feature_columns = [SparseFeat('feedid', vocabulary_size=data['feedid'].max() + 1, embedding_dim=FLAGS.emb_dim, embeddings_initializer=MatrixInit(feed_embeddings))]\
    #                          + [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=FLAGS.emb_dim) for feat in SPARE_FEATURES]
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=FLAGS.emb_dim) for feat in SPARE_FEATURES]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in DENSE_FEATURES]
    word_feature_columns = [
        VarLenSparseFeat(SparseFeat(feat, vocabulary_size=int(statis[feat + '_len']), embedding_dim=FLAGS.word_fea_dim),
                         maxlen=FLAGS.word_fea_len) for feat in WORD_FEATURES]
    tag_feature_columns = [
        VarLenSparseFeat(SparseFeat(feat, vocabulary_size=int(statis[feat + '_len']), embedding_dim=FLAGS.tag_fea_dim),
                         maxlen=FLAGS.tag_fea_len) for feat in TAG_FEATURES]
    # 为模型生成输入数据
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test[name] for name in feature_names}

    for feat in WORD_FEATURES + TAG_FEATURES:
        train_model_input[feat] = np.array(list(train[feat].values))
        val_model_input[feat] = np.array(list(val[feat].values))
        test_model_input[feat] = np.array(list(test[feat].values))

    train_labels = [train[y].values for y in TARGET]
    val_labels = [val[y].values for y in TARGET]

    print('\033[32;1m[DATA]\033[0m load data done, total cost {}'.format(time.strftime("%H:%M:%S", time.gmtime(float(time.time() - T)))))
    print('\033[32;1m[DATA]\033[0m data.shape {}, train.shape {}, val.shape {}, test.shape {}, \n'.format(data.shape, train.shape, val.shape, test.shape))

    input_feature_columns = sparse_feature_columns + dense_feature_columns + word_feature_columns + tag_feature_columns
    # 定义模型并训练
    train_model = Model(input_feature_columns, num_tasks=4, tasks=['binary', 'binary', 'binary', 'binary'], flags=FLAGS)
    train_model.compile("adagrad", loss='binary_crossentropy', loss_weights=[1.1, 1, 1, 1])
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
