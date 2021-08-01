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
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from ydzx.util.utils import load_data, time, evaluate, auc, np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 2, 'epochs')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_integer('entity_dim', 16, 'id embeddings dim')
flags.DEFINE_integer('emb_dim', 16, 'sparse embeddings dim')
flags.DEFINE_integer('conv_dim', 64, 'conv layer dim')
flags.DEFINE_integer('dnn1', 128, 'dnn_hidden_units in layer 1')
flags.DEFINE_integer('dnn2', 128, 'dnn_hidden_units in layer 2')
flags.DEFINE_integer('mem_size', 8, 'memory layer mem size')
flags.DEFINE_integer('val_dim', 8, 'memory layer mem size')
flags.DEFINE_integer('val_len', 20, 'memory layer mem size')
flags.DEFINE_integer('expert_num', 4, 'MMOE expert num')
flags.DEFINE_integer('expert_dim', 16, 'MMOE expert dim')
flags.DEFINE_integer('den_dim', 64, 'MMOE expert dim')


flags.DEFINE_integer('model', 4, 'which model to select')
flags.DEFINE_integer('seed', 2021, 'seed')
flags.DEFINE_float('dropout', 0.0, 'dnn_dropout')
flags.DEFINE_float('l2', 0.00, 'l2 reg')
flags.DEFINE_float('lr', 0.001, 'learning_rate')
flags.DEFINE_string('cuda', '2', 'CUDA_VISIBLE_DEVICES')
flags.DEFINE_string('root_path', '../data/ydzx/v1/', 'data dir')
flags.DEFINE_boolean('submit', True, 'Submit or not')

# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.cuda)
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if 4 == FLAGS.model:
    from ydzx.model.model_pnn_1 import Model
elif 1 == FLAGS.model:
    from ydzx.model.model_pnn_con import Model
elif 2 == FLAGS.model:
    from ydzx.model.model_pnn import Model
else:
    raise Exception('Unknown model:', FLAGS.model)

SPARSE_FEATURES = []
DENSE_FEATURES = []
VAL_FEATURES = []
USER_FEATURES = ['userId', 'device', 'os', 'province', 'city', 'age', 'sex', 'network']
DOC_FEATURES = ['docId', 'c1', 'photoNum', 'c2', 'refreshNum']


def interface():
    print('\033[32;1m' + '=' * 86 + '\033[0m \n')
    print('\033[32;1mSelect sparse features \033[0m ')
    print('-' * 86)
    print('0. all             1. none        2. user_os           3. user_province     4. user_city')
    print('5. user_age        6. user_sex    7. doc_c1            8. train_network     9. user_device')
    orderA = input('please enter the num of sparse features to choice it:')
    if '0' == orderA:
        SPARSE_FEATURES.extend(['device', 'os', 'province', 'city', 'age', 'sex', 'c1', 'network'])
    elif '1' == orderA:
        una = 857
    else:
        order = orderA.split(',')
        sparse_dic = {'9': 'device', '2': 'os', '3': 'province', '4': 'city', '5': 'age', '6': 'sex', '7': 'c1', '8': 'network'}
        for o in order:
            SPARSE_FEATURES.append(sparse_dic[o])

    print('\033[32;1mSelect dense features \033[0m ')
    print('-' * 86)
    print('0. all             1. none        2. photoNum          3. doc_c2            4. refreshNum')

    orderB = input('please enter the num of dense features to choice it:')
    if '0' == orderB:
        DENSE_FEATURES.extend(['photoNum', 'c2', 'refreshNum'])
    elif '1' == orderB:
        una = 857
    else:
        order = orderB.split(',')
        dense_dic = {'2': 'photoNum', '3': 'c2', '4': 'showTime'}
        for o in order:
            DENSE_FEATURES.append(dense_dic[o])

    print('\033[32;1mSelect keyword features \033[0m ')
    print('-' * 86)
    print('0. yes             1. no')
    orderC = input('please enter the num to choice it:')
    if '0' == orderC:
        VAL_FEATURES.append('keyword')

    print('\033[32;1mSelect title features \033[0m ')
    print('-' * 86)
    print('0. yes             1. no')
    orderD = input('please enter the num to choice it:')
    if '0' == orderD:
        VAL_FEATURES.append('title')


def create_input_data():
    print('\033[32;1m[DATA]\033[0m start load data, please wait')
    T = time.time()

    # 加载数据
    train, val, test = load_data(root_path=FLAGS.root_path, keyword_length=FLAGS.val_len, val_features=VAL_FEATURES)
    print('\033[32;1m[DATA]\033[0m load data done, total cost {}'.format(time.strftime("%H:%M:%S", time.gmtime(float(time.time() - T)))))
    print('\033[32;1m[DATA]\033[0m train.shape {}, val.shape {}, test.shape {}, \n'.format(train.shape, val.shape, test.shape))

    print('\033[32;1m[DATA]\033[0m start create input data, please wait')
    T = time.time()
    para = np.load(os.path.join(FLAGS.root_path, 'para.npy'), allow_pickle=True).item()

    # 构造特征嵌入
    user_feature_columns = []
    userId_feature_columns = [SparseFeat('userId', vocabulary_size=para['userId'], embedding_dim=FLAGS.entity_dim)]
    for feat in SPARSE_FEATURES:
        if feat in USER_FEATURES:
            user_feature_columns.append(SparseFeat(feat, vocabulary_size=para[feat], embedding_dim=FLAGS.emb_dim))

    doc_feature_columns = []
    docId_feature_columns = [SparseFeat('docId', vocabulary_size=para['docId'], embedding_dim=FLAGS.entity_dim)]
    for feat in SPARSE_FEATURES + DENSE_FEATURES + VAL_FEATURES:
        if feat in DOC_FEATURES:
            if 'c1' == feat:
                doc_feature_columns.append(SparseFeat('c1', vocabulary_size=para['c1'], embedding_dim=FLAGS.emb_dim))
            elif feat in DENSE_FEATURES:
                doc_feature_columns.append(DenseFeat(feat, 1))
            else:
                doc_feature_columns.append(VarLenSparseFeat(SparseFeat(feat, vocabulary_size=para[feat], embedding_dim=FLAGS.val_dim), maxlen=FLAGS.val_len))
    feature_columns = [user_feature_columns, userId_feature_columns, doc_feature_columns, docId_feature_columns]
    # 为模型生成输入数据
    feature_names = ['userId', 'docId'] + SPARSE_FEATURES + DENSE_FEATURES + VAL_FEATURES
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    for feat in VAL_FEATURES:
        train_model_input[feat] = np.array(list(train[feat].values))
        val_model_input[feat] = np.array(list(val[feat].values))
        test_model_input[feat] = np.array(list(test[feat].values))

    train_labels = train['isClick'].values
    val_labels = val['isClick'].values

    print('\033[32;1m[DATA]\033[0m data create done, total cost {} \n'.format(time.strftime("%H:%M:%S", time.gmtime(float(time.time() - T)))))

    return train_model_input, val_model_input, test_model_input, train_labels, val_labels, test, feature_columns


if __name__ == "__main__":
    interface()
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    embedding_dim = FLAGS.emb_dim
    num_users = 1538384
    num_items = 633391

    train_model_input, val_model_input, test_model_input, train_labels, val_labels, test, feature_columns = create_input_data()

    # 定义模型并训练
    train_model = Model(feature_columns, args=FLAGS, num_users=num_users, num_items=num_items)
    train_model.compile(optimizer="adagrad", loss='binary_crossentropy', metrics=[auc])
    highest_auc = 0.0
    for epoch in range(epochs):
        print('\033[32;1m[EPOCH]\033[0m {}/{}'.format(epoch + 1, epochs))
        _ = train_model.fit(train_model_input, train_labels, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)

        _, val_auc = train_model.evaluate(val_model_input, val_labels, batch_size=batch_size * 4, verbose=1)
        if val_auc > highest_auc:
            highest_auc = val_auc
    print('\033[32;1m[EVAL]\033[0m 验证集最高AUC: \033[31;4m{}\033[0m '.format(highest_auc))

    # 生成预测文件
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    timestamp = str(int(time.time()))

    # 生成提交文件
    if FLAGS.submit:
        test['isClick'] = pred_ans
        file_name = "./submit/submit_" + timestamp + ".csv"
        test[['id', 'isClick']].to_csv(file_name, index=None, float_format='%.6f')
        print('\033[32;1m[FILE]\033[0m save to {}'.format(file_name))
    # train_model.save('./ckpt/{}_{}.h5'.format(highest_auc, timestamp))
