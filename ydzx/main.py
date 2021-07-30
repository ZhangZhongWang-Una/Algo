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
from ydzx.util.utils import load_data, time, evaluate, auc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 2, 'epochs')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_integer('emb_dim', 16, 'embeddings dim')
flags.DEFINE_integer('expert_dim', 8, 'MMOE expert dim')
flags.DEFINE_integer('conv_dim', 64, 'conv layer dim')
flags.DEFINE_integer('dnn1', 128, 'dnn_hidden_units in layer 1')
flags.DEFINE_integer('dnn2', 128, 'dnn_hidden_units in layer 2')
flags.DEFINE_integer('expert_num', 4, 'MMOE expert num')
flags.DEFINE_integer('mem_size', 8, 'memory layer mem size')
flags.DEFINE_integer('day', 1, 'train dataset day select from ? to 14')
flags.DEFINE_integer('model', 1, 'which model to select')
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

if 1 == FLAGS.model:
    from ydzx.model.model_base import Model
else:
    raise Exception('Unknown model:', FLAGS.model)


def create_input_data():
    print('\033[32;1m[DATA]\033[0m start load data, please wait')
    T = time.time()
    train, val, test = load_data(root_path=FLAGS.root_path, keyword_length=10)
    print('\033[32;1m[DATA]\033[0m load data done, total cost {}'.format(time.strftime("%H:%M:%S", time.gmtime(float(time.time() - T)))))
    print('\033[32;1m[DATA]\033[0m train.shape {}, val.shape {}, test.shape {}, \n'.format(train.shape, val.shape, test.shape))

    # user2Id = np.load(os.path.join(FLAGS.root_path, 'user2id.npy'), allow_pickle=True).item()
    # doc2Id = np.load(os.path.join(FLAGS.root_path, 'doc2id.npy'), allow_pickle=True).item()
    # num_users = len(user2Id)
    # num_items = len(doc2Id)

    print('\033[32;1m[DATA]\033[0m start create input data, please wait')
    T = time.time()
    train_model_input = {}
    train_model_input['input_user_id'] = train['userId']
    train_model_input['input_item_id'] = train['docId']
    train_labels = train['isClick'].values

    val_model_input = {}
    val_model_input['input_user_id'] = val['userId']
    val_model_input['input_item_id'] = val['docId']
    val_labels = val['isClick'].values

    test_model_input = {}
    test_model_input['input_user_id'] = test['userId']
    test_model_input['input_item_id'] = test['docId']
    test_labels = test['isClick'].values
    user_id_list = test['userId'].astype(str).tolist()
    print('\033[32;1m[DATA]\033[0m data create done, total cost {} \n'.format(time.strftime("%H:%M:%S", time.gmtime(float(time.time() - T)))))

    return train_model_input, val_model_input, test_model_input, train_labels, val_labels, test_labels, user_id_list


if __name__ == "__main__":
    print('\033[32;1m' + '=' * 86 + '\033[0m \n')
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    embedding_dim = FLAGS.emb_dim
    sparse_features = ['userId', 'docId']
    dense_features = []
    num_users = 1538384
    num_items = 633391

    train_model_input, val_model_input, test_model_input, train_labels, val_labels, test_labels, user_id_list = create_input_data()

    # 定义模型并训练
    train_model = Model(args=FLAGS, num_users=num_users, num_items=num_items)
    train_model.compile(optimizer="adagrad", loss='binary_crossentropy', metrics=[auc])
    highest_auc = 0.0
    for epoch in range(epochs):
        print('\033[32;1m[EPOCH]\033[0m {}/{}'.format(epoch + 1, epochs))
        _ = train_model.fit(train_model_input, train_labels, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)

        _, val_auc = train_model.evaluate(val_model_input, val_labels, batch_size=batch_size * 4, verbose=1)
        pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 4, verbose=1)
        test_auc = evaluate(test_labels, pred_ans, user_id_list)
        print(
            '\033[32;1m[Evaluates]\033[0m AUC:\033[31;4m{:.4f}\033[0m \n'.format(test_auc))
        if val_auc > highest_auc:
            highest_auc = val_auc
    print('\033[32;1m[EVAL]\033[0m 验证集最高AUC: \033[31;4m{}\033[0m '.format(highest_auc))

    # 生成预测文件
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)