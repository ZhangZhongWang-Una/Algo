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
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from own.v3.utils import load_data, evaluate_deepctr, pd, time
from own.v3.model.model_clear import Model


tf.get_logger().setLevel('ERROR')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 2, 'epochs')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_integer('emb_dim', 16, 'embeddings dim')
flags.DEFINE_integer('expert_dim', 8, 'MMOE expert dim')
flags.DEFINE_integer('dnn1', 128, 'dnn_hidden_units in layer 1')
flags.DEFINE_integer('dnn2', 128, 'dnn_hidden_units in layer 2')
flags.DEFINE_integer('day', 1, 'train dataset day')
flags.DEFINE_float('dropout', 0.0, 'dnn_dropout')
flags.DEFINE_float('l2', 0.00, 'l2 reg')
flags.DEFINE_float('lr', 0.001, 'learning_rate')
flags.DEFINE_string('cuda', '0', 'CUDA_VISIBLE_DEVICES')
flags.DEFINE_string('root_path', '../../data/v3/', 'data dir')
flags.DEFINE_boolean('submit', True, 'Submit or not')
flags.DEFINE_boolean('copy', False, 'Concat train and val or not')
# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.cuda)
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if __name__ == "__main__":
    print('\033[32;1m' + '=' * 108 + '\033[0m \n')
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    embedding_dim = FLAGS.emb_dim
    target = ["read_comment", "like", "click_avatar", "forward"]
    # 官方 baseline DNN特征列
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    # 官方 baseline Linear特征列
    dense_features = ['videoplayseconds', 'device']
    # 行为统计特征
    for t in target:
        dense_features.append(t + "sum")
        dense_features.append(t + "sum_user")

    print('\033[32;1m[DATA]\033[0m start load data, please wait')
    T = time.time()
    train, val, test = load_data(FLAGS.root_path)
    train = train[train['date_'] >= FLAGS.day]
    data = pd.concat([train, val])
    if FLAGS.copy:
        train = data.copy()
    print('\033[32;1m[DATA]\033[0m load data done, total cost {}'.format(time.strftime("%H:%M:%S", time.gmtime(float(time.time() - T)))))
    print('\033[32;1m[DATA]\033[0m data.shape {}, train.shape {}, val.shape {}, test.shape {}, \n'.format(data.shape, train.shape, val.shape, test.shape))

    # 计算每个稀疏字段的唯一特征，并记录密集特征字段名称
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                              for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

    # generate input data for model
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test[name] for name in feature_names}

    train_labels = [train[y].values for y in target]
    val_labels = [val[y].values for y in target]

    # Define Model,train,predict and evaluate
    train_model = Model(dnn_feature_columns, num_tasks=4, expert_dim=FLAGS.expert_dim, dnn_hidden_units=(FLAGS.dnn1, FLAGS.dnn2),
                        dnn_dropout=FLAGS.dropout, l2_reg_dnn=FLAGS.l2,
                        tasks=['binary', 'binary', 'binary', 'binary'])
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=FLAGS.lr)
    # train_model.compile(optimizer=optimizer, loss='binary_crossentropy')
    train_model.compile("adagrad", loss='binary_crossentropy')
    highest_auc = 0.0
    for epoch in range(epochs):
        print('\033[32;1m[EPOCH]\033[0m {}/{}'.format(epoch+1, epochs))
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1)

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        val_auc = evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)
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
        for i, action in enumerate(target):
            test[action] = pred_ans[i]
        file_name = "./submit/submit_" + str(int(time.time())) + ".csv"
        test[['userid', 'feedid'] + target].to_csv(file_name, index=None, float_format='%.6f')
        print('\033[32;1m[FILE]\033[0m save to {}'.format(file_name))

    print('\033[32;1m[PARA]\033[0m' +
          ' epoch:{}, batch_size:{}, emb_dim:{}, expert_dim:{}, dnn1:{}, dnn2:{}, dropout:{}, lr:{}, l2:{}, day:{}, copy:{} \n'
          .format(FLAGS.epochs, FLAGS.batch_size, FLAGS.emb_dim, FLAGS.expert_dim, FLAGS.dnn1, FLAGS.dnn2,
          FLAGS.dropout, FLAGS.lr, FLAGS.l2, FLAGS.day, FLAGS.copy))
    print('\033[32;1m' + '=' * 108 + '\033[0m')
