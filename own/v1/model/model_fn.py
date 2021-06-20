# Encoding = UTF-8
import numpy as np
from tensorflow import keras
import tensorflow.compat.v1 as tf
from own.v1.comm import STAGE_END_DAY
from own.v1.utils import *
from own.v1.evaluation import uAUC


class WideDeepLayer(keras.layers.Layer):
    def __init__(self, dnn_hidden_units=(32, 8), output_dim=1, **kwargs):
        dnn_hidden_units = list(dnn_hidden_units)
        self.hidden1_layer = keras.layers.Dense(dnn_hidden_units[0], activation='relu')
        self.hidden2_layer = keras.layers.Dense(dnn_hidden_units[1], activation='relu')
        self.output_layer = keras.layers.Dense(output_dim)
        super(WideDeepLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_wide = inputs[0]
        input_deep = inputs[1]
        hidden1 = self.hidden1_layer(input_deep)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([input_wide, hidden2])
        output = self.output_layer(concat)
        return output


class MMOELayer(keras.layers.Layer):
    def __init__(self, num_tasks, num_experts, output_dim, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.seed = seed
        super(MMOELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.expert_kernel = self.add_weight(
            name='expert_kernel',
            shape=(input_dim, self.num_experts * self.output_dim),
            dtype=tf.float32,
            initializer=keras.initializers.glorot_normal(seed=self.seed))
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(self.add_weight(
                name='gate_weight_'.format(i),
                shape=(input_dim, self.num_experts),
                dtype=tf.float32,
                initializer=keras.initializers.glorot_normal(seed=self.seed)))
        super(MMOELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        expert_out = tf.tensordot(inputs, self.expert_kernel, axes=(-1, 0))
        expert_out = tf.reshape(expert_out, [-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            gate_out = tf.tensordot(inputs, self.gate_kernels[i], axes=(-1, 0))
            gate_out = tf.nn.softmax(gate_out)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=1), [1, self.output_dim, 1])
            output = tf.reduce_sum(tf.multiply(expert_out, gate_out), axis=2)
            outputs.append(output)
        return outputs

    def get_config(self):

        config = {'num_tasks': self.num_tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim}
        base_config = super(MMOELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim] * self.num_tasks


class PredictionLayer(keras.layers.Layer):
    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=keras.initializers.Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def network(features, linear_feature_columns, dnn_feature_columns):
    input_linear, input_dnn = features[0], features[1]
    linear = tf.feature_column.input_layer(input_linear, linear_feature_columns)
    dnn = tf.feature_column.input_layer(input_dnn, dnn_feature_columns)
    input_list = [linear, dnn]
    wide_deep_out = WideDeepLayer()(input_list)
    mmoe_outs = MMOELayer(4, 4, 8)(wide_deep_out)
    task_outputs = []
    tasks = ['binary', 'binary', 'binary', 'binary']
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)
    return task_outputs


def model_fn(
        features,
        labels,
        mode,
        params
):
    """
    :purpose: 模型函数
    :param features: 输入函数中返回的数据特征
    :param labels: 数据标签
    :param mode: 表示调用程序是请求训练、预测还是评估
    :param params: 可传递的额外的参数
    :return: 返回模型训练过程需要使用的损失函数、训练过程和评测方法
    """

    # 定义神经网络的结构并通过输入得到前向传播的结果
    predict = network(features, params['linear_feature_columns'], params['dnn_feature_columns'])

    # 如果在预测模式，那么只需要将结果返回即可
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'result': predict}
        )

    # 定义损失函数
    loss = tf.keras.metrics.binary_crossentropy(labels, predict)

    # 定义优化函数
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=0.9, beta2=0.999,
                                       epsilon=1)

    # 定义训练过程
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    # 定义评测标准，在运行评估操作的时候会计算这里定义的所有评测标准
    eval_metric_ops = tf.metrics.accuracy(predict, labels, name='acc')

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={'accuracy': eval_metric_ops})

    # 定义评估操作
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={'accuracy': eval_metric_ops})


class Model(object):

    def __init__(self, linear_feature_columns, dnn_feature_columns, flags, seed):
        super(Model, self).__init__()
        self.num_epochs_dict = {"read_comment": 1, "like": 1, "click_avatar": 1, "forward": 1}
        self.estimator = None
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.flags = flags
        self.seed = seed

    def build_estimator(self):
        model_checkpoint_stage_dir = self.flags.model_checkpoint_dir
        if not os.path.exists(model_checkpoint_stage_dir):
            # 如果模型目录不存在，则创建该目录
            os.makedirs(model_checkpoint_stage_dir)
        del_file(model_checkpoint_stage_dir)

        model_params = {'learning_rate': self.flags.learning_rate,
                        'linear_feature_columns': self.linear_feature_columns,
                        'dnn_feature_columns': self.dnn_feature_columns
                        }

        self.estimator = tf.estimator.Estimator(model_fn=model_fn,
                                           params=model_params,
                                           model_dir=model_checkpoint_stage_dir)

    def df_to_dataset(self, df, stage, action, shuffle=True, batch_size=128, num_epochs=1):
        if stage != "submit":
            label = df[action]
            ds = tf.data.Dataset.from_tensor_slices((dict(df), label))
        else:
            ds = tf.data.Dataset.from_tensor_slices((dict(df)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=self.seed)
        ds = ds.batch(batch_size)
        if stage in ["online_train", "offline_train"]:
            ds = ds.repeat(num_epochs)
        return ds

    def input_fn_train(self, df, stage, action, num_epochs):
        return self.df_to_dataset(df, stage, action, shuffle=True, batch_size=self.flags.batch_size,
                                  num_epochs=num_epochs)

    def input_fn_predict(self, df, stage, action):
        return self.df_to_dataset(df, stage, action, shuffle=False, batch_size=len(df), num_epochs=1)

    def train(self, inputs, label, batch_size, epochs, shuffle=True):

        ds = tf.data.Dataset.from_tensor_slices((dict(inputs), label))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(inputs), seed=self.seed)
        ds = ds.batch(batch_size)
        ds = ds.repeat(epochs)
        self.estimator.train(input_fn=lambda: ds)

    def evaluate(self, inputs, batch_size, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((dict(inputs)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(inputs), seed=self.seed)
        ds = ds.batch(batch_size)
        predicts = self.estimator.predict(input_fn=lambda: ds)
        return predicts

    # def predict(self):
    #     '''
    #     预测单个行为的发生概率
    #     '''
    #     file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action="all",
    #                                                                    day=STAGE_END_DAY[self.stage])
    #     submit_dir = os.path.join(self.flags.root_path, self.stage, file_name)
    #     df = pd.read_csv(submit_dir)
    #     t = time.time()
    #     predicts = self.estimator.predict(
    #         input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
    #     )
    #     predicts_df = pd.DataFrame.from_dict(predicts)
    #     logits = predicts_df["logistic"].map(lambda x: x[0])
    #     # 计算2000条样本平均预测耗时（毫秒）
    #     ts = (time.time() - t) * 1000.0 / len(df) * 2000.0
    #     return df[["userid", "feedid"]], logits, ts