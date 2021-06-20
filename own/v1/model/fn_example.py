# Encoding = UTF-8
import numpy as np
import tensorflow as tf
from tensorflow import keras


def network(x, feature_column):
    """
    :purpose: 定义模型结构，这里我们通过tf.layers来实现
    :param x: 输入层张量
    :return: 返回前向传播的结果
    """

    net = tf.feature_column.input_layer(x, feature_column)
    net = tf.layers.Dense(units= 100)(net)
    net = tf.layers.Dense(units= 100)(net)
    net = tf.layers.Dense(units= 1)(net)
    return net

def my_model_fn(
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
    predict = network(features, params['feature_columns'])

    # 如果在预测模式，那么只需要将结果返回即可
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode= mode,
            predictions= {'result': predict}
        )

    # 定义损失函数
    loss = tf.reduce_mean(predict - labels)

    # 定义优化函数
    optimizer = tf.train.GradientDescentOptimizer(learning_rate= params['learning_rate'])

    # 定义训练过程
    train_op = optimizer.minimize(loss= loss, global_step= tf.train.get_global_step())

    # 定义评测标准，在运行评估操作的时候会计算这里定义的所有评测标准
    eval_metric_ops = tf.metrics.accuracy(predict, labels, name= 'acc')

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode= mode,
            loss= loss,
            train_op= train_op,
            eval_metric_ops= {'accuracy': eval_metric_ops})

    # 定义评估操作
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode = mode,
            loss = loss,
            eval_metric_ops= {'accuracy': eval_metric_ops})

# 通过自定义的方式生成Estimator类，这里需要提供模型定义的函数并通过params参数指定模型定义时使用的超参数

feature_columns = [tf.feature_column.numeric_column('att', shape= [2])]
model_params = {'learning_rate': 0.01,
                'feature_columns': feature_columns
                }

estimator = tf.estimator.Estimator(model_fn= my_model_fn,
                                   params= model_params,
                                   model_dir= './')

def input_fn():
    """
    :purpose: 输入训练数据
    :return: 特征值和标签值
    """
    data = np.load(r'C:\Users\12394\PycharmProjects\TensorFlow\blog.npz')
    dataset = tf.data.Dataset.from_tensor_slices((data['att'], data['label'])).batch(5)
    iterator = dataset.make_one_shot_iterator()
    att, label = iterator.get_next()
    return {'att': att}, label
# 训 练
estimator.train(input_fn= input_fn, steps= 30000)
# 评 估
estimator.evaluate(input_fn=input_fn, steps= 200)
# 预 测
predictions = estimator.predict(input_fn= input_fn)

for item in predictions:
    print(item['result'])
