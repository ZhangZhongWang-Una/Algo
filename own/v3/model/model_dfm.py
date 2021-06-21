# -*- coding:utf-8 -*-
import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns, get_linear_logit, DEFAULT_GROUP_NAME
from deepctr.layers.utils import combined_dnn_input, concat_func, add_func, softmax, reduce_sum
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.models.afm import AFM

from itertools import chain
from tensorflow.python.keras.initializers import glorot_normal, zeros
from tensorflow.python.keras.layers import Layer


class MMOELayer(Layer):
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
            initializer=glorot_normal(seed=self.seed))
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(self.add_weight(
                name='gate_weight_'.format(i),
                shape=(input_dim, self.num_experts),
                dtype=tf.float32,
                initializer=glorot_normal(seed=self.seed)))
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


def Model(fixlen_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu',
         l2_reg_linear=0.00001):
    print('\033[32;1m[MODEL]\033[0m model_dfm.py \n')
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    dnn_feature_columns = fixlen_feature_columns[:5]
    linear_feature_columns = fixlen_feature_columns[5:]

    features = build_input_features(fixlen_feature_columns)
    inputs_list = list(features.values())

    linear_out = get_linear_logit(features, linear_feature_columns, units=dnn_hidden_units[1], seed=seed,
                                  prefix='linear', l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, fixlen_feature_columns,
                                                                         l2_reg_embedding, seed)

    fm_logit = add_func([FM()(l) for l in sparse_embedding_list])

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,False, seed=seed)(dnn_input)

    concat_out = add_func([linear_out, dnn_out])

    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(concat_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        logit = add_func([fm_logit, logit])
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model
