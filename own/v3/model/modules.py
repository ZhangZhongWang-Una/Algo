import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns, get_linear_logit, DEFAULT_GROUP_NAME
from deepctr.layers.utils import combined_dnn_input, concat_func, add_func, softmax, reduce_sum
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.models.afm import AFM

import itertools
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


class MutilAttLayer(Layer):
    def __init__(self, num_tasks, num_experts, output_dim, dropout_rate, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.seed = seed
        super(MutilAttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.dropout = tf.keras.layers.Dropout(
            self.dropout_rate, seed=self.seed)
        self.attention_W = []
        self.attention_b = []
        self.projection_h = []
        self.projection_p = []
        for i in range(self.num_tasks):
            self.attention_W.append(self.add_weight(
                name='attention_W_'.format(i),
                shape=(input_dim, self.num_experts),
                dtype=tf.float32,
                initializer=glorot_normal(seed=self.seed)))
            self.attention_b.append(self.add_weight(
                name="attention_b_".format(i),
                shape=(self.num_experts,),
                initializer=zeros()))
            self.projection_h.append(self.add_weight(
                name='projection_h_'.format(i),
                shape=(self.num_experts, 1),
                initializer=glorot_normal(seed=self.seed)))
            self.projection_p.append(self.add_weight(
                name='projection_p_'.format(i),
                shape=(input_dim, self.output_dim),
                initializer=glorot_normal(seed=self.seed)))

        super(MutilAttLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = tf.expand_dims(inputs, axis=1)
        outputs = []
        for i in range(self.num_tasks):
            # embeds_vec_list = inputs
            # row = []
            # col = []
            #
            # for r, c in itertools.combinations(embeds_vec_list, 2):
            #     row.append(r)
            #     col.append(c)
            #
            # p = tf.concat(row, axis=1)
            # q = tf.concat(col, axis=1)
            # inner_product = p * q
            #
            # bi_interaction = inner_product
            bi_interaction = inputs

            attention_temp = tf.nn.relu(tf.nn.bias_add(tf.tensordot(
                bi_interaction, self.attention_W[i], axes=(-1, 0)), self.attention_b[i]))
            self.normalized_att_score = softmax(tf.tensordot(
                attention_temp, self.projection_h[i], axes=(-1, 0)), dim=1)
            attention_output = reduce_sum(
                self.normalized_att_score * bi_interaction, axis=1)
            attention_output = self.dropout(attention_output)  # training
            output = tf.matmul(attention_output, self.projection_p[i])
            # output = self.tensordot([attention_output, self.projection_p[i]])
            outputs.append(output)
        return outputs


class MMOEAttLayer(Layer):
    def __init__(self, num_tasks, num_experts, output_dim, dropout_rate, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.seed = seed
        super(MMOEAttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.expert_kernel = self.add_weight(
            name='expert_kernel',
            shape=(input_dim, self.num_experts * self.output_dim),
            dtype=tf.float32,
            initializer=glorot_normal(seed=self.seed))
        self.dropout = tf.keras.layers.Dropout(
            self.dropout_rate, seed=self.seed)
        self.attention_W = []
        self.attention_b = []
        self.projection_h = []
        for i in range(self.num_tasks):
            self.attention_W.append(self.add_weight(
                name='attention_W_'.format(i),
                shape=(input_dim, self.num_experts),
                dtype=tf.float32,
                initializer=glorot_normal(seed=self.seed)))
            self.attention_b.append(self.add_weight(
                name="attention_b_".format(i),
                shape=(self.num_experts,),
                initializer=zeros()))
            self.projection_h.append(self.add_weight(
                name='projection_h_'.format(i),
                shape=(self.num_experts, 1),
                initializer=glorot_normal(seed=self.seed)))

        super(MMOEAttLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        expert_out = tf.tensordot(inputs, self.expert_kernel, axes=(-1, 0))
        expert_out = tf.reshape(expert_out, [-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            attention_temp = tf.nn.relu(tf.nn.bias_add(tf.tensordot(
                inputs, self.attention_W[i], axes=(-1, 0)), self.attention_b[i]))
            normalized_att_score = softmax(tf.tensordot(
                attention_temp, self.projection_h[i], axes=(-1, 0)), dim=1)
            gate_out = tf.tile(tf.expand_dims(normalized_att_score, axis=1), [1, self.output_dim, 1])
            output = tf.reduce_sum(tf.multiply(expert_out, gate_out), axis=2)
            outputs.append(output)
        return outputs


class MemoryLayer(Layer):
    def __init__(self, memory_size, seed=1024, **kwargs):
        self.memory_size = memory_size
        self.seed = seed
        super(MemoryLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.key = self.add_weight(
            name='key',
            shape=(input_dim, self.memory_size),
            dtype=tf.float32,
            initializer=glorot_normal(seed=self.seed))
        self.mem = self.add_weight(
            name='mem',
            shape=(self.memory_size, input_dim),
            dtype=tf.float32,
            initializer=glorot_normal(seed=self.seed))
        super(MemoryLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        att_key = tf.matmul(inputs, self.key)
        att_mem = softmax(att_key)
        mem = tf.matmul(att_mem, self.mem)
        output = tf.multiply(mem, inputs)
        return output