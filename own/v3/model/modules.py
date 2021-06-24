import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns, get_linear_logit, DEFAULT_GROUP_NAME
from deepctr.layers.utils import combined_dnn_input, concat_func, add_func, softmax, reduce_sum
from deepctr.layers.core import PredictionLayer, DNN

from tensorflow.python.keras import backend as K
import itertools
from itertools import chain
from tensorflow.python.keras.initializers import glorot_normal, zeros
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2


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


class ConvLayer(Layer):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(ConvLayer, self).__init__(**kwargs)
        # self.conv = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=1, activation='relu', use_bias=True)
        self.conv = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=1)
        self.batchNormal = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')

    def call(self, inputs, **kwargs):
        inputs = tf.expand_dims(inputs, axis=1)
        conv_out = self.conv(inputs)
        batchNormal_out = self.batchNormal(conv_out)
        act_out = self.activation(batchNormal_out)
        output = tf.squeeze(act_out, axis=1)
        return output


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
        # att_key = tf.matmul(inputs, self.key)
        att_key = tf.tensordot(inputs, self.key, axes=(-1, 0))
        att_mem = softmax(att_key)
        mem = tf.tensordot(att_mem, self.mem, axes=(-1, 0))
        # mem = tf.matmul(att_mem, self.mem)
        output = tf.multiply(mem, inputs)
        return output


class PLELayer(Layer):
    def __init__(self, num_tasks, num_level, experts_num, experts_units, seed=1024, **kwargs):
        self.num_tasks = num_tasks
        self.num_level = num_level
        self.experts_num = experts_num
        self.experts_units = experts_units
        self.selector_num = 2
        self.seed = seed
        super(PLELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.experts_weight_share = [
            self.add_weight(
                name='experts_weight_share_1',
                dtype=tf.float32,
                shape=(input_dim, self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed)),
            self.add_weight(
                name='experts_weight_share_2',
                dtype=tf.float32,
                shape=(self.experts_units, self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed))
        ]
        self.experts_bias_share = [
            self.add_weight(
                name='expert_bias_share_1',
                dtype=tf.float32,
                shape=(self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed)),
            self.add_weight(
                name='expert_bias_share_2',
                dtype=tf.float32,
                shape=(self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed)
            )
        ]
        self.gate_weight_share = [
            self.add_weight(
                name='gate_weight_share_1',
                dtype=tf.float32,
                shape=(input_dim, self.experts_num * (self.num_tasks + 1)),
                initializer=glorot_normal(seed=self.seed)
            )
        ]
        self.gate_bias_share = [
            self.add_weight(
                name='gate_bias_share_1',
                dtype=tf.float32,
                shape=(self.experts_num * (self.num_tasks + 1),),
                initializer=glorot_normal(seed=self.seed)
            )
        ]
        self.experts_weight = [[], []]
        self.experts_bias = [[], []]
        self.gate_weight = [[], []]
        self.gate_bias = [[], []]

        for i in range(self.num_level):
            if 1 == i:
                input_dim = self.experts_units

            for j in range(self.num_tasks):
                # experts Task j
                self.experts_weight[i].append(self.add_weight(
                    name='experts_weight_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(input_dim,self.experts_units, self.experts_num),
                    initializer=glorot_normal(seed=self.seed)
                ))
                self.experts_bias[i].append(self.add_weight(
                    name='expert_bias_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(self.experts_units, self.experts_num),
                    initializer=glorot_normal(seed=self.seed)
                ))
                # gates Task j
                self.gate_weight[i].append(self.add_weight(
                    name='gate_weight_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(input_dim, self.experts_num * self.selector_num),
                    initializer=glorot_normal(seed=self.seed)
                ))
                self.gate_bias[i].append(self.add_weight(
                    name='gate_bias_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(self.experts_num * self.selector_num,),
                    initializer=glorot_normal(seed=self.seed)
                ))
        super(PLELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        gate_output_task_final = [inputs, inputs, inputs, inputs]
        gate_output_share_final = inputs
        for i in range(self.num_level):
            # experts shared outputs
            experts_output_share = tf.tensordot(gate_output_share_final, self.experts_weight_share[i], axes=1)
            experts_output_share = tf.add(experts_output_share, self.experts_bias_share[i])
            experts_output_share = tf.nn.relu(experts_output_share)
            experts_output_task_tmp = []
            for j in range(self.num_tasks):
                experts_output_task = tf.tensordot(gate_output_task_final[j], self.experts_weight[i][j], axes=1)
                experts_output_task = tf.add(experts_output_task, self.experts_bias[i][j])
                experts_output_task = tf.nn.relu(experts_output_task)
                experts_output_task_tmp.append(experts_output_task)
                gate_output_task = tf.matmul(gate_output_task_final[j], self.gate_weight[i][j])
                gate_output_task = tf.add(gate_output_task, self.gate_bias[i][j])
                gate_output_task = tf.nn.softmax(gate_output_task)
                gate_output_task = tf.multiply(concat_func([experts_output_task, experts_output_share], axis=2),
                                               tf.expand_dims(gate_output_task, axis=1))
                gate_output_task = tf.reduce_sum(gate_output_task, axis=2)
                gate_output_task = tf.reshape(gate_output_task, [-1, self.experts_units])
                gate_output_task_final[j] = gate_output_task

            if 0 == i:
                # gates shared outputs
                gate_output_shared = tf.matmul(gate_output_share_final, self.gate_weight_share[i])
                gate_output_shared = tf.add(gate_output_shared, self.gate_bias_share[i])
                gate_output_shared = tf.nn.softmax(gate_output_shared)
                gate_output_shared = tf.multiply(concat_func(experts_output_task_tmp + [experts_output_share], axis=2),
                                                 tf.expand_dims(gate_output_shared, axis=1))
                gate_output_shared = tf.reduce_sum(gate_output_shared, axis=2)
                gate_output_shared = tf.reshape(gate_output_shared, [-1, self.experts_units])
                gate_output_share_final = gate_output_shared
        return gate_output_task_final