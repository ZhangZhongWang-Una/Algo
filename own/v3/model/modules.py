import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns, get_linear_logit, DEFAULT_GROUP_NAME
from deepctr.layers.utils import combined_dnn_input, concat_func, add_func, softmax, reduce_sum
from deepctr.layers.core import PredictionLayer, DNN

from tensorflow.python.keras import backend as K
import itertools
from itertools import chain
from tensorflow.python.keras.initializers import glorot_normal, zeros, Initializer
from tensorflow.python.keras.layers import Layer, Dropout, BatchNormalization
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
    def __init__(self, filters=64, **kwargs):
        self.filters = filters
        super(ConvLayer, self).__init__(**kwargs)
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
    def __init__(self, memory_size=8, dropout_rate=0.00, seed=1024, **kwargs):
        self.memory_size = memory_size
        self.seed = seed
        self.dropout_rate = dropout_rate
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
        self.dropout = Dropout(self.dropout_rate)
        self.batchNormal = BatchNormalization()
        super(MemoryLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs = self.batchNormal(inputs)
        att_key = tf.tensordot(inputs, self.key, axes=(-1, 0))
        att_mem = softmax(att_key)
        mem = tf.tensordot(att_mem, self.mem, axes=(-1, 0))
        # mem = tf.matmul(att_mem, self.mem)
        output = tf.multiply(mem, inputs)
        output = self.dropout(output)

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


class CGCLayer(Layer):
    def __init__(self, num_tasks=4, experts_num=4, experts_units=16, dropout_rate=0.00, seed=1024, **kwargs):
        self.num_tasks = num_tasks
        self.experts_num = experts_num
        self.experts_units = experts_units
        self.selector_num = 2
        self.dropout_rate = dropout_rate
        self.seed = seed
        super(CGCLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.dropout = Dropout(self.dropout_rate)
        self.batchNormal = BatchNormalization()
        self.experts_weight_share = self.add_weight(
                name='experts_weight_share_1',
                dtype=tf.float32,
                shape=(input_dim, self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed))
        self.experts_weight = []
        self.gate_weight = []

        for j in range(self.num_tasks):
            # experts Task j
            self.experts_weight.append(self.add_weight(
                name='experts_weight_task{}'.format(j),
                dtype=tf.float32,
                shape=(input_dim, self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed)
            ))
            # gates Task j
            self.gate_weight.append(self.add_weight(
                name='gate_weight_task{}'.format(j),
                dtype=tf.float32,
                shape=(input_dim, self.experts_num * self.selector_num),
                initializer=glorot_normal(seed=self.seed)
            ))

        super(CGCLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs = self.batchNormal(inputs)
        outputs = []
        # experts shared outputs
        experts_output_share = tf.tensordot(inputs, self.experts_weight_share, axes=(-1, 0))
        for j in range(self.num_tasks):
            experts_output_task = tf.tensordot(inputs, self.experts_weight[j], axes=(-1, 0))

            gate_output_task = tf.matmul(inputs, self.gate_weight[j])
            gate_output_task = tf.nn.softmax(gate_output_task)
            gate_output_task = tf.multiply(concat_func([experts_output_task, experts_output_share], axis=2),
                                           tf.expand_dims(gate_output_task, axis=1))
            gate_output_task = tf.reduce_sum(gate_output_task, axis=2)
            gate_output_task = tf.reshape(gate_output_task, [-1, self.experts_units])
            gate_output_task = self.dropout(gate_output_task)
            outputs.append(gate_output_task)

        return outputs


class MatrixInit(Initializer):
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, shape, dtype=None, partition_info=None):
        return K.variable(value=self.matrix, dtype=dtype)

    def get_config(self):
        return {
            'matrix': self.matrix
        }


class DCNMLayer(Layer):

    def __init__(self, low_rank=32, num_experts=4, layer_num=2, l2_reg=0, dropout_rate=0.00, seed=1024, **kwargs):
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.layer_num = layer_num
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.seed = seed
        super(DCNMLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])

        # U: (dim, low_rank)
        self.U_list = [self.add_weight(name='U_list' + str(i),
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]
        # V: (dim, low_rank)
        self.V_list = [self.add_weight(name='V_list' + str(i),
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]
        # C: (low_rank, low_rank)
        self.C_list = [self.add_weight(name='C_list' + str(i),
                                       shape=(self.num_experts, self.low_rank, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]

        self.gating = [tf.keras.layers.Dense(1, use_bias=False) for i in range(self.num_experts)]

        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        self.dropout = Dropout(self.dropout_rate)
        self.batchNormal = BatchNormalization()
        # Be sure to call this somewhere!
        super(DCNMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        # inputs = self.batchNormal(inputs)
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](tf.squeeze(x_l, axis=2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = tf.einsum('ij,bjk->bik', tf.transpose(self.V_list[i][expert_id]), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = tf.nn.tanh(v_x)
                v_x = tf.einsum('ij,bjk->bik', self.C_list[i][expert_id], v_x)  # (bs, low_rank, 1)
                v_x = tf.nn.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = tf.einsum('ij,bjk->bik', self.U_list[i][expert_id], v_x)  # (bs, dim, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(tf.squeeze(dot_, axis=2))

            # (3) mixture of low-rank experts
            output_of_experts = tf.stack(output_of_experts, 2)  # (bs, dim, num_experts)
            gating_score_of_experts = tf.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = tf.matmul(output_of_experts, tf.nn.softmax(gating_score_of_experts, 1))
            x_l = moe_out + x_l  # (bs, dim, 1)
            x_l = self.dropout(x_l)
        x_l = tf.squeeze(x_l, axis=2)
        return x_l
