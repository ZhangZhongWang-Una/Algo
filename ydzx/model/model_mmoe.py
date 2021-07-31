import tensorflow.python.keras as keras
import tensorflow as tf
import tensorflow.python.keras.backend as K
from deepctr.feature_column import build_input_features, input_from_feature_columns
from tensorflow.python.keras.initializers import zeros, random_normal, truncated_normal, constant, glorot_normal
from tensorflow.python.keras.layers import Layer, Embedding, Input, Dense, BatchNormalization,\
    Activation, Conv1D, GlobalAveragePooling1D,subtract, maximum, add, Dropout, multiply
from tensorflow.python.keras.regularizers import l2
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input, concat_func, add_func, softmax, reduce_sum


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
        # inputs = tf.squeeze(inputs, axis=1)
        att_key = tf.tensordot(inputs, self.key, axes=(-1, 0))
        att_mem = softmax(att_key)
        mem = tf.tensordot(att_mem, self.mem, axes=(-1, 0))
        # mem = tf.matmul(att_mem, self.mem)
        output = tf.multiply(mem, inputs)
        output = self.dropout(output)

        return output


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


def Model(input_feature_columns, args, num_users, num_items):
    print('\033[32;1m[MODEL]\033[0m Model_base.py \n')
    features = build_input_features(input_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, input_feature_columns, '1e-5', args.seed)
    emb_out = combined_dnn_input(sparse_embedding_list, dense_value_list)

    # 交互层
    mem_out = MemoryLayer(memory_size=args.mem_size)(emb_out)
    conv_out = ConvLayer(args.conv_dim)(mem_out)
    dnn_out = DNN((args.dnn1, args.dnn2), 'relu', args.l2, args.dropout, use_bn=False, seed=args.seed)(conv_out)
    mmoe_outs = CGCLayer(1, args.expert_num, args.expert_dim)(dnn_out)

    logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_outs)
    output = PredictionLayer('binary')(logit)

    model = keras.models.Model(
        inputs=inputs_list,
        outputs=output,
        name='model_drsn'
    )
    return model