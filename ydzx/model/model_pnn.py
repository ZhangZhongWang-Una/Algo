import tensorflow.python.keras as keras
import tensorflow as tf
import tensorflow.python.keras.backend as K
from deepctr.feature_column import build_input_features, input_from_feature_columns
from tensorflow.python.keras.initializers import zeros, random_normal, truncated_normal, constant, glorot_normal
from tensorflow.python.keras.layers import Layer, Embedding, Input, Dense, BatchNormalization,\
    Activation, Conv1D, GlobalAveragePooling1D,subtract, maximum, add, Dropout, multiply, Concatenate, Reshape
from tensorflow.python.keras.regularizers import l2
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input, concat_func, add_func, softmax, reduce_sum
from deepctr.layers.interaction import InnerProductLayer, OutterProductLayer
from deepctr.models import PNN

from ydzx.model.model_deepctr import ConvLayer


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


class AffinityLayer(Layer):
    def __init__(self, matrix_dim, l2_rate, seed=2021):
        super(AffinityLayer, self).__init__()
        self.matrix_dim = matrix_dim
        self.affinity_matrix = self.add_weight(
            name='affinity_matrix',
            shape=(matrix_dim, matrix_dim),
            dtype=tf.float32,
            initializer=truncated_normal(stddev=0.3, seed=seed),
            regularizer=l2(l2_rate))

    def call(self, inputs, **kwargs):
        input_u, input_i = inputs
        tmp = tf.matmul(input_u, self.affinity_matrix)
        f = tf.matmul(tmp, input_i, transpose_b=True)
        # f = tf.expand_dims(f, -1)
        att1 = tf.tanh(f)
        print(att1)
        pool_user = tf.reduce_mean(att1, 0)
        pool_item = tf.reduce_mean(att1, 1)

        user_flat = tf.squeeze(pool_user, -1)
        item_flat = tf.squeeze(pool_item, -1)

        weight_user = tf.nn.softmax(user_flat)
        weight_item = tf.nn.softmax(item_flat)

        weight_user_exp = tf.expand_dims(weight_user, -1)
        weight_item_exp = tf.expand_dims(weight_item, -1)

        output_u = input_u * weight_user_exp
        output_i = input_i * weight_item_exp

        return output_u, output_i


class FeaLayer(Layer):
    def __init__(self, ):
        super(FeaLayer, self).__init__()

    def call(self, inputs, **kwargs):
        user, item = inputs
        # id_u = tf.squeeze(id_u, axis=1)
        # id_i = tf.squeeze(id_i, axis=1)

        output = tf.concat([multiply([user, item]), user, item], axis=-1)

        return output


def Model(feature_columns, args, num_users, num_items):
    print('\033[32;1m[MODEL]\033[0m Model_pnn.py \n')
    use_inner = True
    use_outter = False

    id_feature_columns, input_feature_columns = feature_columns
    id_features = build_input_features(id_feature_columns)
    edge_features = build_input_features(input_feature_columns)

    inputs_list = list(id_features.values()) + list(edge_features.values())

    # itemId_emb = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(itemId_emb)

    id_embedding_list, _ = input_from_feature_columns(id_features, id_feature_columns, '1e-5', args.seed)
    edge_sparse_ebmeddings, edge_dense_ebmeddings = input_from_feature_columns(edge_features, input_feature_columns, '1e-5', args.seed)
    edge_emb = combined_dnn_input(edge_sparse_ebmeddings, edge_dense_ebmeddings)

    # user_emb = tf.keras.layers.Flatten()(concat_func(user_embedding_list))
    user_inner_product = tf.keras.layers.Flatten()(InnerProductLayer()(id_embedding_list))
    user_outter_product = OutterProductLayer('mat')(id_embedding_list)
    user_linear_signal = tf.keras.layers.Reshape([sum(map(lambda x: int(x.shape[-1]), id_embedding_list))])(concat_func(id_embedding_list))

    # ipnn deep input
    if use_inner and use_outter:
        user_emb = Concatenate()([user_linear_signal, user_inner_product, user_outter_product, edge_emb])
    elif use_inner:
        user_emb = Concatenate()([user_linear_signal, user_inner_product, edge_emb])
    elif use_outter:
        user_emb = Concatenate()([user_linear_signal, user_outter_product, edge_emb])
    else:
        user_emb = Concatenate()([user_linear_signal, edge_emb])

    # 交互层
    mem_out = MemoryLayer(memory_size=args.mem_size)(user_emb)
    conv_out = ConvLayer(args.conv_dim)(mem_out)
    dnn_out = DNN((args.dnn1, args.dnn2), 'relu', args.l2, args.dropout, use_bn=False, seed=args.seed)(conv_out)

    logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(dnn_out)
    output = PredictionLayer('binary')(logit)

    model = keras.models.Model(
        inputs=inputs_list,
        outputs=output,
        name='model_drsn'
    )
    return model