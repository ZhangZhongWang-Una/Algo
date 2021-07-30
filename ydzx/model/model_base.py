import tensorflow.python.keras as keras
import tensorflow as tf
import tensorflow.python.keras.backend as K
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
        inputs = tf.squeeze(inputs, axis=1)
        att_key = tf.tensordot(inputs, self.key, axes=(-1, 0))
        att_mem = softmax(att_key)
        mem = tf.tensordot(att_mem, self.mem, axes=(-1, 0))
        # mem = tf.matmul(att_mem, self.mem)
        output = tf.multiply(mem, inputs)
        output = self.dropout(output)

        return output


def Model(args, num_users, num_items):
    print('\033[32;1m[MODEL]\033[0m Model_base.py \n')
    # input layer
    input_user_id = Input(shape=(1,), name='input_user_id')
    input_item_id = Input(shape=(1,), name='input_item_id')
    # embedding layer
    user_embeddings = Embedding(num_users,
                                args.emb_dim,
                                name='user_id_embeddings',
                                embeddings_initializer=random_normal(mean=0, stddev=0.02),
                                embeddings_regularizer=l2(args.l2))
    item_embeddings = Embedding(num_items,
                                args.emb_dim,
                                name='item_id_embeddings',
                                embeddings_initializer=random_normal(mean=0, stddev=0.02),
                                embeddings_regularizer=l2(args.l2))

    user_bias_embeddings = Embedding(num_users,
                                     1,
                                     name='user_bias_embeddings',
                                     embeddings_initializer=random_normal(mean=0, stddev=0.02),
                                     embeddings_regularizer=l2(args.l2))
    item_bias_embeddings = Embedding(num_items,
                                     1,
                                     name='item_bias_embeddings',
                                     embeddings_initializer=random_normal(mean=0, stddev=0.02),
                                     embeddings_regularizer=l2(args.l2))
    user_id_embeds = user_embeddings(input_user_id)
    item_id_embeds = item_embeddings(input_item_id)
    user_bias = user_bias_embeddings(input_user_id)
    item_bias = item_bias_embeddings(input_item_id)
    emb_out = concat_func([user_id_embeds, item_id_embeds], axis=-1)

    # 交互层
    mem_out = MemoryLayer(memory_size=args.mem_size)(emb_out)
    conv_out = ConvLayer(args.conv_dim)(mem_out)
    dnn_out = DNN((args.dnn1, args.dnn2), 'relu', args.l2, args.dropout, use_bn=False, seed=args.seed)(conv_out)

    logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(dnn_out)
    output = PredictionLayer('binary')(logit)

    model = keras.models.Model(
        inputs=[input_user_id, input_item_id],
        outputs=output,
        name='model_drsn'
    )
    return model