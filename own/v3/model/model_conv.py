# -*- coding:utf-8 -*-
from own.v3.model.modules import *


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


def Model(dnn_feature_columns, num_tasks, tasks, flags):
    print('\033[32;1m[MODEL]\033[0m model_conv.py \n')
    l2_reg_embedding = 1e-5
    dnn_activation = 'relu'

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, flags.seed)
    emb_out = combined_dnn_input(sparse_embedding_list, dense_value_list)

    mem_out = MemoryLayer(memory_size=flags.mem_size)(emb_out)
    conv_out = ConvLayer(64)(mem_out)
    dnn_out = DNN((flags.dnn1, flags.dnn2), dnn_activation, flags.l2, flags.dropout, use_bn=False, seed=flags.seed)(conv_out)
    mmoe_outs = MMOELayer(num_tasks, flags.expert_num, flags.expert_dim)(dnn_out)
    # mmoe_outs = MMOEAttLayer(num_tasks, flags.expert_num, flags.expert_dim, flags.dropout)(dnn_out)
    # mmoe_outs = MutilAttLayer(num_tasks, flags.expert_num, flags.expert_dim, flags.dropout)(dnn_out)


    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outputs)
    return model
