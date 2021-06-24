# -*- coding:utf-8 -*-
from own.v3.model.modules import *
from deepctr.layers.sequence import Transformer
import numpy as np
import tensorflow.python.keras as keras
from deepctr.layers.interaction import AFMLayer, CrossNetMix, FGCNNLayer, FM, BiInteractionPooling, CIN
from deepctr.models import AFM


class EmbeddingLayer(Layer):
    def __init__(self,seed=1024, **kwargs):
        self.seed = seed
        super(EmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        emb = tf.stack(inputs, axis=1)
        emb = tf.squeeze(emb, axis=2)
        return emb


def Model(dnn_feature_columns, num_tasks, tasks, flags):
    print('\033[32;1m[MODEL]\033[0m model_trm.py \n')
    l2_reg_embedding = 1e-5
    dnn_activation = 'relu'

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, flags.seed)
    # emb_out = combined_dnn_input(sparse_embedding_list, dense_value_list)
    emb_out = EmbeddingLayer()(sparse_embedding_list)
    mem_out = MemoryLayer(memory_size=flags.mem_size)(emb_out)
    cin_out = CIN()(mem_out)
    cnm_out = CrossNetMix()(cin_out)
    # dnn_out = DNN((flags.dnn1, flags.dnn2), dnn_activation, flags.l2, flags.dropout, use_bn=False, seed=flags.seed)(mem_out)
    mmoe_outs = MMOELayer(num_tasks, flags.expert_num, flags.expert_dim)(cnm_out)

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outputs)
    return model

