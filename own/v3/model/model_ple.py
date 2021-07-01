# -*- coding:utf-8 -*-
from own.v3.model.modules import *
from deepctr.models import din

def Model(dnn_feature_columns, num_tasks, tasks, flags):
    print('\033[32;1m[MODEL]\033[0m model_ple.py \n')
    l2_reg_embedding = 1e-5
    dnn_activation = 'relu'

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, flags.seed)
    emb_out = combined_dnn_input(sparse_embedding_list, dense_value_list)

    mem_out = MemoryLayer(memory_size=flags.mem_size)(emb_out)
    dnn_out = DNN((flags.dnn1, flags.dnn2), dnn_activation, flags.l2, flags.dropout, use_bn=False, seed=flags.seed)(mem_out)
    # mmoe_outs = MMOELayer(num_tasks, flags.expert_num, flags.expert_dim)(dnn_out)
    # mmoe_outs = PLELayer(num_tasks, 2, flags.expert_num, flags.expert_dim)(mem_out)
    mmoe_outs = CGCLayer(num_tasks, flags.expert_num, flags.expert_dim)(dnn_out)
    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outputs)
    return model
