from own.v3.model.modules import *


def Model(dnn_feature_columns, num_tasks, tasks, flags):
    print('\033[32;1m[MODEL]\033[0m model_att.py \n')
    l2_reg_embedding = 1e-5
    dnn_activation = 'relu'
    task_dnn_units = None

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, flags.seed)
    emb_out = combined_dnn_input(sparse_embedding_list, dense_value_list)

    dnn_out = DNN((flags.dnn1, flags.dnn2), dnn_activation, flags.l2, flags.dropout, use_bn=False, seed=flags.seed)(emb_out)

    mmoe_outs = MMOELayer(num_tasks, flags.expert_num, flags.expert_dim)(dnn_out)
    # mmoe_outs = MutilAttLayer(num_tasks, flags.expert_num, flags.expert_dim, flags.dropout)(dnn_out)
    # mmoe_outs = MMOEAttLayer(num_tasks, flags.expert_num, flags.expert_dim, flags.dropout)(dnn_out)

    if task_dnn_units != None:
        mmoe_outs = [DNN((flags.dnn1, flags.dnn2), dnn_activation, flags.l2, flags.dropout, use_bn=False, seed=flags.seed)
                     (mmoe_out) for mmoe_out in mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outputs)
    return model
