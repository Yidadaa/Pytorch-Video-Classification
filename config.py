img_w = 224
img_h = 224

dataset_params = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True
}

cnn_encoder_params = {
    'cnn_out_dim': 256,
    'drop_prob': 0.3,
    'bn_momentum': 0.01
}

rnn_decoder_params = {
    'use_gru': True,
    'cnn_out_dim': 256,
    'rnn_hidden_layers': 3,
    'rnn_hidden_nodes': 256,
    'num_classes': 10,
    'drop_prob': 0.3
}

learning_rate = 1e-5
epoches = 10
log_interval = 2 # 打印间隔，默认每2个batch_size打印一次
save_interval = 1 # 模型保存间隔，默认每个epoch保存一次