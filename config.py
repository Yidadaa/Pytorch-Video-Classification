img_w = 224
img_h = 224

train_dataset_params = {
    'batch_size': 30,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True
}

learning_rate = 1e-5
epoches = 120
log_interval = 20 # 每20个batchsize打印一次信息