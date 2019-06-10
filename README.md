# Pytorch-Video-Classification
Make video classification on UCF101 using CNN and RNN based on Pytorch framework.

# Environments
```bash
# 1. torch >= 1.0
conda create -n lstm-cnn
source activate lstm-cnn # or `conda activate lstm-cnn`
# GPU version
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
# CPU version
conda install pytorch-cpu torchvision-cpu -c pytorch

# 2. depencies
pip install pandas scikit-learn tqdm opencv-python

# 3. prepare datasets
cp -r path/to/your/UCF ./data
cd ./data && python make_train_test.py

# 4. train your network
python train.py
```
