import torch
from torch.utils import data
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self):
        '''
        定义一个数据集，从UCF101中读取数据
        '''
        # 定义图像的预处理函数
        self.transform = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    def read_images(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        pass