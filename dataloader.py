import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import cv2
from PIL import Image
import config

class Dataset(data.Dataset):
    def __init__(self, data_list, skip_frame=1, time_step=30):
        '''
        定义一个数据集，从UCF101中读取数据
        '''
        # 定义图像的预处理函数
        self.transform = transforms.Compose([
            transforms.Resize((config.img_w, config.img_h)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        # 用来将类别转换为one-hot数据
        self.labels = []

        self.skip_frame = skip_frame
        self.time_step = time_step
        self.data_list = self._build_data_list(data_list)

    def __len__(self):
        return len(self.data_list) // self.time_step

    def __getitem__(self, index):
        # 每次读取time_step帧图片
        imgs = self.data_list[index:index + self.time_step]
        # 图片读取为RGB模式，并堆叠到一起
        X = torch.stack([
            self.transform(Image.open(img[2]).convert('RGB')) for img in imgs
        ], dim=0)
        # 为这些图片指定类别标签
        y = self._label_category(imgs[0][0])
        return X, y

    def _build_data_list(self, data_list=[]):
        '''
        构建数据集
        '''
        data_group = {}
        for x in tqdm(data_list, desc='Building dataset'):
            # 将视频分别按照classname和videoname分组
            [classname, videoname] = x[0:2]
            if classname not in data_group:
                data_group[classname] = {}
            if videoname not in data_group[classname]:
                data_group[classname] = []
            data_group[classname][videoname].append(x)

        # 处理类别变量
        self.labels = list(data_group.keys())

        ret_list = []

        # 填充数据
        for classname in data_group:
            video_group = data_group[classname]
            for videoname in video_group:
                # 如果某个视频的帧总数没法被time_step整除，那么需要按照最后一帧进行填充
                video_pad_count = len(video_group[videoname]) % self.time_step
                video_group[videoname] = [video_group[videoname][-1]] * (self.time_step - video_pad_count)
                ret_list += video_group[videoname]

        return ret_list

    def _label_one_hot(self, label):
        '''
        将标签转换成one-hot形式
        '''
        if label not in self.labels:
            raise RuntimeError('不存在的label！')
        one_hot = [0] * len(self.labels)
        one_hot[self.labels.index(label)] = 1
        return one_hot

    def _label_category(self, label):
        '''
        将标签转换成整型
        '''
        if label not in self.labels:
            raise RuntimeError('不存在的label！')
        return self.labels.index(label)