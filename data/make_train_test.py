import os
import cv2
import numpy as np
from tqdm import tqdm

file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(file_dir, 'UCF')

def split(size=0.2):
    # 生成测试集和训练集目录
    for folder in ['train', 'test']:
        folder_path = os.path.join(file_dir, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print('Folder {} is created'.format(folder_path))
    # 划分测试集和训练集
    train_set = []
    test_set = []
    classes = os.listdir(src_dir)
    for classname in classes:
        # 读取所有视频路径
        videos = os.listdir(os.path.join(src_dir, classname))
        # 打乱视频名称
        np.random.shuffle(videos)
        # 确定测试集划分点
        split_size = int(len(videos) * size)

        # 生成训练集和测试集的文件夹
        for i in range(2):
            part = ['train', 'test'][i]
            class_dir = os.path.join(file_dir, part, classname)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

        # 遍历每个视频，将每个视频的图像帧提取出来
        for i in tqdm(range(len(videos))):
            video_path = os.path.join(src_dir, classname, videos[i])
            video_fd = cv2.VideoCapture(video_path)

            if not video_fd.isOpened():
                print('Skpped: {}'.format(video_path))
                continue

            video_type = 'test' if i <= split_size else 'train'

            frame_index = 0
            success, frame = video_fd.read()
            video_name = videos[i].rsplit('.')[0]
            while success:
                img_path = os.path.join(file_dir, video_type, classname, '%s_%d.jpg' % (video_name, frame_index))
                cv2.imwrite(img_path, frame)
                info = [classname, video_name, img_path]
                # 将视频帧信息保存起来
                if video_type == 'test':
                    test_set.append(info)
                else:
                    train_set.append(info)
                frame_index += 1
                success, frame = video_fd.read()

            video_fd.release()

        # 将训练集和测试集数据保存到文件中，方便写dataloader
        datas = [train_set, test_set]
        names = ['train', 'test']
        for i in range(2):
            with open(file_dir + '/' + names[i] + '.csv', 'w') as f:
                f.write('\n'.join([','.join(line) for line in datas[i]]))

if __name__ == '__main__':
    test_size = 0.2 # 测试集比例
    split(size=test_size)
