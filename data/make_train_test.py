import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# 数据集的默认位置
default_output_dir = os.path.dirname(os.path.abspath(__file__))
default_src_dir = os.path.join(default_output_dir, 'UCF')
default_test_size = 0.2

def split(src_dir=default_src_dir, output_dir=default_src_dir, size=default_test_size):
    # 设置默认参数
    src_dir = default_output_dir if src_dir is None else src_dir
    output_dir = default_output_dir if output_dir is None else output_dir
    size = default_test_size if size is None else size

    # 生成测试集和训练集目录
    for folder in ['train', 'test']:
        folder_path = os.path.join(output_dir, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print('Folder {} is created'.format(folder_path))

    # 划分测试集和训练集
    train_set = []
    test_set = []
    classes = os.listdir(src_dir)
    num_classes = len(classes)
    for class_index, classname in enumerate(classes):
        # 读取所有视频路径
        videos = os.listdir(os.path.join(src_dir, classname))
        # 打乱视频名称
        np.random.shuffle(videos)
        # 确定测试集划分点
        split_size = int(len(videos) * size)

        # 生成训练集和测试集的文件夹
        for i in range(2):
            part = ['train', 'test'][i]
            class_dir = os.path.join(output_dir, part, classname)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

        # 遍历每个视频，将每个视频的图像帧提取出来
        for i in tqdm(range(len(videos)), desc='[%d/%d]%s' % (class_index, num_classes, classname)):
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
                img_path = os.path.join(output_dir, video_type, classname, '%s_%d.jpg' % (video_name, frame_index))
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
            with open(output_dir + '/' + names[i] + '.csv', 'w') as f:
                f.write('\n'.join([','.join(line) for line in datas[i]]))

def parse_args():
    parser = argparse.ArgumentParser(usage='命令示例 python3 make_train_test.py -i path/to/UCF -o path/to/output -s 0.3')
    parser.add_argument('-i', '--input_dir', help='UCF数据集所在路径')
    parser.add_argument('-o', '--output_dir', help='生成的训练集和测试集所在路径')
    parser.add_argument('-s', '--split_size', help='测试集占比')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    split(src_dir=args.input_dir, output_dir=args.output_dir, size=args.split_size)
