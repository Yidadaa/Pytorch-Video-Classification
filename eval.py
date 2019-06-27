import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import pandas
import os
import argparse
import cv2

from dataloader import Dataset
from model import CNNEncoder, RNNDecoder
import config

def load_imgs_from_video(path: str)->list:
    """Extract images from video.

    Args:
        path(str): The path of video.

    Returns:
        A list of PIL Image.
    """
    video_fd = cv2.VideoCapture(path)
    video_fd.set(16, True)
    # flag 16: 'CV_CAP_PROP_CONVERT_RGB'
    # indicating the images should be converted to RGB.

    if not video_fd.isOpened():
        raise ValueError('Invalid path! which is: {}'.format(path))

    images = [] # type: list[Image]

    success, frame = video_fd.read()
    while success:
        images.append(Image.fromarray(frame))
        success, frame = video_fd.read()

    return images

def _eval(checkpoint: str, video_path: str, labels=[])->list:
    """Inference the model and return the labels.

    Args:
        checkpoint(str): The checkpoint where the model restore from.
        path(str): The path of videos.
        labels(list): Labels of videos.

    Returns:
        A list of labels of the videos.
    """
    if not os.path.exists(video_path):
        raise ValueError('Invalid path! which is: {}'.format(video_path))

    print('Loading model from {}'.format(checkpoint))
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Build model
    model = nn.Sequential(
        CNNEncoder(**config.cnn_encoder_params),
        RNNDecoder(**config.rnn_decoder_params)
    )
    model.to(device)
    model.eval()

    # Load model
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])
    print('Model has been loaded from {}'.format(checkpoint))

    label_map = [-1] * config.rnn_decoder_params['num_classes']
    # load label map
    if 'label_map' in ckpt:
        label_map = ckpt['label_map']

    # Do inference
    pred_labels = []
    video_names = os.listdir(video_path)
    with torch.no_grad():
        for video in tqdm(video_names, desc='Inferencing'):
            # read images from video
            images = load_imgs_from_video(os.path.join(video_path, video))
            # apply transform
            images = [Dataset.transform(None, img) for img in images]
            # stack to tensor, batch size = 1
            images = torch.stack(images, dim=0).unsqueeze(0)
            # do inference
            images = images.to(device)
            pred_y = model(images) # type: torch.Tensor
            pred_y = pred_y.argmax(dim=1).cpu().numpy().tolist()
            pred_labels.append([video, pred_y[0], label_map[pred_y[0]]])
            print(pred_labels[-1])

    if len(labels) > 0:
        acc = accuracy_score(pred_labels, labels)
        print('Accuracy: %0.2f' % acc)

    # Save results
    pandas.DataFrame(pred_labels).to_csv('result.csv', index=False)
    print('Results has been saved to {}'.format('result.csv'))

    return pred_labels

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 eval.py -i path/to/videos -r path/to/checkpoint')
    parser.add_argument('-i', '--video_path', help='path to videos')
    parser.add_argument('-r', '--checkpoint', help='path to the checkpoint')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    _eval(args.checkpoint, args.video_path)
