import torch
import pandas
import argparse

from dataloader import Dataset
import config

def merge_labels_to_ckpt(ck_path:str, train_file:str):
    '''Merge labels to a checkpoint file.

    Args:
        ck_path(str): path to checkpoint file
        train_file(str): path to train set index file, eg. train.csv

    Return:
        This function will create a {ck_path}_patched.pth file.
    '''
    # load model
    print('Loading checkpoint')
    ckpt = torch.load(ck_path)

    # load train files
    print('Loading dataset')
    raw_data = pandas.read_csv(train_file)
    train_set = Dataset(raw_data.to_numpy())

    # patch file name
    print('Patching')
    patch_path = ck_path.replace('.pth', '') + '_patched.pth'

    ck_dict = { 'label_map': train_set.labels }
    names = ['epoch', 'model_state_dict', 'optimizer_state_dict']
    for name in names:
        ck_dict[name] = ckpt[name]

    torch.save(ck_dict, patch_path)
    print('Patched checkpoint has been saved to {}'.format(patch_path))

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 tools.py -i path/to/train.csv -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your dataset index file')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    merge_labels_to_ckpt(args.restore_from, args.data_path)
