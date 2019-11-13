import numpy as np
import pickle as pkl
import cv2
import os
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to the data')
parser.add_argument('--split', type=str, help='path to the split folder')
args = parser.parse_args()


def save_imgs(prex1, prex2, tag):
    data_file = prex1 + tag + '_images_png.pkl'
    label_file = prex1 + tag + '_labels.pkl'
    with open(data_file, 'rb') as f:
        array = pkl.load(f)
    with open(label_file, 'rb') as f:
        labels = pkl.load(f)
    for idx, (img, glabel, slabel) in enumerate(
            tqdm.tqdm(zip(array, labels['label_general'], labels['label_specific']), total=len(array))):
        file_name = prex2 + tag + '_' + str(glabel) + '_' + str(slabel) + '_' + str(idx) + '.png'
        im = cv2.imdecode(img, 1)
        cv2.imwrite(file_name, im)
        # print('Finish ' + file_name)


if __name__ == '__main__':
    prex1 = args.data
    prex2 = prex1 + 'data/'
    if not os.path.isdir(prex2):
        os.makedirs(prex2)
    save_imgs(prex1, prex2, 'test')
    save_imgs(prex1, prex2, 'train')
    save_imgs(prex1, prex2, 'val')

    if not os.path.isdir(args.split):
        os.makedirs(args.split)
    data = os.listdir(prex2)
    with open(args.split + '/train.csv', 'w') as f:
        for name in data:
            if 'train' in name:
                label = name.split('.')[0].split('_')[-2]
                f.write('{},{}\n'.format(name, label))

    with open(args.split + '/val.csv', 'w') as f:
        for name in data:
            if 'val' in name:
                label = name.split('.')[0].split('_')[-2]
                f.write('{},{}\n'.format(name, label))

    with open(args.split + '/test.csv', 'w') as f:
        for name in data:
            if 'test' in name:
                label = name.split('.')[0].split('_')[-2]
                f.write('{},{}\n'.format(name, label))
