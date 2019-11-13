import os

import PIL.Image as Image
import numpy as np

__all__ = ['DatasetFolder']


class DatasetFolder(object):

    def __init__(self, root, split_dir, split_type, transform, out_name=False):
        assert split_type in ['train', 'test', 'val', 'query', 'repr']
        split_file = os.path.join(split_dir, split_type + '.csv')
        assert os.path.isfile(split_file)
        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']

        data, ori_labels = [x[0] for x in split], [x[1] for x in split]
        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]

        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels
        self.out_name = out_name
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert os.path.isfile(self.root+'/'+self.data[index])
        img = Image.open(self.root + '/' + self.data[index]).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            return img, label, self.data[index]
        else:
            return img, label
