""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman

Modified by YANG Ruixin for multi-label classification
2021/03/18
https://github.com/yang-ruixin
yang_ruixin@126.com (in China)
rxn.yang@gmail.com (out of China)
"""

# ================================
import csv
import numpy as np
# ================================

import torch.utils.data as data
import os
import torch
import logging
import cv2
from PIL import Image

from .parsers import create_parser

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training, batch_size=batch_size)
        else:
            self.parser = parser
        self.transform = transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if target is None:
                target = torch.tensor(-1, dtype=torch.long)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)


# ================================
class DatasetAttributes:
    """
    Get all the possible labels
    """
    def __init__(self, annotation_path,label):
        self.d = {}
        for i in range(0,len(label)):
            self.d['{}_lables'.format(label[i])] = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for i in range(0, len(label)):
                    self.d['{}_lables'.format(label[i])].append(row['{}'.format(label[i])])

        for i in range(0, len(label)):
            self.d['{}_lables'.format(label[i])] = np.unique(self.d['{}_lables'.format(label[i])])

        for i in range(0, len(label)):
            self.d['num_{}'.format(label[i])] = len(self.d['{}_lables'.format(label[i])])

        for i in range(0, len(label)):
            self.d['{}_id_to_name'.format(label[i])] = dict(zip(range(len(self.d['{}_lables'.format(label[i])])), self.d['{}_lables'.format(label[i])]))

        for i in range(0, len(label)):
            self.d['{}_name_to_id'.format(label[i])] = dict(zip(self.d['{}_lables'.format(label[i])], range(len(self.d['{}_lables'.format(label[i])]))))



class DatasetML(data.Dataset):
    def __init__(
            self,
            annotation_path,
            attributes,label,
            transform=None):

        super().__init__()

        self.transform = transform
        self.attr = attributes
        self.label = label
        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.d = {}
        for i in range(0, len(label)):
            self.d['{}_lables'.format(label[i])] = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                print(annotation_path,row)
                for i in range(0, len(label)):
                    self.d['{}_lables'.format(label[i])].append(self.attr.d['{}_name_to_id'.format(label[i])][row['{}'.format(label[i])]])

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # read image
        #img = Image.open(img_path)
        img = cv2.imread(img_path)
        if img is None:
            img = cv2.imread(self.data[idx-1])
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)
        labels = {}
        for i in range(0, len(self.label)):
            labels['{}_labels'.format(self.label[i])] = self.d['{}_lables'.format(self.label[i])][idx]

        return img, labels

    def __len__(self):
        # return len(self.samples)
        return len(self.data)
# ================================
