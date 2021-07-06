"""
人员数据集
"""
import cv2
import torch.utils.data as data
import os
import torch
import logging
import numpy as np

from PIL import Image
from torchvision import transforms

from .parsers import create_parser
from .parsers.parser_human_image import ParserHumanImage

_logger = logging.getLogger(__name__)

_ERROR_RETRY = 50


def transform_heatmap(mask_data_ori, width=224, height=224):
    mask_data = (mask_data_ori*255).astype(int)
    h = mask_data.shape[1]
    w = mask_data.shape[2]
    kp_num = mask_data.shape[0]
    zeroh = np.zeros((kp_num, h, width - w), int)
    arr_t1 = np.concatenate((mask_data, zeroh), axis=2)
    zerow = np.zeros((kp_num, height - h, width), int)
    arr_t2 = np.concatenate((arr_t1, zerow), axis=1)
    return arr_t2

def transform(mask_data_ori, width=224, height=224):
    data = mask_data_ori.transpose(1, 2, 0)
    data = cv2.resize(data, (width, height))
    data = data.transpose(2, 0, 1)
    return data.astype(np.float32)

class HumanImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            is_training=False,
            parser=None,
    ):
        if parser is None or isinstance(parser, str):
            self.parser = ParserHumanImage(root=root, is_training=is_training)
        else:
            self.parser = parser
        self.transform = transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img_path, target = self.parser[index]
        try:
            img = np.load(img_path)
            if img is None:
                print("img None:{}".format(img_path))
            if self.transform is not None:
                img = self.transform(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {img_path}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if target is None:
            target = torch.tensor(0, dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)
