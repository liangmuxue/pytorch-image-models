""" A dataset parser that reads human images from folders
"""
import os
import numpy as np
from timm.utils.misc import natural_key

from .parser import Parser
from .class_map import load_class_map
from .constants import IMG_EXTENSIONS


def find_human_images_and_targets(folder, label_file, is_training=False,
                                  sort=True):
    images_and_targets = []
    if is_training:
        root = folder + "/train"
    else:
        root = folder + "/test"
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            items = line.split(" ")
            print('items {}'.format(items))
            kp_file = items[0]
            hege = int(items[1])
            # buhege = int(items[2])
            # back = int(items[7])
            if hege == 1:
                label = 0
            if hege == 0:
                label = 1
            # if back == 1:
            #     label = 3
            images_and_targets.append([kp_file, label])
    class_to_idx = {"hege":0,"buhege":1}
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


class ParserHumanImage(Parser):

    def __init__(
            self,
            root,
            is_training=False):
        super().__init__()

        self.root = root
        if is_training:
            label_file = "datasets/face_key_train.txt"
        else:
            label_file = "datasets/face_key_val.txt"
        self.samples, self.class_to_idx = find_human_images_and_targets(root, label_file,is_training=is_training)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return path, target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
