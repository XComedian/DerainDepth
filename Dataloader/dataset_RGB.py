import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def make_dataset(root, is_train):
    if is_train:
        data_path = os.path.join(root, 'train')
        input_path = os.path.join(data_path, 'input')
        target_path = os.path.join(data_path, 'target')
        input_lst = [os.path.join(input_path, x) for x in os.listdir(input_path) if is_image_file(x)]
        target_lst = [os.path.join(target_path, x) for x in os.listdir(target_path) if is_image_file(x)]
        input_lst.sort()
        target_lst.sort()
        return list(zip(input_lst, target_lst))
    else:
        data_path = os.path.join(root, 'test')
        input_path = os.path.join(data_path, 'input')
        target_path = os.path.join(data_path, 'target')
        input_lst = [os.path.join(input_path, x) for x in os.listdir(input_path) if is_image_file(x)]
        target_lst = [os.path.join(target_path, x) for x in os.listdir(target_path) if is_image_file(x)]
        input_lst.sort()
        target_lst.sort()
        return list(zip(input_lst, target_lst))


class Dataset_RGB(Dataset):
    def __init__(self, root, is_train, transform=None, triple_transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.triple_transform = triple_transform
        self.data = make_dataset(root, is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_path, target_path = self.data[index]
        input_img = Image.open(input_path)
        target_img = Image.open(target_path)

        if self.triple_transform:
            input_img, target_img = self.triple_transform(input_img, target_img)
            
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img
    

if __name__ == '__main__':
    root = '/home/disk/ning/dataset/Rain200L'
    dataset = Dataset_RGB(root, is_train=True)
    input_img, target_img = dataset[0]
    print(input_img.size, target_img.size)


