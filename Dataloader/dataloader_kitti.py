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
        data_path = os.path.join(root, 'training')
        input_path = os.path.join(data_path, 'image_2_3_rain50')
        target_path = os.path.join(data_path, 'image_2')
        depth_path = os.path.join(data_path, 'depth_2')
        # only use image_2 images

        input_lst = [os.path.join(input_path, x) for x in os.listdir(input_path) if is_image_file(x) and x.endswith('2_50.jpg')]
        target_lst = [os.path.join(target_path, x) for x in os.listdir(target_path) if is_image_file(x)]
        depth_lst = [os.path.join(depth_path, x) for x in os.listdir(depth_path) if is_image_file(x)]
        input_lst.sort()
        target_lst.sort()
        depth_lst.sort()
        return list(zip(input_lst, target_lst, depth_lst))
    else:
        data_path = os.path.join(root, 'testing')
        input_path = os.path.join(data_path, 'image_2_3_rain50')
        target_path = os.path.join(data_path, 'image_2')
        depth_path = os.path.join(data_path, 'depth_2')
        # only use image_2 images

        input_lst = [os.path.join(input_path, x) for x in os.listdir(input_path) if is_image_file(x) and x.endswith('2_50.jpg')]
        target_lst = [os.path.join(target_path, x) for x in os.listdir(target_path) if is_image_file(x)]
        depth_lst = [os.path.join(depth_path, x) for x in os.listdir(depth_path) if is_image_file(x)]
        input_lst.sort()
        target_lst.sort()
        depth_lst.sort()
        return list(zip(input_lst, target_lst, depth_lst))


class Dataset_KITTI(Dataset):
    def __init__(self, root, is_train, transform=None, triple_transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.triple_transform = triple_transform
        self.data = make_dataset(root, is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_path, target_path, depth_path = self.data[index]
        input_img = Image.open(input_path)
        target_img = Image.open(target_path)
        depth_img = Image.open(depth_path)

        if self.triple_transform:
            input_img, target_img, depth_img = self.triple_transform(input_img, target_img, depth_img)
            
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            depth_img = self.transform(depth_img)

        return input_img, target_img, depth_img
    

if __name__ == '__main__':
    root = '/home/disk/ning/dataset/StereoRain_dataset/k12'
    dataset = Dataset_KITTI(root, is_train=True)
    input_img, target_img, depth_img = dataset[0]
    print(input_img.size, target_img.size, depth_img.size)


