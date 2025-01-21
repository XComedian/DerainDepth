import os
import re
import torch.utils.data as data
from PIL import Image


def extract_common_identifier(file_path):
    base_name = os.path.basename(file_path)
    parts = base_name.split('_')
    common_identifier = '_'.join(parts[:3])
    return common_identifier


def filter_data_list(data_list):
    filtered_list = []
    for sublist in data_list:
        common_identifier = extract_common_identifier(sublist[0])
        if all(common_identifier in path for path in sublist):
            filtered_list.append(sublist)
    return filtered_list


def make_dataset(root, is_train):
    if is_train:

        input = open(os.path.join(root, 'txt/train_input.txt'))
        ground_t = open(os.path.join(root, 'txt/train_gt.txt'))
        depth_t = open(os.path.join(root, 'txt/train_depth.txt'))
        input_list = [x.strip('\n') for x in input]
        input_list = [x.split('_')[0] + '/' + x for x in input_list] 
        # print(input_list)
        input_list = [re.sub(r'(_rain)', r'_leftImg8bit\1', x) for x in input_list]
        image = [(os.path.join(root, 'rain', img_name)) for img_name in
                 input_list]
        gt = [(os.path.join(root, 'image', img_name.strip('\n'))) for img_name in
                 ground_t]
        depth = [(os.path.join(root, 'depth', img_name.strip('\n'))) for img_name in
              depth_t]

        input.close()
        ground_t.close()
        depth_t.close()

        pairs = [[image[i], gt[i], depth[i]]for i in range(len(image))]
        # filtered_pairs = filter_data_list(pairs)
        # return filtered_pairs
        return pairs

    else:

        input = open(os.path.join(root, 'txt/test_input.txt'))
        ground_t = open(os.path.join(root, 'txt/test_gt.txt'))
        depth_t = open(os.path.join(root, 'txt/test_depth.txt'))
        input_list = [x.strip('\n') for x in input]
        input_list = [x.split('_')[0] + '/' + x for x in input_list] 
        input_list = [re.sub(r'(_rain)', r'_leftImg8bit\1', x) for x in input_list]

        image = [(os.path.join(root, 'rain', img_name)) for img_name in
                 input_list]
        gt = [(os.path.join(root, 'image', img_name.strip('\n'))) for img_name in
              ground_t]
        depth = [(os.path.join(root, 'depth', img_name.strip('\n'))) for img_name in
                 depth_t]

        input.close()
        ground_t.close()
        depth_t.close()

        pairs = [[image[i], gt[i], depth[i]]for i in range(len(image))]
        # filtered_pairs = filter_data_list(pairs)
        return pairs



class ImageFolder(data.Dataset):
    def __init__(self, root, triple_transform=None, transform=None, target_transform=None, is_train=True):
        self.root = root
        self.imgs = make_dataset(root, is_train)
        self.triple_transform = triple_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, depth_path = self.imgs[index]
        img = Image.open(img_path)
        target = Image.open(gt_path)
        depth = Image.open(depth_path)
        if self.triple_transform is not None:
            img, target, depth = self.triple_transform(img, target, depth)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            depth = self.target_transform(depth)

        return img, target, depth

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    data_path = '/home/disk/ning/DAFNet/data'
    dataset = ImageFolder(data_path, is_train=True)
    print(len(dataset))
    