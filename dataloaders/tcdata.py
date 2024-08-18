import os
import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image
import torch

NO_LABEL = -1


def make_union_dataset(root, edge=False):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'image')) if f.endswith('.png')]
    label_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'mask')) if f.endswith('.png')]
    data_list = []
    if edge:
        edge_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'edge')) if f.endswith('.png')]
        for img_name in img_list:
            if img_name in label_list:
                data_list.append((os.path.join(root, 'image', img_name + '.png'),
                                  os.path.join(root, 'mask', img_name + '.png'),
                                  os.path.join(root, 'edge', img_name + '.png')))
            else:
                data_list.append((os.path.join(root, 'image', img_name + '.png'),
                                  -1, -1
                                  ))
    else:
        for img_name in img_list:
            if img_name in label_list:
                data_list.append((os.path.join(root, 'image', img_name + '.png'),
                                  os.path.join(root, 'mask', img_name + '.png')
                                  ))
            else:
                data_list.append((os.path.join(root, 'image', img_name + '.png'),
                                  -1
                                  ))

    return data_list


def make_labeled_dataset(root, edge=False):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'image')) if f.endswith('.png')]
    label_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'mask')) if f.endswith('.png')]
    data_list = []
    if edge:
        # for img_name in img_list:
        for img_name in label_list:
            data_list.append((os.path.join(root, 'image', img_name + '.png'),
                              os.path.join(root, 'mask', img_name + '.png'),
                              os.path.join(root, 'edge', img_name + '.png')
                              ))
    else:
        # for img_name in img_list:
        for img_name in label_list:
            data_list.append((os.path.join(root, 'image', img_name + '.png'),
                              os.path.join(root, 'mask', img_name + '.png')))
    return data_list


class TCdata(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None, mod='union', cls=False,
                  edge=False, multi_task=False):
        assert (mod in ['union', 'labeled'])
        self.root = root
        self.mod = mod
        if self.mod == 'union':
            self.imgs = make_union_dataset(root, edge)
        else:
            self.imgs = make_labeled_dataset(root, edge)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.edge = edge
        self.cls = cls
        self.multi_task = multi_task

    def __getitem__(self, index):
        img_path, gt_path, edge_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img_hsv = img.convert('HSV')

        # get cls tag
        n = -1
        i_name = img_path.split("/")[-1]
        for d in os.listdir('./train/img'):
            if i_name in os.listdir(os.path.join('./train/img', d)):
                n = d
        l = torch.zeros(7)
        l[int(n) - 1] = 1
        cls = l

        if gt_path != -1:
            target = Image.open(gt_path).convert('L')
            if self.joint_transform is not None:
                edge = Image.open(edge_path).convert('L')
                img, target, edge, img_hsv = self.joint_transform(img, target, edge)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                img_hsv = self.transform(img_hsv)
                target = self.target_transform(target)
                edge = self.target_transform(edge)
            if self.cls and self.edge:
                sample = {'image': img, 'label': target, 'cls': cls, 'edge': edge, 'img_hsv': img_hsv}
        return sample

    def __len__(self):
        return len(self.imgs)


def relabel_dataset(dataset, edge_able=False):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        if not edge_able:
            path, label = dataset.imgs[idx]
        else:
            path, label, edge = dataset.imgs[idx]
        if label == -1:
            unlabeled_idxs.append(idx)
    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs


