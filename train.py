import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import math
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import torch.nn as nn
from networks.TCSeg import build_model
from utils import ramps, losses
from dataloaders.tcdata import TCdata, relabel_dataset
from dataloaders import joint_transforms_edge as joint_transforms
from utils.util import AverageMeter, TwoStreamBatchSampler


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./train',
                    help='Name of Experiment')
parser.add_argument('--max_iterations', type=int, default=45000, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float, default=0.005, help='maximum epoch number to train')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--edge', type=float, default='10', help='edge learning weight')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--scale', type=int, default=224, help='batch size of 8 with resolution of 416*416 is exactly OK')
parser.add_argument('--cls', type=float, default=1, help='classification loss weight')
parser.add_argument('--repeat', type=int, default=3, help='repeat')
args = parser.parse_args()

train_data_path = args.root_path
save_path = "./weight"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
max_iterations = args.max_iterations
base_lr = args.base_lr
lr_decay = args.lr_decay
loss_record = 0
batch_size = 8

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
else:
    cudnn.benchmark = True

num_classes = 2


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)
        s = math.ceil(s / 2)
        s = math.ceil(s / 2)
        s = math.ceil(s / 2)
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))

        result = result.permute(0, 3, 1, 2)

        return result


def create_model(ema=False):
    if ema:
        net = build_model(ema=True)
        net_cuda = net.cuda()
        for param in net_cuda.parameters():
            param.detach_()
    else:
        net = build_model()
        net_cuda = net.cuda()

    return net_cuda


if __name__ == "__main__":
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = create_model()

    joint_transform = joint_transforms.Compose([
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomVerticalFlip(),
        joint_transforms.RandomRotate(),
        joint_transforms.Resize((args.scale, args.scale))
    ])

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()

    trainset = TCdata(root=train_data_path, joint_transform=joint_transform, transform=img_transform,
                   target_transform=target_transform, mod='labeled', multi_task=True, edge=True, cls=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(trainset, batch_size=8, num_workers=0, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * base_lr},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': base_lr, 'weight_decay': 0.0001}
    ], momentum=0.9)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    model.train()
    ce = nn.CrossEntropyLoss()

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        mask_loss1_record, mask_con_loss1_record, edge_loss_record, edge_con_loss_record = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        class_loss_record = AverageMeter()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            optimizer.param_groups[0]['lr'] = 2 * base_lr * (1 - float(iter_num) / max_iterations
                                                             ) ** lr_decay
            optimizer.param_groups[1]['lr'] = base_lr * (1 - float(iter_num) / max_iterations
                                                         ) ** lr_decay
            image_batch, label_batch, edge_batch, class_batch, hsv_batch = sampled_batch['image'], sampled_batch[
                'label'], \
                                                                           sampled_batch['edge'], sampled_batch['cls'], \
                                                                           sampled_batch['img_hsv']
            image_batch, label_batch, edge_batch, class_batch, hsv_batch = image_batch.cuda(), label_batch.cuda(), edge_batch.cuda(), class_batch.cuda(), hsv_batch.cuda()

            ema_inputs = value_noise(image_batch, hsv_batch)

            mask_out, edge_out, cls_out = model(image_batch)

            # calculate the loss
            # class loss
            cls_loss = [2 * ce(cls_out[0], class_batch), ce(cls_out[-1], class_batch)]
            cls_loss = sum(cls_loss)/batch_size

            # edge loss
            edge_loss = []
            for ix in up_edge:
                edge_loss.append(losses.bce2d_new(ix, edge_batch, reduction='mean'))
            edge_loss = sum(edge_loss)
            mask_loss1 = []
            mask_loss2 = []
            for ix in mask_out:
                mask_loss1.append(
                    F.binary_cross_entropy_with_logits(ix,  label_batch, reduction='mean'))

            # calculate the consistency loss of cnn and transformer
            mask_loss2.append(
                losses.dice_loss(torch.sigmoid(mask_out[0]), torch.sigmoid(mask_out[-1])))

            mask_loss = sum(mask_loss1) + sum(mask_loss2)
            supervised_loss = mask_loss + edge_loss * args.edge + cls_loss * args.cls

            loss = supervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            # loss_all_record.update(loss.item(), batch_size)
            mask_loss1_record.update(mask_loss1[-1].item(), batch_size)
            edge_loss_record.update(edge_loss.item(), batch_size)
            class_loss_record.update(cls_loss, batch_size)

            logging.info(
                'iteration %d : mask : %f5 , edge: %f5 , cls: %f5, lr: %f5' %
                (iter_num, mask_loss1_record.avg, edge_loss_record.avg, class_loss_record.avg,
                 optimizer.param_groups[1]['lr']))
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(save_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
