import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
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
parser.add_argument('--root_path', type=str, default='./train', help='Name of Experiment')
parser.add_argument('--max_iterations', type=int, default=45000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=6, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--edge', type=float, default='10', help='edge learning weight')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=7.0, help='consistency_rampup')
parser.add_argument('--scale', type=int, default=224, help='batch size of 8 with resolution of 416*416 is exactly OK')
parser.add_argument('--cls', type=float, default=1, help='classification loss weight')
parser.add_argument('--repeat', type=int, default=3, help='repeat')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


train_data_path = args.root_path
save_path = "./weight"
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
lr_decay = args.lr_decay
loss_record = 0

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


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


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


def value_noise(image, i_hsv, lam=0.5):
    B, C, H, W = image.shape
    tmp = torch.ones((B, 1, H, W)).cuda()
    tmp[:, 0, :, :] = tmp[:, 0, :, :] - i_hsv[:, 2, :, :]
    std = torch.std(tmp.view(B, 1, H * W), dim=2, keepdim=True).expand(B, 1, H * W).view(B, 1, H, W).cuda()
    mean = torch.zeros((B, 1, H, W)).cuda()
    noi = torch.zeros((B, C, H, W)).cuda()
    for i in range(C):
        noi[:, i, :, :] = (torch.normal(mean, std) * tmp * lam).squeeze(1)
    noi = torch.clamp(noi, -0.20, 0.20)

    ema_image = image + noi
    return ema_image


if __name__ == "__main__":
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = create_model()
    ema_model = create_model(ema=True)

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
                   target_transform=target_transform, mod='union', multi_task=True, edge=True, cls=True)

    labeled_idxs, unlabeled_idxs = relabel_dataset(db_train, edge_able=True)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(trainset, batch_sampler=batch_sampler, num_workers=0, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * base_lr},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': base_lr, 'weight_decay': 0.0001}
    ], momentum=0.9)

    consistency_criterion = losses.sigmoid_mse_loss
    ce = nn.CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    model.train()
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
            image_batch, label_batch, edge_batch, class_batch, hsv_batch = sampled_batch['image'], sampled_batch['label'], \
                                                                sampled_batch['edge'], sampled_batch['cls'], sampled_batch['img_hsv']
            image_batch, label_batch, edge_batch, class_batch, hsv_batch = image_batch.cuda(), label_batch.cuda(), edge_batch.cuda(), class_batch.cuda(), hsv_batch.cuda()

            ema_inputs = value_noise(image_batch, hsv_batch)

            mask_out, edge_out, cls_out = model(image_batch)

            with torch.no_grad():
                mask_out_ema, edge_out_ema, cls_out_ema = ema_model(ema_inputs)

            # calculate the loss
            # class loss
            cls_loss = []
            cls_con_loss = []

            cls_loss.append(2*ce(cls_out[0][:labeled_bs], class_batch[:labeled_bs]))
            cls_con_loss.append(2*ce(cls_out[0][labeled_bs:], class_batch[labeled_bs:]))
            cls_loss.append(ce(cls_out[-1][:labeled_bs], class_batch[:labeled_bs]))
            cls_con_loss.append(ce(cls_out[-1][labeled_bs:], class_batch[labeled_bs:]))

            cls_loss = sum(cls_loss)/(labeled_bs*2)
            cls_con_loss = sum(cls_con_loss)/((batch_size-labeled_bs)*2)

            # edge lossWW
            edge_loss = []
            edge_con_loss = []
            for (ix, ix_ema) in zip(edge_out, edge_out_ema):
                edge_loss.append(losses.bce2d_new(ix[:labeled_bs], edge_batch[:labeled_bs], reduction='mean'))
                edge_con_loss.append(consistency_criterion(ix[labeled_bs:], ix_ema[labeled_bs:]))
            edge_loss = sum(edge_loss)
            edge_con_loss = sum(edge_con_loss)

            # mask loss
            mask_loss1 = []
            mask_loss2 = []
            mask_con_loss1 = []

            for (ix, ix_ema) in zip(mask_out, mask_out_ema):
                mask_loss1.append(
                    F.binary_cross_entropy_with_logits(ix[:labeled_bs], label_batch[:labeled_bs], reduction='mean'))
                mask_con_loss1.append(consistency_criterion(ix[labeled_bs:], ix_ema[labeled_bs:]))

            # calculate the consistency loss of cnn and transformer
            mask_loss2.append(
                losses.dice_loss(torch.sigmoid(mask_out[0][:labeled_bs]), torch.sigmoid(mask_out[-1][:labeled_bs])))

            mask_loss = sum(mask_loss1) + sum(mask_loss2)
            supervised_loss = mask_loss + edge_loss * args.edge + cls_loss * args.cls

            consistency_weight = get_current_consistency_weight(epoch_num)
            consistency_loss = consistency_weight * (
                    edge_con_loss + sum(mask_con_loss1) + cls_con_loss)

            loss = supervised_loss + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            mask_loss1_record.update(mask_loss1[-1].item(), labeled_bs)
            edge_loss_record.update(edge_loss.item(), labeled_bs)
            mask_con_loss1_record.update(mask_con_loss1[-1].item(), batch_size - labeled_bs)
            edge_con_loss_record.update(edge_con_loss.item(), batch_size - labeled_bs)
            class_loss_record.update(cls_loss, labeled_bs)

            logging.info(
                'iteration %d : mask : %f5 , edge: %f5 , cls: %f5, mask_f_con: %f5  edge_con: %f5 loss_weight: %f5, lr: %f5' %
                (iter_num, mask_loss1_record.avg, edge_loss_record.avg, class_loss_record.avg,
                 mask_con_loss1_record.avg, edge_con_loss_record.avg, consistency_weight,
                 optimizer.param_groups[1]['lr']))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(save_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
