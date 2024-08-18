import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import os
from utils.util import crf_refine
import torchvision.utils as vutils


def test_all_case(net, image_list, num_classes=1, save_result=True,
                  test_save_path = None, trans_scale=416, GT_access=True):
    normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # SBU
    img_transform = transforms.Compose([
        transforms.Resize((trans_scale, trans_scale)),
        transforms.ToTensor(),
        normal,
    ])
    to_pil = transforms.ToPILImage()
    for (img_path, target_path) in tqdm(image_list):
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_var = img_transform(img).unsqueeze(0).cuda()
        mask_out, edge_out, cls_out  = net(img_var)
        res = torch.sigmoid(mask_out[-1])
        prediction = np.array(to_pil(res.data.squeeze(0).cpu()))
        prediction = crf_refine(np.array(img.convert('RGB').resize((trans_scale, trans_scale))), prediction)
        prediction = np.array(transforms.Resize((h, w))(Image.fromarray(prediction.astype('uint8')).convert('L')))
        if save_result:
            Image.fromarray(prediction).save(os.path.join(test_save_path, img_name[:-4] + '.png'), "PNG")
            print(os.path.join(test_save_path, img_name[:-4] + '.png'))

    return Jaccard