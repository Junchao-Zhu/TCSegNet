import os
import argparse
import torch
from test_util import test_all_case
from networks.TCSeg import build_model
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default="./test")
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--base_lr', type=float,  default=0.005, help='base learning rate')
parser.add_argument('--edge', type=float, default='10', help='edge learning weight')
parser.add_argument('--epoch_name', type=str,  default='iter_15000.pth', help='choose one epoch/iter as pretained')
parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
parser.add_argument('--scale', type=int,  default=224, help='batch size of 8 with resolution of 416*416 is exactly OK')
parser.add_argument('--cls', type=float,  default=5.0, help='cls loss weight')
parser.add_argument('--repeat', type=int,  default=6, help='repeat')


FLAGS = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
save_path = './weight/iter_15000.pth'
test_save_path = './prediction'
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(snapshot_path)
num_classes = 1


img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(FLAGS.root_path, 'image')) if f.endswith('.png')]
data_path = [(os.path.join(FLAGS.root_path, 'image', img_name + '.png'),
             '****')
            for img_name in img_list]


def test_calculate_metric():
    net = build_model().cuda()
    net.load_state_dict(torch.load(snapshot_path))
    print("init weight from {}".format(snapshot_path))
    net.eval()

    avg_metric = test_all_case(net, data_path, num_classes=num_classes,
                               trans_scale=FLAGS.scale)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print('Test ber results: {}'.format(metric))
