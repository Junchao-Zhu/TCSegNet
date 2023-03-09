import random

from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, edge):
        assert img.size == mask.size
        # assert img.size == edge.size
        for t in self.transforms:
            img, mask, edge = t(img, mask, edge)
        img_hsv = img.convert('HSV')
        return img, mask, edge, img_hsv


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, edge):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), edge.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, edge


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, edge):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), edge.resize(self.size, Image.NEAREST)


class RandomVerticalFlip(object):
    def __call__(self, img, mask, edge):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM), edge.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask, edge


class RandomRotate(object):
    def __call__(self, img, mask, edge):
        if random.random() < 0.3:
            return img.transpose(Image.ROTATE_90), mask.transpose(Image.ROTATE_90), edge.transpose(
                Image.ROTATE_90)
        elif 0.3 < random.random() < 0.6:
            return img.transpose(Image.ROTATE_180), mask.transpose(Image.ROTATE_180), edge.transpose(
                Image.ROTATE_180)
        else:
            return img.transpose(Image.ROTATE_270), mask.transpose(Image.ROTATE_270), edge.transpose(
                Image.ROTATE_270)