import random

from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, edge):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask, edge = t(img, mask, edge)
            img_hsv = img.convert("HSV")
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





