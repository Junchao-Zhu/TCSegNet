from medpy import metric
import os
from PIL import Image
import numpy as np
from pandas.core.frame import DataFrame


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.recall(pred, gt)
    asd = 1
    return dice, jc, hd, asd
