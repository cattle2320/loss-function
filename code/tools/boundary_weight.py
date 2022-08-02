import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

def boundary_weight(target, max_position, smooth=1e-6):
    target = target.numpy()
    max_position = max_position.numpy()
    Union = target + max_position

    if Union.any():
        posmask = Union.astype(np.bool)
        negmask = ~posmask
        # 计算标签内部的距离图
        posdis = distance_transform_edt(posmask)
        weight_1 = (np.max(posdis) - posdis + 1) * posmask

        # 计算标签外部的距离图
        negdis = distance_transform_edt(negmask)
        weight_2 = (np.max(negdis) - negdis + 1) * negmask

        final_weight = weight_1 / (np.max(weight_1) + smooth) + weight_2 / (np.max(weight_2) + smooth)
        return torch.from_numpy(final_weight)
    else:
        print('debug')