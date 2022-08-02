import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import segmentation as skimage_seg

def compute_sdf(target):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    target = target.astype(np.uint8)

    posmask = target.astype(np.bool)
    negmask = ~posmask
    posdis = distance_transform_edt(posmask)
    negdis = distance_transform_edt(negmask)
    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
    get = negdis - posdis
    get[boundary == 1] = 0

    return torch.from_numpy(get)

def boundary_loss(net_out, target, smooth=1e-6):
    weight_map = compute_sdf(target.cpu().numpy())

    if weight_map.device != net_out.device:
        weight_map = weight_map.to(net_out.device).type(torch.float32)

    loss = weight_map * net_out
    return loss.mean()

class Mismatch_loss(nn.Module):
    def __init__(self):
        super(Mismatch_loss, self).__init__()

    def forward(self, net_out, target, max_positiones):
        num_class = target.shape[1]
        img_number = target.shape[0]

        img_losses = torch.zeros(img_number)
        for j in range(net_out.shape[0]):

            losses = torch.zeros(num_class)
            for i in range(0, net_out.shape[1]):
                max_target = torch.max(target[j, i, ...])
                max_position = torch.max(max_positiones[j, i, ...])
                if max_target == 0 and max_position == 0:
                    continue
                else:
                    loss = boundary_loss(net_out[j, i, ...], target[j, i, ...])
                    losses[i] = loss
            img_losses[j] = losses.sum() / torch.count_nonzero(losses, dim=0)
        return img_losses.sum()/img_number


if __name__ == '__main__':
    net_out = torch.tensor(
      [[[[0.3, 0.3, 0.3, 0.9, 0.3, 0.3, 0.3, 0.3],
         [0.3, 0.9, 0.9, 0.9, 0.9, 0.3, 0.9, 0.9],
         [0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
         [0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
         [0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
         [0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
         [0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
         [0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]]]
    )
    # max_netout是根据net_out进行softmax得来的
    max_netout = torch.tensor(
      [[[[0, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 0, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1]]]]
    )
    target = torch.tensor(
        [[[[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1]]]]
    )
    classes = Mismatch_loss()
    loss = classes(net_out, target, max_netout)
    print(loss)