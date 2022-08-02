import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_dtm(img_gt, out_shape):
    fg_dtm = np.zeros(out_shape)
    posmask = img_gt.astype(np.bool)
    if posmask.any():
        posdis = distance_transform_edt(posmask)
        fg_dtm = posdis
    return torch.from_numpy(fg_dtm)


def hausdorff_loss(net_out, target, smooth=1e-6):
    net_out_distance = compute_dtm(net_out.detach().cpu().numpy() > 0.5, net_out.shape)
    target_distance = compute_dtm(target.cpu().numpy(), target.shape)

    if net_out_distance.device != net_out.device:
        net_out_distance = net_out_distance.to(net_out.device).type(torch.float32)
    if target_distance.device != net_out.device:
        target_distance = target_distance.to(net_out.device).type(torch.float32)

    loss_c = (net_out - target) ** 2  # 计算标签和预测的差值大小
    distance_weight = net_out_distance ** 2 + target_distance ** 2
    pixels = net_out.shape[0] * net_out.shape[1]
    loss = (loss_c * distance_weight).sum() / pixels
    return loss


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
                elif max_target == 1:
                    loss = hausdorff_loss(net_out[j, i, ...], target[j, i, ...])
                    losses[i] = loss
            img_losses[j] = losses.sum() / torch.count_nonzero(losses, dim=0)
        return img_losses.sum() / img_number


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