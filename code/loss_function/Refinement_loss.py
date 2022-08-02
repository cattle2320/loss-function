import torch
import torch.nn as nn


def overall_Mismatch(net_out, target, max_positiones, smooth=1e-6, alpha=3, beta=0.05):
    # 1、预测为1和标签为1代表重合的部分，说明其预测概率已达到标准，需要通过log或（标签-预测）的方法得到其损失
    # 像素位置 * 损失 * 单个像素权重
    molecule_1 = ((target * max_positiones) * -torch.log(net_out) * (
                (1 - net_out) ** alpha)).sum()              # 计算损失，针对单个像素点设置权重，使损失更关注预测概率小的像素点
    denominator_1 = (target * max_positiones).sum()         # 计算重合像素个数
    loss_1 = molecule_1 / (denominator_1 + smooth)

    # 2、预测为1和标签为0代表标签外部的元素，其预测概率值就是其损失，损失结果置于较大的权重
    molecule_2 = (((max_positiones - target) * max_positiones) * net_out * (net_out ** alpha)).sum()
    denominator_2 = ((max_positiones - target) * max_positiones).sum()
    loss_2 = molecule_2 / (denominator_2 + smooth)

    # 3、预测为0和标签为1代表标签内部的元素，其预测概率值是其精度，需要通过log或（标签-预测）的方法得到其损失
    molecule_3 = (((target - max_positiones) * target) * -torch.log(net_out) * ((1 - net_out) ** alpha)).sum()
    denominator_3 = ((target - max_positiones) * target).sum()
    loss_3 = molecule_3 / (denominator_3 + smooth)

    # 4、除了预测和标签并集的部分之外的那些像素
    molecule_4 = ((1 - (target + ((max_positiones - target) * max_positiones))) * net_out * (net_out ** alpha)).sum()
    denominator_4 = (1 - (target + ((max_positiones - target) * max_positiones))).sum()
    loss_4 = molecule_4 / (denominator_4 + smooth)

    loss = (loss_1 + loss_3) * beta + (loss_2 + loss_4) * (1 - beta)
    return loss

class new_loss(nn.Module):
    def __init__(self):
        super(new_loss, self).__init__()

    def forward(self, net_out, target, max_positiones, smooth=1e-6):
        img_numbers, num_classes, H, W = target.shape

        img_losses = torch.zeros(img_numbers)
        for j in range(net_out.shape[0]):

            losses = torch.zeros(num_classes)
            for i in range(0, net_out.shape[1]):
                max_target = torch.max(target[j, i, ...])
                max_position = torch.max(max_positiones[j, i, ...])

                if max_target == 0 and max_position == 0:
                    continue
                else:
                    loss_iou_c = overall_Mismatch(net_out[j, i, ...], target[j, i, ...], max_positiones[j, i, ...])
                    losses[i] = loss_iou_c

            img_losses[j] = losses.sum() / torch.count_nonzero(losses, dim=0)
        return img_losses.sum() / img_numbers


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
    classes = new_loss()
    loss = classes(net_out, target, max_netout)
    print(loss)