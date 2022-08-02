import torch
import torch.nn as nn

def ss_loss(net_out, target, smooth=1e-6):
    # 计算标签真值之内的损失
    molecule_1 = (((target - net_out) ** 2) * target).sum()
    denominator_1 = target.sum()
    loss_1 = molecule_1 / (denominator_1 + smooth)
    # 计算真值之外的损失
    molecule_2 = (((target - net_out) ** 2) * (1 - target)).sum()
    denominator_2 = (1 - target).sum()
    loss_2 = molecule_2 / (denominator_2 + smooth)

    return 0.05 * loss_1 + (1-0.05) * loss_2


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
                    loss = ss_loss(net_out[j, i, ...], target[j, i, ...])
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