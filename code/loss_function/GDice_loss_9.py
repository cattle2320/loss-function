import torch
import torch.nn as nn


def GDice_loss(net_out, target, smooth=1e-6):
    weight_c = 1 / (target.sum() ** 2)
    molecule = weight_c * ((target * net_out).sum())
    denominator = weight_c * (target.sum() + net_out.sum())
    acc = 2*molecule / denominator
    return 1-acc

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
                    loss = GDice_loss(net_out[j, i, ...], target[j, i, ...])
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