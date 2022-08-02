import torch
import torch.nn as nn

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_loss(net_out_c, target_c, smooth=1e-6):
    net_out_c = torch.flatten(net_out_c)
    target_c = torch.flatten(target_c)

    loss_c = (target_c - net_out_c).abs()  # 计算标签和预测的差值大小
    loss_c_sorted, loss_index = torch.sort(input=loss_c, dim=0, descending=True)  # 越靠近前面的标签 表示这个像素点与真值的误差越大
    target_c_sorted = target_c[loss_index]
    loss = torch.dot(loss_c_sorted, lovasz_grad(target_c_sorted))
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
                else:
                    loss = lovasz_loss(net_out[j, i, ...], target[j, i, ...])
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