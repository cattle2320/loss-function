import torch


def one_hot_code(net_out, target):
    shp_x = net_out.shape
    shp_y = target.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            target = target.view(target.shape[0], 1, *target.shape[1:])
        if all([i == j for i, j in zip(net_out.shape, target.shape)]):
            one_hot = target
        else:
            idx = target.long()
            one_hot = torch.zeros(shp_x)
            if net_out.device.type == 'cuda':
                one_hot = one_hot.cuda(net_out.device.index)
            one_hot.scatter_(1, idx, 1)
    return one_hot


if __name__ == '__main__':
    net_out = torch.randn((1,5,5,5))
    target = torch.tensor(
       [[[0,0,0,0,0],
         [1,1,1,1,1],
         [2,2,3,2,2],
         [3,3,3,3,3],
         [4,4,4,4,4]]]
    )
    one_hot = one_hot_code(net_out, target)
    print(one_hot)
