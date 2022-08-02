import torch
import numpy as np

def normalization_torch(net_out):
    max = torch.max(net_out)
    min = torch.min(net_out)

    result = (net_out - min) / (max - min)
    return result

def normalization_list(net_out):
    net_out = np.array(net_out)

    if net_out.size > 0:
        max_ = np.max(net_out)
        min_ = np.min(net_out)
        result = (net_out - min_) / (max_ - min_)
    else:
        result = net_out
    return result

if __name__ == '__main__':
    ce = []
    result = normalization_list(ce)
    print(result)