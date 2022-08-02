import numpy as np

ce_1 = [0.96363489, 0.93209466, 0.66799972, 0.98584963, 0.9238924, 0.78409156, 0.86761879, 0.43679296, 0.94970635, 0.78774783, 0.53381493, 0.73308713]
def list_mean(data):
    result = np.nanmean(data)
    return result


if __name__ == '__main__':
    e = list_mean(np.array(ce_1))
    print(e)
