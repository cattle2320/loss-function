import numpy as np
from tools.list_mean import list_mean
import matplotlib.pyplot as plt
import pandas as pd
import cfg

ce_1=[0.9631122 ,0.93132433,0.03257432,0.98530444,0.78355266,0.78282429,0.32904416,0.28523679,0.83718107,0.19937046,0.11121822,0.60721964]
wce1_2= [0.90331695,0.68865285,0.51010367,0.89781251,0.82396862,0.68299483,0.80816805,0.37870086,0.86003906,0.73770385,0.5080597 ,0.53324385]
topk_2=[0.96307039,0.9318538 ,0.03713474,0.98584963,0.78844684,0.78409156,0.3332936 ,0.2960551 ,0.8338218 ,0.21505317,0.13250346,0.61721501]
focal_3=[0.96339122,0.93124776,0.06233925,0.98488746,0.79345198,0.78354804,0.37669333,0.30202131,0.840442  ,0.24367655,0.13060171,0.62944208]
DPCE_4=[0.96363489,0.93209466,0.03186617,0.98572794,0.77136266,0.78393764,0.28685341,0.27728461,0.82701729,0.19122394,0.09913549,0.60491281]
dice_5=[0.94655924,0.90789301,0.1376738 ,0.9778201 ,0.812423  ,0.7492831,0.41114682,0.36060609,0.83911579,0.49994983,0.35492301,0.54556676]
lovasz_6=[0.93451675,0.87407534,0.4458763 ,0.97256535,0.83384465,0.7502171,0.75924603,0.        ,0.88304434,0.71091777,0.        ,0.64871354]
tversky_7=[0.93721811,0.89745082,0.1086354 ,0.96901848,0.85682168,0.76700752,0.50333579,0.38137256,0.87743164,0.44242994,0.31004825,0.6241242 ]
GDice_8=[0.93923134,0.90345168,0.14831916,0.97758946,0.8071966 ,0.75038775,0.42969133,0.34968466,0.84432166,0.48239087,0.31664333,0.57403965]
asym_9=[0.94830067,0.89055059,0.11702674,0.96897886,0.84893249,0.75699036,0.44674418,0.38687575,0.87043608,0.56427619,0.40919786,0.58708098]
PGDice_10=[]
boundary_11=[]
hausdorff_12 = []
sensitivity_specificity_loss13 = [0.9427661 ,0.88066108,0.59300955,0.96880376,0.87692063,0.76833879,0.86761879,0.43679296,0.91878996,0.78774783,0.53381493,0.68728877]
focal_tversky_loss14 = [0.956307  ,0.90645344,0.        ,0.97347835,0.85477347,0.76398649,0.46439755,0.33762223,0.88008982,0.53499008,0.34080639,0.54797416]
overall_Mismatch15 = [0.95190618,0.90886937,0.66799972,0.97516483,0.9238924 ,0.77786385,0.        ,0.        ,0.94970635,0.        ,0.        ,0.73308713]

def nanmean(list):
    for i, data in enumerate(list):
        print('{}------------{}'.format(list_name[i], list_mean(data)))

def Transpose(list):
    # 数组转置
    data_all = np.zeros(shape=(len(list), len(list[0])))
    for i, data in enumerate(list):
        if len(data) !=0:
            data_all[i] = data
    num = np.array(data_all)
    zhuanzhi = num.transpose()
    return zhuanzhi

def plot(x,y,linestyle, label):
    plt.plot(x, y, linestyle, label=label)
    plt.legend()        # 显示上面的label

def ply(data, name, style):
    # x_axis_data = [i for i in range(len(data[0]))]  # x
    x_axis_data = name

    plot(x=x_axis_data, y=data[0], linestyle=style, label='ce')
    plot(x=x_axis_data, y=data[1], linestyle=style, label='topk loss')
    plot(x=x_axis_data, y=data[2], linestyle=style, label='focal loss')
    plot(x=x_axis_data, y=data[3], linestyle=style, label='DPCE loss')
    plot(x=x_axis_data, y=data[4], linestyle=style, label='Dice loss')
    plot(x=x_axis_data, y=data[5], linestyle=style, label='lovasz loss')
    plot(x=x_axis_data, y=data[6], linestyle=style, label='tversky loss')
    plot(x=x_axis_data, y=data[7], linestyle=style, label='GDice loss')
    plot(x=x_axis_data, y=data[8], linestyle=style, label='Asym loss')
    # plot(x=x_axis_data, y=data[9], linestyle=style, label='PGDice loss')
    # plot(x=x_axis_data, y=data[10], linestyle=style, label='Boundary loss')
    # plot(x=x_axis_data, y=data[11], linestyle=style, label='Hausdorff loss')
    plot(x=x_axis_data, y=data[12], linestyle=style, label='sensitivity specificity loss')
    plot(x=x_axis_data, y=data[13], linestyle=style, label='focal tversky loss')
    plot(x=x_axis_data, y=data[14], linestyle=style, label='overall Mismatch')
    # plot(x=x_axis_data, y=data[15], linestyle=style, label='wce loss')

    plt.xlabel('category')               # x_label
    plt.ylabel('acc')                    # y_label

    plt.xticks(ticks=x_axis_data)
    # plt.yticks(ticks=[i for i in range(0, max)])      # 不显示所有y刻度标签

    plt.grid(True)                  # 显示网格线

    # plt.ylim(-1,1)                #仅设置y轴坐标范围
    plt.show()

def cfg_name(file_path):
    pd_label_color = pd.read_csv(file_path, sep=',')
    colormap = []
    for i in range(len(pd_label_color.index)):
        tmp = pd_label_color.iloc[i]
        name = tmp['name']
        colormap.append(name)
    return colormap

if __name__ == '__main__':
    list_name = ['ce_1', "topk_2", "focal_3", "DPCE_4", "dice_5", "lovasz_6", "tversky_7", "GDice_8", "asym_9", "PGDice_10", "boundary_11",
        "hausdorff_12", "sensitivity_specificity_loss13", "focal_tversky_loss14", "overall_Mismatch15", 'wce1_2']

    list = [ce_1, topk_2, focal_3, DPCE_4, dice_5, lovasz_6, tversky_7, GDice_8, asym_9, PGDice_10, boundary_11,
        hausdorff_12, sensitivity_specificity_loss13, focal_tversky_loss14, overall_Mismatch15, wce1_2]

    Transpose_data = Transpose(list)

    max_idex = []
    for i in Transpose_data:
        p = np.argmax(i)
        max_idex.append(p+1)
    print(max_idex)



    name = cfg_name(r'D:\Desktop\FCN-master\Datasets\CamVid\class_dict.csv')

    # for i in range(len(name)):
    #     list[i] = Transpose_data[i]

    ply([data for data in list], name, style='-')



