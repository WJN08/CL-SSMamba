import os
import time
import test
import train
import torch
import model_AE_transformer
import model_AE_mamba
import modal_AE_spaspemamba
import model
import confusespa
import threespe
import confusespanoSILC
import argparse
import numpy as np
import matplotlib
import scipy.io as sio
import evaluate as Eva
import matplotlib.pyplot as plt
import torch.utils.data as Data
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
import evaluate as Eva
from multiprocessing import Process
from sklearn.decomposition import PCA
from numpy import *

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # 正态分布，mean=0, std=0.02
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        # 正态分布，mean=1.0, std=0.02
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        # 初始化整个矩阵为常数0
        torch.nn.init.constant_(m.bias.data, 0.0)

def HAD(args):
    start_time = time.time()
    data = args.data  # 2D
    dat = args.dat
    gt = args.GT
    latent_layer_dim = args.latent_layer_dim

    torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DFAN = threespe.OrthoAE_Unet(args.input_dim, latent_layer_dim, args)
    AE = threespe.OrthoAE_Unet(args.input_dim, latent_layer_dim, args)
    DFAN.cuda()

    ####加载网络权重
    DFAN.apply(weights_init_normal)
    AE.apply(weights_init_normal)
    # AE.load_state_dict(torch.load("AE_0119.pth", map_location=lambda storage, loc: storage))

    # enc_fea, Pretrain_DFAN, args = train.pre_DFAN(data, dat, DFAN, AE, args)
    # train_loss, Trian_DFAN, weight = train.DFAN(data, DFAN, args)

    ############test#######################
    #
    # train_loss, Trian_DFAN, weight = train.DFAN(data, Pretrain_DFAN, args)
    # lat_fea, output = test.DFAN(data, Trian_DFAN, args)
    print(data.shape)
    lat_fea, output = test.DFAN(data, DFAN, args)
    rec_result = np.array(output.cpu(), dtype=float)
    # rec_result = rec_result.reshape(rec_result.shape[0] * rec_result.shape[1], rec_result.shape[2])
    print(rec_result.shape)
    print(data.shape)
    dat = args.dat
    date = dat.reshape(-1, dat.shape[2])
    min_max_normal = preprocessing.MinMaxScaler()
    rec_result = min_max_normal.fit_transform(rec_result)
    ######计算的结果#######
    AD_result, _ = model_AE_transformer.RX(rec_result - date)

    end_time = time.time()
    runing_time = end_time - start_time
    PD_PF_auc, PF_tau_auc, PF, _, _ = \
        Eva.false_alarm_rate(gt.reshape((-1, 1)), AD_result.reshape((-1, 1)))

    print('Dataset:', args.dataset_name)
    print('AUC: PD_PF_auc=%.5f / PF_tau_auc=%.5f' % (PD_PF_auc, PF_tau_auc))
    print('runing-time=%.5f' % runing_time)

    AD_result = AD_result.reshape(dat.shape[0], -1)

    if os.path.exists(args.save_dir):
        pass
    else:
        os.makedirs(args.save_dir)

    S_save = dict()
    S_save['gt'] = gt
    S_save['det'] = AD_result

    ############保存和显示重建的背景
    # result = AD_result.reshape((args.length, args.width))
    AD_result = {'result': AD_result}  # 包装为字典
    result_file_name = args.dataset_name + str('.mat')
    sio.savemat(os.path.join(args.save_dir, result_file_name), AD_result)
    plt.imshow(AD_result['result'])  # ,'gray'
    plt.axis('off')
    plt.margins(0, 0)
    plt.savefig("Result/AA4.png", bbox_inches='tight', pad_inches=0.0)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./Data/AA4_split.mat")
    # parser.add_argument('--data_dir', type=str, default="./Data/abu-airport-4.mat")
    # parser.add_argument('--data_dir', type=str, default="./Data/HAD100_40.mat")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pre_epochs', type=int, default=200)
    parser.add_argument('--itetations', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=40000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--latent_layer_dim', type=int, default=64)
    parser.add_argument('--anomal_prop', type=float, default=0.003)
    parser.add_argument('--bandwidth', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.3)  # DBSCAN参数， 邻域半径：  其值越大，聚类的类别数量越少
    parser.add_argument('--min_samples', type=int, default=3)  # DBSCAN参数，  邻域半径最少点：其值越大，聚类的类别数量越少，

    args = parser.parse_args()
    dataset = sio.loadmat(args.data_dir)
    datatrain = np.array(dataset['data_train'], dtype=float)
    data = np.array(dataset['data'], dtype=float)
    print(datatrain.shape)
    path, file = os.path.split(args.data_dir)
    file_name = file[:-4]
    args.dataset_name = file_name
    # HSI_3D = np.array(dataset['data'], dtype=float)
    GT = np.array(dataset['groundtruth'], dtype=float)
    ###PCA
    # #标准化datatrain
    INPUT_DIMENSION_CONV = 188
    rows = datatrain.shape[0]
    cols = datatrain.shape[1]
    data2d = datatrain.reshape(-1, datatrain.shape[2])
    data2d = preprocessing.scale(data2d)  # , axis=1
    pca = PCA(n_components=INPUT_DIMENSION_CONV)
    data2d = pca.fit_transform(data2d)
    data2d -= np.min(data2d)
    data2d /= np.max(data2d)
    data_new3d = data2d.reshape(rows, cols, INPUT_DIMENSION_CONV)
    print(data_new3d.shape)
    ## 标准化data 100 * 100 * 188
    row = data.shape[0]
    col = data.shape[1]
    data2 = data.reshape(-1, data.shape[2])
    data2 = preprocessing.scale(data2)  # , axis=1
    pca = PCA(n_components=INPUT_DIMENSION_CONV)
    data2 = pca.fit_transform(data2)
    data2 -= np.min(data2)
    data2 /= np.max(data2)
    data_new3 = data2.reshape(row, col, INPUT_DIMENSION_CONV)
    # dataset = dataset.reshape(length * width, INPUT_DIMENSION_CONV)
    # min_max_normal = preprocessing.MinMaxScaler()
    # args.data = min_max_normal.fit_transform(data_new3d)
    args.data = data_new3d
    args.input_dim = INPUT_DIMENSION_CONV
    # args.length = length
    # args.width = width
    args.dat = data_new3
    args.GT = GT
    ####################################wu聚类##################
    # args = parser.parse_args()
    # dataset = sio.loadmat(args.data_dir)
    # # datatrain = np.array(dataset['data_train'], dtype=float)
    # data = np.array(dataset['data'], dtype=float)
    # path, file = os.path.split(args.data_dir)
    # file_name = file[:-4]
    # args.dataset_name = file_name
    # # HSI_3D = np.array(dataset['data'], dtype=float)
    # GT = np.array(dataset['map'], dtype=float)
    # ### PCA #####################
    # # #标准化datatrain
    # INPUT_DIMENSION_CONV = 188
    # rows = data.shape[0]
    # cols = data.shape[1]
    # data2d = data.reshape(-1, data.shape[2])
    # data2d = preprocessing.scale(data2d)  # , axis=1
    # pca = PCA(n_components=INPUT_DIMENSION_CONV)
    # data2d = pca.fit_transform(data2d)
    # data2d -= np.min(data2d)
    # data2d /= np.max(data2d)
    # data_new3d = data2d.reshape(rows, cols, INPUT_DIMENSION_CONV)
    # print(data_new3d.shape)
    # ## 标准化data 100 * 100 * 188
    # row = data.shape[0]
    # col = data.shape[1]
    # data2 = data.reshape(-1, data.shape[2])
    # data2 = preprocessing.scale(data2)  # , axis=1
    # pca = PCA(n_components=INPUT_DIMENSION_CONV)
    # data2 = pca.fit_transform(data2)
    # data2 -= np.min(data2)
    # data2 /= np.max(data2)
    # data_new3 = data2.reshape(row, col, INPUT_DIMENSION_CONV)
    # # dataset = dataset.reshape(length * width, INPUT_DIMENSION_CONV)
    # # min_max_normal = preprocessing.MinMaxScaler()
    # # args.data = min_max_normal.fit_transform(data_new3d)
    # args.data = data_new3d
    # args.input_dim = INPUT_DIMENSION_CONV
    # args.length = row
    # args.width = col
    # args.dat = data_new3
    # args.GT = GT


    # args = parser.parse_args()
    # dataset = sio.loadmat(args.data_dir)
    # # datatrain = np.array(dataset['data_train'], dtype=float)
    # data = np.array(dataset['data'], dtype=float)
    # path, file = os.path.split(args.data_dir)
    # file_name = file[:-4]
    # args.dataset_name = file_name
    # # HSI_3D = np.array(dataset['data'], dtype=float)
    # # GT = np.array(dataset['map'], dtype=float)
    # GT = sio.loadmat("Data/ang20170908t225309_40.mat")["gt"]
    # GT = np.array(GT, dtype=float)
    # ### PCA #####################
    # # #标准化datatrain
    # INPUT_DIMENSION_CONV = 188
    # rows = data.shape[0]
    # cols = data.shape[1]
    # data2d = data.reshape(-1, data.shape[2])
    # data2d = preprocessing.scale(data2d)  # , axis=1
    # pca = PCA(n_components=INPUT_DIMENSION_CONV)
    # data2d = pca.fit_transform(data2d)
    # data2d -= np.min(data2d)
    # data2d /= np.max(data2d)
    # data_new3d = data2d.reshape(rows, cols, INPUT_DIMENSION_CONV)
    # print(data_new3d.shape)
    # ## 标准化data 100 * 100 * 188
    # row = data.shape[0]
    # col = data.shape[1]
    # data2 = data.reshape(-1, data.shape[2])
    # data2 = preprocessing.scale(data2)  # , axis=1
    # pca = PCA(n_components=INPUT_DIMENSION_CONV)
    # data2 = pca.fit_transform(data2)
    # data2 -= np.min(data2)
    # data2 /= np.max(data2)
    # data_new3 = data2.reshape(row, col, INPUT_DIMENSION_CONV)
    # # dataset = dataset.reshape(length * width, INPUT_DIMENSION_CONV)
    # # min_max_normal = preprocessing.MinMaxScaler()
    # # args.data = min_max_normal.fit_transform(data_new3d)
    # args.data = data_new3d
    # args.input_dim = INPUT_DIMENSION_CONV
    # # args.length = length
    # # args.width = width
    # args.dat = data_new3
    # args.GT = GT

    dir = os.getcwd()
    save_dir = os.path.join(dir, 'Result')
    args.save_dir = save_dir

    HAD(args)



