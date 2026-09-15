import torch
from torch import nn, optim
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np

cuda = True if torch.cuda.is_available() else False
print('cuda:', cuda)
# 生成相应数据类型的torch
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def DFAN(data, AE, args):
    print("Third Stage: Anomaly Detection")
    AE.eval()
    AE.load_state_dict(torch.load("models/AE_0171.pth", map_location=lambda storage, loc: storage))
    # AE.load_state_dict(torch.load(AE, map_location=lambda storage, loc: storage))
    data = torch.from_numpy(args.data).float()
    # tset_load_data = Data.DataLoader(data, batch_size=40000, shuffle=False)
    # TEST_SIZE = args.length * args.width
    # test_time = TEST_SIZE
    # y_pred = np.empty((test_time, 188))
    # # y_pred = np.zeros(shape=(args.length * args.width))
    # with torch.no_grad():
    #     for i in range(test_time):
    #         print(i)
    #         torch.cuda.empty_cache()
    #         X_spe = data[i]
    #         # print(X_spe.shape)
    #         X_spe = Variable(X_spe.type(FloatTensor))
    #         X_spe = X_spe.reshape(1, X_spe.size(0))
    #         lat_fea, output = AE(X_spe.cuda(), args)
    #         # print(output.shape)
    #         output = output.reshape(output.shape[0] * output.shape[1])
    #         # print(output.shape)
    #         # label = output.data.cpu().numpy()  # torch.max(pred,1)[1].data.cpu().numpy().squeeze()
    #         y_pred[i, :] = output.data.cpu().numpy()  # 存储整个张量
    #
    # y_pred = y_pred.reshape(args.length, args.width, 188)
    # return lat_fea, y_pred
    tset_load_data = Data.DataLoader(data, batch_size=40000, shuffle=False)
    with torch.no_grad():
        for batch_idx, data in enumerate(tset_load_data):
            lat_fea, output, dec, dec1 ,_ = AE(data.cuda(), args)
    return lat_fea, output

def DFAN1(data, AE, model_weight, args):
    print("Third Stage: Anomaly Detection")
    AE.eval()
    AE.load_state_dict(torch.load(model_weight, map_location=lambda storage, loc: storage))
    data = torch.from_numpy(args.data).float()
    tset_load_data = Data.DataLoader(data, batch_size=40000, shuffle=False)
    with torch.no_grad():
        for batch_idx, data in enumerate(tset_load_data):
            lat_fea, output, dec, dec1, _ = AE(data.cuda(), args)
    return lat_fea, output