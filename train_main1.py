import torch
from torch import nn, optim
import torch.utils.data as Data
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
import test
import model_AE
from sklearn import preprocessing
import evaluate as Eva
import mask as STF

#####衡量两个分布之间的差距，可以衡量不同学习器之间的差距########
class KLD(nn.Module):
    def __init__(self):
        super(KLD, self).__init__()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

    def forward(self, dec, dec1):
        KLD_loss = 0
        KLD_loss += self.criterion_KLD(F.log_softmax(dec, dim=1), F.softmax(dec1, dim=1).detach())

        return KLD_loss

#######EWC损失下的持续########
class EWC(nn.Module):
    def __init__(self):
        super(EWC, self).__init__()
        # self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

    def criterion(self,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        #for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
        #    loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        for name, param in self.model.named_parameters():
            loss_reg += torch.sum(self.fisher[name]*(self.old_param[name] - param).pow(2))/2

        return self.lamb * loss_reg

#### 预训练DFAN
def pre_DFAN(train_data, dat,  AE, AE_OLD, args):
    epochs = args.pre_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    dat = torch.from_numpy(dat).float().to('cuda')
    x_s = dat
    # dat = dat.reshape(-1, dat.shape[2])
    train_data = torch.from_numpy(train_data).float()
    train_load_data = Data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(AE.parameters(), lr=learning_rate)
    min_max_normal = preprocessing.MinMaxScaler()
    ######### 加载之前的模型 ##############
    fisher1 = {}
    fisher2 = {}
    fisher3 = {}
    fisher4 = {}
    # 训练时，把下面注释解开
    for name, param in AE_OLD.named_parameters():
        if name == "encoder":
            fisher1[name] = param.data.clone().detach()
        if name == "encoder1":
            fisher2[name] = param.data.clone().detach()
        ## loss_reg += torch.sum(self.fisher[name] * (self.old_param[name] - param).pow(2)) / 2
    print("First Stage: Pretraining DFAN")
    for epoch in range(epochs):
        train_loss = 0
        AE.train()
        prev_acc1 = 20
        for idx, data in enumerate(train_load_data):
            data = data.cuda()
            optimizer.zero_grad()
            z_c, output, dec, dec1, dec2 = AE(data, args)
            loss_function = torch.nn.MSELoss()
            #####不同的学习器之间的损失，拉近学习器之间的距离#####
            loss_kld = KLD()
            # print(output.shape)  # 10000 188
            # print(dat.shape)    # 10000 188
            dat = dat.reshape(-1, data.shape[2])

            ##########  多个持续学习器损失 ##############
            loss_cl = loss_kld(dec, dec1)
            loss_cl += loss_kld(dec2, dec1)
            loss_cl += loss_kld(dec, dec2)
            ########## EWC损失 ##############
            loss_ewc = 0
            for name, param in AE.named_parameters():
                if name == "encoder":
                    fisher3[name] = param.data.clone().detach()
                    loss_ewc += torch.sum((fisher1[name] - fisher3[name]).pow(2)) / 2
                if name == "encoder1":
                    fisher4[name] = param.data.clone().detach()
                    loss_ewc += torch.sum((fisher2[name] - fisher4[name]).pow(2)) / 2
            # 运行 Small Target Filter
            # stf = STF.SmallTargetFilter(radius=3, sigma_s=1.0, sigma_c=1.0)
            # filtered_mask = stf.forward(x_s)
            # print("ffff", filtered_mask.shape)
            # 将 rec_result 从 NumPy 转换为 PyTorch 张量
            # rec_result = torch.tensor(output, dtype=torch.float32, device=dat.device)  # 确保与 dat 在同一设备上
            # rec_result = min_max_normal.fit_transform(rec_result)
            # datatest = args.dat
            # datatest = datatest.reshape(-1, datatest.shape[2])
            # AD_result, _ = model_AE.RX(rec_result - datatest)
            # AD_result = filtered_mask * output
            loss_ce = loss_function(output, dat)
            ############### total loss ###################
            loss = loss_ce + loss_ewc + loss_cl
            print(loss)
            print("Is contiguous:", output.is_contiguous())
            # loss.backward()
            # 打印梯度张量的信息
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
            file = open("result.txt", "a")
            if epoch > 1 and loss.item() < prev_acc1:
                print(loss.item())
                prev_acc1 = loss.item()

                filename_ae = 'models/AE_%04d.pth' % (epoch)
                torch.save(AE.state_dict(), filename_ae)
                s1 = str(epoch) + " "
                # 写入数据
                file.write(s1)

                lat_fea, output = test.DFAN1(train_data, AE, filename_ae, args)
                print("dat", dat.shape)
                print(output.shape)
                rec_result = np.array(output.cpu(), dtype=float)
                rec_result = min_max_normal.fit_transform(rec_result)
                datatest = args.dat
                datatest = datatest.reshape(-1, datatest.shape[2])
                AD_result, _ = model_AE.RX(rec_result - datatest)
                # print("filtered_mask device:", filtered_mask.device)
                # filtered_mask = filtered_mask.view(-1)
                # # print("AD_result device:", AD_result.device)
                # filtered_mask = filtered_mask.cpu().numpy()
                # AD_result = filtered_mask * AD_result
                print(AD_result.shape)
                print("000000000")
                PD_PF_auc, PF_tau_auc, PF, _, _ = \
                    Eva.false_alarm_rate(args.GT.reshape((-1, 1)), AD_result.reshape((-1, 1)))
                PD_PF_auc = str(PD_PF_auc) + "\n"
                file.writelines(PD_PF_auc)
            # 关闭文件
            file.close()
            torch.cuda.empty_cache()

    return z_c, AE, args

def DFAN(data, model, args):
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    input = torch.from_numpy(data).float()
    train_data = Data.DataLoader(input, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    min_max_normal = preprocessing.MinMaxScaler()
    print("Second Stage: Training DFAN")
    for epoch in range(epochs):
        train_loss = 0
        prev_acc1 = 20
        model.train()
        for idx, data in enumerate(train_data):
            data = data.cuda()
            lat_fea,  output = model(data, args)
            args.intra_dis, args.weight, args.inter_dis, intra_center_dis_mean = utils.AAM(lat_fea, args)
            loss_function = torch.nn.MSELoss(reduction='none')
            rec_loss_all_pixels = torch.mean(loss_function(output, data), dim=1)
            rec_loss = torch.matmul(args.weight.cuda(), rec_loss_all_pixels) / rec_loss_all_pixels.shape[0]
            intra_loss = args.intra_dis
            inter_loss = args.inter_dis
            intra_inter_loss_mean = intra_loss + inter_loss
            loss = rec_loss + intra_inter_loss_mean.cuda()
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
            file = open("result.txt", "a")
            if epoch > 1 and loss.item() < prev_acc1:
                print(loss.item())
                prev_acc1 = loss.item()

                filename_ae = 'models/AE_%04d.pth' % (epoch)
                torch.save(model.state_dict(), filename_ae)
                s1 = str(epoch) + " "
                # 写入数据
                file.write(s1)
                lat_fea, output = test.DFAN1(train_data, model, filename_ae, args)

                rec_result = np.array(output.cpu(), dtype=float)
                rec_result = min_max_normal.fit_transform(rec_result)
                AD_result, _ = model_AE.RX(rec_result - args.data)
                PD_PF_auc, PF_tau_auc, PF, _, _ = \
                    Eva.false_alarm_rate(args.GT.reshape((-1, 1)), AD_result.reshape((-1, 1)))
                PD_PF_auc = str(PD_PF_auc) + "\n"
                file.writelines(PD_PF_auc)
            # 关闭文件
            file.close()
            torch.cuda.empty_cache()

    return train_loss, model,args.weight
