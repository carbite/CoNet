import argparse
import time

import numpy as np
import sys
import torch
import torch.utils.data
import torch.nn as nn

import torch.optim as optim

sys.path.append('/home/lvfengmao/wanggang/GeneInference/CoNet/')

from utils.load_data import MyDataSet
from model.CoNet_3000 import CoNet_3000
from model.CoNet_6000_3000 import CoNet_6000_3000
from model.CoNet_6000_4000_2000 import CoNet_6000_4000_2000
'''
1、定义超参数
'''
# 采用的网络模型
MODEL = 'CoNet_6000_3000'
# 训练批次数
NUM_EPOCH = 200
# batch的大小
BATCH_SIZE = 5000
# 输入维度大小
IN_SIZE = 943
# 输出维度大小
OUT_SIZE = 9520
'''AE'''
HIDDEN_D1_SIZE = 6000
HIDDEN_D2_SIZE = 3000
HIDDEN_D3_SIZE = 1000
D_SIZE = 1000
# dropout
DROPOUT_RATE_AE = 0.1
# 学习率
LEARNING_RATE_LR = 5e-4
LEARNING_RATE_AE = 5e-4

LAMDA = 1e-3
def get_arguments():
    """
    Parse all the arguments provided from the CLI.

    Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="dense idea's arguments")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="the network structure that you want use,CoNet_3000? or CoNet_6000_3000 and so on")
    parser.add_argument("--num-epoch", type=int, default=NUM_EPOCH,
                        help="iter numbers")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="batch's size")
    parser.add_argument("--hidden-d1-size", type=int, default=HIDDEN_D1_SIZE,
                        help="AE's first hidden layer's size")
    parser.add_argument("--hidden-d2-size", type=int, default=HIDDEN_D2_SIZE,
                        help="AE's second hidden layer's size")
    parser.add_argument("--hidden-d3-size", type=int, default=HIDDEN_D3_SIZE,
                        help="AE's third hidden layer's size")
    parser.add_argument("--d-size", type=int, default=D_SIZE,
                        help="AE's middle layer's size")
    parser.add_argument("--dropout-rate-ae", type=float, default=DROPOUT_RATE_AE,
                        help="dropout rate, 0.1,0.25?")
    parser.add_argument("--learning-rate-lr", type=float, default=LEARNING_RATE_LR,
                        help="learning rate of LR, 5e-4?")
    parser.add_argument("--learning-rate-ae", type=float, default=LEARNING_RATE_AE,
                        help="learning rate of AE, 5e-4?")
    parser.add_argument("--lamda", type=float, default=LAMDA,
                        help="lamda for L1, 1e-3?")
    return parser.parse_args()

args = get_arguments()


def main():
    '''
    2、读取数据
    '''
    print('loading data...:'+args.dataset)

    tr_set = MyDataSet(x_path='../dataset/bgedv2_X_tr_float64.npy', y_path='../dataset/bgedv2_Y_tr_float64.npy')
    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batch_size, shuffle=True)

    X_va = torch.from_numpy(np.array(np.load('../dataset/bgedv2_X_va_float64.npy'))).type(torch.FloatTensor).cuda()
    Y_va = torch.from_numpy(np.array(np.load('../dataset/bgedv2_Y_va_float64.npy'))).type(torch.FloatTensor).cuda()

    X_te = torch.from_numpy(np.array(np.load('../dataset/bgedv2_X_te_float64.npy'))).type(torch.FloatTensor).cuda()
    Y_te = torch.from_numpy(np.array(np.load('../dataset/bgedv2_Y_te_float64.npy'))).type(torch.FloatTensor).cuda()

    X_1000G = torch.from_numpy(np.array(np.load('./dataset/1000G_X_float64.npy'))).type(torch.FloatTensor).cuda()
    Y_1000G = torch.from_numpy(np.array(np.load('./dataset/1000G_Y_float64.npy'))).type(torch.FloatTensor).cuda()

    X_GTEx = torch.from_numpy(np.array(np.load('../dataset/GTEx_X_float64.npy'))).type(torch.FloatTensor).cuda()
    Y_GTEx = torch.from_numpy(np.array(np.load('../dataset/GTEx_Y_float64.npy'))).type(torch.FloatTensor).cuda()

    '''
    2、定义网络
    '''
    net = globals()[args.model](IN_SIZE, OUT_SIZE, args.d_size, args.hidden_d1_size, args.hidden_d2_size,
                                args.hidden_d3_size, args.dropout_rate_ae).cuda()
    net = nn.DataParallel(net, device_ids=[0])
    '''
    3、定义Loss和优化器
    '''

    criterion = nn.MSELoss(reduce=True, size_average=False)
    net_optimizer = optim.Adam([
        {'params': net.module.fcnet.parameters(), 'lr': args.learning_rate_lr, 'weight_decay': args.lamda},
        {'params': net.module.encoder.parameters()},
        {'params': net.module.decoder.parameters()}
    ], lr=args.learning_rate_ae)
    '''
    4、开始训练网络
    '''

    MAE_te_best = 10.0
    MAE_GTEx_best = 10.0
    net_parameters_GEO = {}
    net_parameters_GTEx = {}

    outlog = open('../../res/CoNet/'+args.model+'.log', 'w')
    log_str = '\t'.join(map(str, ['epoch', 'MAE_va', 'MAE_te','MAE_1000G',  'MAE_GTEx', 'MAE_tr',  'time(sec)']))
    print(log_str)
    outlog.write(log_str + '\n')
    sys.stdout.flush()

    for epoch in range(args.num_epoch):
        for i, data in enumerate(tr_loader, 0):
            t_old = time.time()
            '''
            开始训练了
            '''
            # forward
            net.train()
            x_batch, y_batch = data
            x_batch = x_batch.type(torch.FloatTensor).cuda()
            y_batch = y_batch.type(torch.FloatTensor).cuda()

            y_fc,y_ae = net.module(x_batch, y_batch)

            y_fc_loss = criterion(y_fc, y_batch)
            y_ae_loss = criterion(y_ae, y_batch)
            all_loss = y_fc_loss + y_ae_loss
            # backward
            net_optimizer.zero_grad()
            all_loss.backward()
            net_optimizer.step()

            torch.cuda.empty_cache()

            '''
            开始验证了
            '''
            with torch.no_grad():
                net.eval()
                #计算output
                va_outputs, _ = net.module(X_va, Y_va)
                te_outputs, _ = net.module(X_te, Y_te)
                l000G_outputs, _ = net.module(X_1000G, Y_1000G)
                GTEx_outputs, _ = net.module(X_GTEx, Y_GTEx)

                #计算MAE
                MAE_tr = np.abs(y_batch.detach().cpu().numpy() - y_fc.detach().cpu().numpy()).mean()
                MAE_va = np.abs(Y_va.detach().cpu().numpy() - va_outputs.detach().cpu().numpy()).mean()
                MAE_te = np.abs(Y_te.detach().cpu().numpy() - te_outputs.detach().cpu().numpy()).mean()
                MAE_1000G = np.abs(Y_1000G.detach().cpu().numpy() - l000G_outputs.detach().cpu().numpy()).mean()
                MAE_GTEx = np.abs(Y_GTEx.detach().cpu().numpy() - GTEx_outputs.detach().cpu().numpy()).mean()


                t_new = time.time()
                log_str = '\t'.join(
                    map(str, [(epoch * np.ceil(88807/args.batch_size)) + i + 1, '%.6f' % MAE_va, '%.6f' % MAE_te,
                              '%.6f' % MAE_1000G, '%.6f' % MAE_GTEx,
                              '%.6f' % MAE_tr, int(t_new - t_old)]))
                print(log_str)
                outlog.write(log_str + '\n')
                sys.stdout.flush()
                # 保留最优MAE_te
                if MAE_te < MAE_te_best:
                    MAE_te_best = MAE_te
                    net_parameters_GEO = net.state_dict()
                if MAE_GTEx < MAE_GTEx_best:
                    MAE_GTEx_best = MAE_GTEx
                    net_parameters_GTEx = net.state_dict()
        print("epoch %d training over" % epoch)
    # 保存训练出来的模型
    torch.save(net_parameters_GEO, '../../res/CoNet/' + args.model + '_GEO.pt')
    torch.save(net_parameters_GTEx, '../../res/dense/' + args.model + '_GTEx.pt')
    print('MAE_te_best : %.6f' % (MAE_te_best))
    print('MAE_GTEx_best : %.6f' % (MAE_GTEx_best))
    outlog.write('MAE_te_best : %.6f' % (MAE_te_best) + '\n')
    outlog.write('MAE_GTEx_best : %.6f' % (MAE_GTEx_best) + '\n')
    outlog.close()
    print('Finish Training')
main()