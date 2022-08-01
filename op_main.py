import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import numpy as np
import pandas as pd

from data_loader import pre_loader, kdd_loader, pp_loader, selct_2_oper_data
from model import DKT
from train import train

import os
import random
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def OperatorNetInit(data_path, n_epochs, subset_list, target_KC):

    print("="*20 + "DATA LOADING" + "="*20)
    #batches, n_items = pre_loader(args.data_path, target_KC="KC01")
    batches, n_items = kdd_loader(data_path, subset_list, target_KC)

    # train, validation, test 나누기

    ratios = [.6, .3, .1]

    train_cnt = int(len(batches) * ratios[0])
    valid_cnt = int(len(batches) * ratios[1])
    test_cnt = len(batches) - train_cnt - valid_cnt

    cnts = [train_cnt, valid_cnt, test_cnt]

    # batches의 학생별 데이터를 섞어주는 코드
    batches = random.sample(batches, len(batches))

    # 섞은 데이터셋을 train, Valid, Test로 나누기
    train_data = batches[:cnts[0]]
    valid_data = batches[cnts[0]:cnts[0] + cnts[1]]
    test_data = batches[cnts[0] + cnts[1]:]

    print('size of train_data : ', len(train_data))
    print('size of valid_data : ', len(valid_data))
    print('size of test_data : ', len(test_data))


    print("="*20 + "TRAIN PREPARE" + "="*20)
    # hyper parameters
    input_size = len(batches[0][0])
    hidden_size = 50
    batch_size = 64
    n_epochs = n_epochs
    early_stop = 100

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    #device 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    print("-"*20 + "MODEL PREPARE" + "-"*20)
    #model 선언
    model = DKT(input_size=input_size, hidden_size=hidden_size, n_items=n_items, device=device)
    model = model.to(device)
    print(model)

    #optimizer 정의
    optimizer = optim.Adam(model.parameters())

    auc_history = []
    acc_history = []
    f1_score_history = []

    train_loader = pp_loader(train_data, batch_size)
    valid_loader = pp_loader(valid_data, batch_size)
    test_loader = pp_loader(test_data, batch_size)

    
    return model, n_epochs,train_loader,device,optimizer,valid_loader,n_items,acc_history, auc_history,f1_score_history







'''print("="*20 + "TRAIN START" + "="*20)
train(model, n_epochs,train_loader,device,optimizer,valid_loader,n_items,acc_history, auc_history,f1_score_history)
print("="*20 + "TRAIN END" + "="*20)

plt.figure(figsize=(10,3))
plt.ylim(0.6,1)
plt.plot(f1_score_history, label="F1-Score")
plt.plot(auc_history, label="AUC")
plt.plot(acc_history, label="ACC")
plt.xlabel("Score")
plt.legend()
plt.grid()
plt.show()
plt.savefig('result.png')

'''