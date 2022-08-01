import rpy2.robjects as robjects
import argparse
import os
import json
import warnings
import pandas as pd
import numpy as np
import time

from op_main import OperatorNetInit
from train import train

warnings.filterwarnings(action='ignore')

print("#"*20," FIR-DKT START ", "#"*20)

num_partial_kc = 12
num_candidate_kc = 10
num_total_kc = 20
num_logs = 1000
target_err = 0.05


# 전체 세트에서 첫번째 subset 반환
def SelPartialKCs(json_path, data_path):
    sub_kcs = np.zeros(num_partial_kc)
    
    #json_path = './info_back.json'
    # 기존 json 파일 읽어오기
    with open(json_path, 'r') as file:
        data = json.load(file)

    # 데이터 수정
    data[0]['input_file_name'] = str(data_path)

    # 기존 json 파일 덮어쓰기
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent="\t", ensure_ascii=False)

    # R 관계 분석 실행
    robjects.r.source('./20220731_RF_LASSO.R', encoding='utf-8')
    
    # 결과 파일 가져와서 정제하기
    subset_df = pd.read_csv('./OUT/relation.csv')
    
    all_kcs = list(subset_df['before'].unique())
    
    
    # sub_kcs 는 첫번째 원소는 before, 그 뒤는 모두 rank 순으로 after임.
    sub_kcs = {}
    for i in range(len(all_kcs)):
        cond = subset_df['before'] == all_kcs[i]
        after_kcs = list(subset_df[cond]['after'])

        sub_kcs[all_kcs[i]] = after_kcs
    
    
    # 후보로 뽑힌 12개 KC 반환
    return all_kcs, sub_kcs


def DoOperatorNet(data_path, n_epochs, kc_candidates, target_KC):
    # operator net을 위한 init
    model, n_epochs,train_loader,device,optimizer,valid_loader,n_items,acc_history, auc_history,f1_score_history = OperatorNetInit(data_path, n_epochs, kc_candidates, target_KC)

    print("="*20 + "TRAIN START" + "="*20)
    train(model, n_epochs,train_loader,device,optimizer,valid_loader,n_items,acc_history, auc_history,f1_score_history)
    print("="*20 + "TRAIN END" + "="*20)

    print(f1_score_history,auc_history, acc_history)
    
    ranked_kc_rel = f1_score_history
    cur_err = auc_history

    return ranked_kc_rel, cur_err


def GetRankedKCGraph(json_path, data_path):
    arr_ranked_kc_graph = np.zeros((num_total_kc, num_total_kc))
    
    # 전체 중 첫번째 subset (12개 KC)
    all_kcs, sub_kcs = SelPartialKCs(json_path, data_path)
    
    
    for kc_idx in range(num_total_kc):
        # 12개 중 상위 10개 (임의로 상위 10개로 선정한 것. 나중에 수정 예정)
        kc_candidates = sub_kcs[all_kcs[kc_idx]][:10]
        
        # operator net(train)
        n_epochs=10
        ranked_kc_rel, cur_err = DoOperatorNet(data_path, n_epochs, kc_candidates, all_kcs[kc_idx])

        print(ranked_kc_rel, cur_err)
        exit()

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='input data file')
parser.add_argument('--n_epochs', type=int, default=30, help='# of train epochs')
args = parser.parse_args()
json_path = './info_back.json'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
GetRankedKCGraph(json_path, args.data_path)
