import rpy2.robjects as robjects
import argparse
import os
import json
import warnings
import pandas as pd
import numpy as np
import time
import random
import collections
from tqdm import tqdm

startt = time.strftime('%c', time.localtime(time.time()))

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
    robjects.r.source('./20220802_RF_LASSO.R', encoding='utf-8')

    # 결과 파일 가져와서 정제하기
    time.sleep(1)
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
    model, n_epochs,train_loader,device,optimizer,valid_loader,n_items, items,acc_history, auc_history,f1_score_history = OperatorNetInit(data_path, n_epochs, kc_candidates, target_KC)

    print("="*20 + "TRAIN START" + "="*20)
    train(model, n_epochs,train_loader,device,optimizer,valid_loader,n_items,acc_history, auc_history,f1_score_history)
    print("="*20 + "TRAIN END" + "="*20)
    
    influence_mat = model.influence_matrix()
    influence_sum = []
    for i in range(n_items-1):
        influence_sum.append(sum(influence_mat[i]))
    
    rank = sorted(range(len(influence_sum)), key=lambda k: influence_sum[k], reverse=True)
    
    #print("Rank of KCs:")
    #print(items[rank])
#    for i in range(10):
#        print(items[rank.index(i)])

#    print(f1_score_history,auc_history, acc_history)
    
    ranked_kc_rel = items[rank]
    cur_err = max(auc_history)

    return ranked_kc_rel, cur_err


def GetRankedKCGraph(json_path, data_path,n_epochs):
    arr_ranked_kc_graph = np.zeros((num_total_kc, num_total_kc))
    
    # 전체 중 첫번째 subset (12개 KC)
    all_kcs, sub_kcs = SelPartialKCs(json_path, data_path)
    
    # 결과 저장 할 데이터 프레임 만들기
    #result_df = pd.DataFrame(index=range(0), columns=['target','rel1','rel2','rel3','rel4','rel5','rel6','rel7','rel8','rel9','rel10','auc'])
    result_df = pd.DataFrame(index=range(0), columns=['target','rel1','rel2','rel3','rel4','rel5','auc'])
    
    
    #for kc_idx in range(num_total_kc):
    for kc_idx in range(2):
        kc_candidates = random.sample(sub_kcs[all_kcs[kc_idx]], 5)
        best_auc = 0
        for i in range(10): 

            # operator net(train)
            ranked_kc_rel, cur_err = DoOperatorNet(data_path, n_epochs, kc_candidates, all_kcs[kc_idx])

            #print("Rank of KCs: ", ranked_kc_rel)
            #print("Error: ", cur_err)
            result_list = []
            result_list.append(all_kcs[kc_idx])
            result_list.extend(ranked_kc_rel)
            result_list.append(cur_err)
            result_df = result_df.append(pd.Series(result_list, index=result_df.columns), ignore_index=True)
            
            # 결과에 대한 판단 시작
            # 1. best보다 좋을 때 : 겹치는 것 남김
            if cur_err > best_auc:
                if i == 0:
                    best_auc = cur_err
                    best_kcs = ranked_kc_rel
                    kc_candidates = random.sample(sub_kcs[all_kcs[kc_idx]], 5)
                    #print("step 1-1 / ",  kc_candidates)
                else:
                    add_kc = list(set(ranked_kc_rel).intersection(set(result_df.filter(regex='rel', axis=1).iloc[-2])))
                    kc_candidates = []
                    kc_candidates.extend(add_kc)
                    cand = list(set(sub_kcs[all_kcs[kc_idx]]) - set(add_kc))
                    kc_candidates.extend(random.sample(cand, 5-len(add_kc)))
                    #print("step 1-2 / ",  kc_candidates)
            # 2. 직전 결과보다 현제 결과가 좋을 때 (best보다는 낮음) : 새로 들어온 것 남김
            elif cur_err > result_df['auc'].iloc[-2]:
                kc_candidates = []
                add_kc = list(set(ranked_kc_rel) - set(result_df.filter(regex='rel', axis=1).iloc[-2]))
                kc_candidates.extend(add_kc)
                cand = list(set(sub_kcs[all_kcs[kc_idx]]) - set(add_kc))
                kc_candidates.extend(random.sample(cand, 5-len(add_kc)))
                #print("step 2 / ",  kc_candidates)
            # 3. 직전 결과보다 현재 결과가 나쁠 때 : 새로 들어온 것 후보에서 제외
            else:
                sub_kc = set(ranked_kc_rel) - set(result_df.filter(regex='rel', axis=1).iloc[-2])
                cand = list(set(sub_kcs[all_kcs[kc_idx]]) - sub_kc)
                kc_candidates = random.sample(cand, 5)
                #print("step 3 / ",  kc_candidates)
        # 상위 40% auc에 가장 많이 등장했던 KC 하나와 best rels의 kc들을 하나씩 교체하면서 실험 후 최종적으로 best 선정
        const = result_df.auc.quantile(q=0.6)#.value_counts('rel1','rel2','rel3','rel4','rel5').idmax()

        temp = result_df[(result_df['target']==all_kcs[kc_idx]) & (result_df['auc'] > const)].filter(regex='rel', axis=1)
        cand_kcs = []
        cand_kcs.extend(temp.values.tolist())
        cand_kcs = sum(cand_kcs, [])

        remove_set = set(best_kcs)

        cand_kcs = [i for i in cand_kcs if i not in remove_set]

        most_kc = collections.Counter(cand_kcs).most_common(1)[0][0]

        for k in range(len(best_kcs)):
            kc_candidates = best_kcs.copy()
            kc_candidates[k] = most_kc

            ranked_kc_rel, cur_err = DoOperatorNet(data_path, n_epochs, kc_candidates, all_kcs[kc_idx])

            #print("Rank of KCs: ", ranked_kc_rel)
            #print("Error: ", cur_err)
            result_list = []
            result_list.append(all_kcs[kc_idx])
            result_list.extend(ranked_kc_rel)
            result_list.append(cur_err)
            result_df = result_df.append(pd.Series(result_list, index=result_df.columns), ignore_index=True)   

    
    result_df.to_csv('./OUT/FIR_Result_cond_test.csv', index=False)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='input data file')
parser.add_argument('--n_epochs', type=int, default=150, help='# of train epochs')
args = parser.parse_args()
json_path = './info_back.json'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
GetRankedKCGraph(json_path, args.data_path, args.n_epochs)

endt = time.strftime('%c', time.localtime(time.time()))
print(startt)
print(endt)

# usage
# python merge_main.py --data_path=./data/my_algebra_picture3000_ordered_binary.csv --n_epochs=150

