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
import copy

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

def SelectKCSet(KC_picking_tb, KC_candidates):
    pick_n = 5 - len(KC_candidates)
    m = pick_n - 1
    
    KC_picking_tb = dict(sorted(KC_picking_tb.items(), key=lambda x:x[1], reverse=False))
#    print("low : ", KC_picking_tb)
    low_set = []
    for key, value in KC_picking_tb.items():
        if key not in KC_candidates:
            low_set.append(key)
        if len(low_set) >= 3:
            break
    pick1 = random.sample(low_set, 1)
    KC_candidates.extend(pick1)
    
    KC_picking_tb = dict(sorted(KC_picking_tb.items(), key=lambda x:x[1], reverse=True))
#    print("high : ", KC_picking_tb)
    high_set = []
    for key, value in KC_picking_tb.items():
        if key not in KC_candidates:
            high_set.append(key)
        if len(high_set) >= 4:
            break
    pick2 = random.sample(high_set, m)
    KC_candidates.extend(pick2)
    
    # picking table update
    picks = pick1+pick2
    print("pick1 : ", pick1)
    print("pick2 : ", pick2)
    for i in range(len(picks)):
        KC_picking_tb[picks[i]] += 1 
    print("4 KC_picking_tb +++++ : ", KC_picking_tb)

    return KC_candidates, KC_picking_tb

def GetRankedKCGraph(json_path, data_path,n_epochs):
    arr_ranked_kc_graph = np.zeros((num_total_kc, num_total_kc))
    
    # 전체 중 첫번째 subset (12개 KC)
    all_kcs, sub_kcs = SelPartialKCs(json_path, data_path)
    
    # 결과 저장 할 데이터 프레임 만들기
    #result_df = pd.DataFrame(index=range(0), columns=['target','rel1','rel2','rel3','rel4','rel5','rel6','rel7','rel8','rel9','rel10','auc'])
    result_df = pd.DataFrame(index=range(0), columns=['target','rel1','rel2','rel3','rel4','rel5','auc'])
    
    
    try: 
        for kc_idx in range(num_total_kc):
        #for kc_idx in range(15,20):
            KC_picking_tb = dict(zip(sub_kcs[all_kcs[kc_idx]], [0]*12))
            print("KC_picking_tb : ", KC_picking_tb)

            cnt_best_kcs = 0
            best_auc = 0
            cur_err = 0
            best_kcs = []
            kc_candidates = random.sample(sub_kcs[all_kcs[kc_idx]], 5)
            new_kcs = []
            
            # operator net(train)
            ranked_kc_rel, cur_err = DoOperatorNet(data_path, n_epochs, kc_candidates, all_kcs[kc_idx])

            result_list = []
            result_list.append(all_kcs[kc_idx])
            result_list.extend(ranked_kc_rel)
            result_list.append(cur_err)
            result_df = result_df.append(pd.Series(result_list, index=result_df.columns), ignore_index=True)
            
            before_kcs = copy.deepcopy(ranked_kc_rel)
            best_kcs = copy.deepcopy(ranked_kc_rel)
            best_auc = copy.deepcopy(cur_err)

            print("처음 : ", ranked_kc_rel)

            kc_candidates = random.sample(sub_kcs[all_kcs[kc_idx]], 5)
            print("다음 턴에 들어갈 KC KC: ", kc_candidates)
            
            # picking table update
            for i in range(len(kc_candidates)):
                KC_picking_tb[kc_candidates[i]] += 1
            print("1 KC_picking_tb +++++ : ", KC_picking_tb)

            for i in range(5): 
                print("$"*20,"iter : ", i)
                
                # operator net(train)
                print("11ranked_kc_rel :", kc_candidates)
                ranked_kc_rel, cur_err = DoOperatorNet(data_path, n_epochs, kc_candidates, all_kcs[kc_idx])
                print("22ranked_kc_rel :", ranked_kc_rel)

                result_list = []
                result_list.append(all_kcs[kc_idx])
                result_list.extend(ranked_kc_rel)
                result_list.append(cur_err)
                result_df = result_df.append(pd.Series(result_list, index=result_df.columns), ignore_index=True)

                before_auc = result_df['auc'].iloc[-2]

                
                # 결과에 대한 판단 시작
                # best set이 똑같은 게 3번 나오면 for문 종료
                if cnt_best_kcs >= 3:
                    print("Early Stopping")
                    print("Best KC Set: ", best_kcs)
                    print("Best AUC: ", best_auc)
                    break
        
                # 현재와 best set이 같을 때
                if set(ranked_kc_rel) == set(best_kcs):
                    print("같은 셋이 나왔음")
                    cnt_best_kcs += 1
                    before_kcs = copy.deepcopy(best_kcs)
                    new_kcs = []
                # 현재 AUC가 직전 AUC보다 높다면
                elif cur_err >= before_auc:
                    print("다른 셋이면서 현재 AUC 직전 것보다 높음")
                    # 현재 AUC가 best보다 높다면
                    if cur_err > best_auc:
                        print("다른 셋이면서 현재 AUC best보다 높음")
                        cnt_best_kcs = 1
                        before_kcs = copy.deepcopy(best_kcs)
                        best_kcs = copy.deepcopy(ranked_kc_rel)
                        best_auc = cur_err
                        new_kcs = list(set(ranked_kc_rel) - set(before_kcs))
                    else: # 다른 셋이면서 직전과 best 사이
                        print("다른 셋이면서 현재 AUC가 best와 직전 사이")
                        new_kcs = list(set(ranked_kc_rel) - set(before_kcs))
                # 현재 AUC가 직전 AUC보다 낮다면
                else:
                    print("현재 AUC 직전보다 낮음")
                    new_kcs = []
                
                
                kc_candidates = []
                kc_candidates.extend(new_kcs)

                # picking table update
                for i in range(len(new_kcs)):
                    KC_picking_tb[new_kcs[i]] += 1
                print("2 KC_picking_tb +++++ : ", KC_picking_tb)

                org_kcs = list(set.difference(set(ranked_kc_rel) - set(new_kcs)))
                # picking table update
                for i in range(len(org_kcs)):
                    KC_picking_tb[org_kcs[i]] -= 1
                print("3 KC_picking_tb ----- : ", KC_picking_tb)

                if len(kc_candidates) < 5:
                    kc_candidates, KC_picking_tb = SelectKCSet(KC_picking_tb, kc_candidates)
                
                print()
                print("best_kcs :", best_kcs)
                print("before_kcs :", before_kcs)        
                print("ranked_kc_rel :", ranked_kc_rel)
                print("new_kcs :", new_kcs)
                print()
                
                before_kcs = copy.deepcopy(ranked_kc_rel)
                print("다음 턴에 들어갈 KC KC: ", kc_candidates)
                print()
                
            with open('./OUT/0822KC_picking_tb.txt','w',encoding='UTF-8') as f:
                for code,name in KC_picking_tb.items():
                    f.write(f'{code} : {name}\n')
            
        result_df.to_csv('./OUT/selector_update_test0822.csv', index=False)
    except:
        print("문제 : ", all_kcs[kc_idx])
        print("kc_idx : ", kc_idx)
        result_df.to_csv('./OUT/selector_update_test0822_error.csv', index=False)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='input data file')
parser.add_argument('--n_epochs', type=int, default=150, help='# of train epochs')
args = parser.parse_args()
json_path = './info_back.json'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GetRankedKCGraph(json_path, args.data_path, args.n_epochs)

endt = time.strftime('%c', time.localtime(time.time()))
print(startt)
print(endt)

# usage
# python merge_main.py --data_path=./data/my_algebra_picture3000_ordered_binary.csv --n_epochs=150

