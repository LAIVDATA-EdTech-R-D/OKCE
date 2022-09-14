import argparse
import warnings
import pandas as pd
import time
import copy

from icecream import ic

startt = time.strftime('%c', time.localtime(time.time()))

from op_main import OperatorNetInit
from train import train

warnings.filterwarnings(action='ignore')

print("#"*20," FIR-DKT START ", "#"*20)


def DictSet(data_path):
    df = pd.read_csv(data_path)

    kc_list = df.columns.tolist()
    kc_list.remove('student')

    sub_kcs = {}

    for kc in kc_list:
        feature_kcs = copy.deepcopy(kc_list)
        feature_kcs.remove(kc)
        sub_kcs[kc] = feature_kcs
    
    return kc_list, sub_kcs


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

    sorted_influence = sorted(influence_sum)
    sorted_influence.reverse()

    return ranked_kc_rel, cur_err, sorted_influence


def GetRankedKCGraph(data_path, n_epochs):

    all_kcs, sub_kcs = DictSet(data_path)

    # 결과 저장 할 데이터 프레임 만들기
    col_ary = []
    col_ary.append('target')

    for i in range(len(all_kcs)-1):
        string = 'rel'+str(i+1)
        col_ary.append(string)
    
    for i in range(len(all_kcs)-1):
        string = 'rel'+str(i+1)+'_score'
        col_ary.append(string)

    col_ary.append('auc')
    result_df = pd.DataFrame(index=range(0), columns=col_ary)

    try: 
        for kc_idx in range(len(all_kcs)):

            ranked_kc_rel, cur_err, sorted_influence = DoOperatorNet(data_path, n_epochs, sub_kcs[all_kcs[kc_idx]], all_kcs[kc_idx])

            result_list = []
            result_list.append(all_kcs[kc_idx])
            result_list.extend(ranked_kc_rel)
            result_list.extend(sorted_influence)
            result_list.append(cur_err)
            result_df = result_df.append(pd.Series(result_list, index=result_df.columns), ignore_index=True)


        result_df.to_csv('../data/NonFS/SSM_NonFS_AddScore.csv', index=False)
    except:
        result_df.to_csv('../data/NonFS/SSM_NonFS_AddScore_error.csv', index=False)
        ic(all_kcs[kc_idx])
        ic(sub_kcs[all_kcs[kc_idx]])


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='input data file')
parser.add_argument('--n_epochs', type=int, default=150, help='# of train epochs')
args = parser.parse_args()

GetRankedKCGraph(args.data_path, args.n_epochs)

endt = time.strftime('%c', time.localtime(time.time()))
print(startt)
print(endt)

# usage
# python merge_main_nonFS.py --data_path=../data/kc_dedup_smath12_reshape.csv --n_epochs=150
# python merge_main_nonFS.py --data_path=../data/my_algebra_picture3000_ordered_binary_rename.csv --n_epochs=150
