import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json

def  selct_2_oper_data(data_path, target, KC_list):
    df = pd.read_csv(data_path)
    df = df.sort_values('Unnamed: 0')
    df = df.reset_index(drop=True)
    # selector net에서 오는 데이터 example
    target = target
        #"Calculate.total.in.proportion.with.fractions"
    KC_list = KC_list
        #["Enter.given.part.in.proportion", "Enter.Calculated.value.of.rate", 
        #    'Enter.denominator.of.form.of.1', 'Enter.proportion.label.in.numerator']
    temp_list = KC_list
    temp_list.append(target)
    df = pd.concat([df['Unnamed: 0'],df[temp_list]],axis=1)
    re_cols = list(df.columns)
    re_cols.remove(target)
    re_cols.append(target)
    df = df[re_cols]
    stus = list(df['Unnamed: 0'].unique())
    index = []
    skill = []
    acc = []
    for i in range(len(stus)):
        for j in range(len(re_cols[1:])):
            index.append(stus[i])
            skill.append(re_cols[j+1])
            acc.append(df.loc[i][j+1])
    result = pd.DataFrame({ 'student':index, 'KC':skill, 'acc':acc })
    return result



def pre_loader(data_path, target_KC):
    data_pd = pd.read_csv(data_path)
    #print("Unique KC ", data_pd['KC'].unique())
    
    new_df = pd.DataFrame(index=range(0), columns = ['knowre_user_id', 'kc_uid', 'acc'])
    students = list(data_pd['knowre_user_id'].unique())
    for i in range(len(students)):
        cond = data_pd.knowre_user_id == students[i]
        temp1 = data_pd[cond]
        tail = temp1[temp1['kc_uid'] == target_KC]
        temp1 = temp1[temp1['kc_uid'] != target_KC]

        result = pd.concat([temp1,tail])
        new_df = pd.concat([new_df,result])
    new_df = new_df.reset_index(drop=True)
    print(new_df)

    idx_students, students, students_id_to_idx, one_hot_vectors, n_items = loader(new_df)
    batches, n_items = batch_loader(idx_students, students, students_id_to_idx, one_hot_vectors, n_items)

    return batches, n_items


def kdd_loader(data_path, subset_list, target_KC):

    # subset_list 학습 순서 매기기(KC 순서대로)
    ordering_json_path = '../json/KC_Ordering.json'
    # 기존 json 파일 읽어오기
    with open(ordering_json_path, 'r') as file:
        OrderingData = json.load(file)

    A = subset_list
    B = OrderingData[data_path]
    subset_list = [i for _,i in sorted(zip(B,A))]

    data_pd = pd.read_csv(data_path)
    subset_list.insert(0, "student")
    subset_list.append(target_KC)
    data_pd = data_pd[subset_list]
    cols = list(data_pd.columns)[1:]
    cols.remove(target_KC)
    cols.append(target_KC)
    students = list(data_pd['student'].unique())
    new_df = pd.DataFrame(index=range(0), columns = ['student', 'kc', 'acc'])
    
    index = []
    skill = []
    acc = []
    for i in range(len(students)):
        index.extend([students[i]]*len(cols))
        skill.extend(cols)
        acc.extend(data_pd.iloc[i:i+1,1:].values.reshape(-1).tolist())

    new_df = pd.DataFrame(zip(index, skill, acc), columns = ['student', 'kc', 'acc'])

    new_df = new_df.reset_index(drop= True)
    idx_students, students, students_id_to_idx, one_hot_vectors, n_items, items = loader(new_df)
    batches, n_items = batch_loader(idx_students, students, students_id_to_idx, one_hot_vectors, n_items)
    #print(new_df)
    return batches, n_items, items


def loader(new_df):
    data_np = new_df.to_numpy()
    # 변수 정리
    # 학생, 문항, 정답을 각각 np.array로 나눠서 담음
    students = data_np[:, 0]
    items = data_np[:, 1]
    answers = data_np[:, 2] 

    # 전체 학생 수
    n_students = np.unique(students, return_counts=True)
    n_students = n_students[0].shape[0]

    # 전체 문항 수
    n_items = np.unique(items, return_counts=True)
    n_items = n_items[0].shape[0]    

    #인덱스 테스트
    idx_students = np.unique(students)
    idx_items = np.unique(items)
    
    # one hot vector
    one_hot_vectors = np.zeros((len(items), n_items * 2))

    # items + answers to one-hot vectors
    # idx = list(idx_items).index(data_np[0][1])
    # print(idx) -> index에 해당하는 수가 나옴

    for i, data in enumerate(data_np):
        # 첫 행의 문항이 idx_items의 몇 번째 인덱스인지 파악해서 idx에 저장
        # data_np[i][1]은 i행의 문항임
        idx = list(idx_items).index(data_np[i][1])
        # 정답이라면 처음~M에 1(정답값)을 더하고,
        one_hot_vectors[i, idx] += data_np[i][2] # 정답값
        # 오답이면 M+1~2M에 1(정답값)을 더한다.
        one_hot_vectors[i, idx + n_items] += 1 - data_np[i][2] # 정답값
    
    # 리스트로 변경
    students = list(students)
    one_hot_vectors = list(one_hot_vectors)
    idx_students = list(idx_students)

    # unique한 idx_students의 각각의 요소들에 0부터 인덱스를 딕셔너리 형태로 달아줌
    students_id_to_idx = {}

    for i, unique_student in enumerate(idx_students):
        students_id_to_idx[unique_student] = i
    
    return idx_students, students, students_id_to_idx, one_hot_vectors, n_items, items
    batch_loader(idx_students, students, students_id_to_idx, one_hot_vectors, n_items)


def batch_loader(idx_students, students, students_id_to_idx, one_hot_vectors, n_items):
    # batches 만들기
    # batches는 []안에 unique한 학생의 수만큼의 빈 []를 넣어두는 변수
    # 이후에 각 학생별 [] 안에 학생이 푼 one_hot_vector를 집어넣음

    dummy_batches = []

    #dummy_batches에 학생의 유니크 한 수만큼의 빈 리스트 넣기
    for i in range(len(idx_students)):
        dummy_batches.append([])

    #학생의 인덱스에 맞는 빈 리스트에 one_hot_vector를 리스트형태로 넣기
    for i in range(len(students)):
        idx = students_id_to_idx[students[i]]
        dummy_batches[idx].append(one_hot_vectors[i])

    #진짜 batches를 받을 수 있도록 처리
    batches = []

    #dummy_batches에서 학생별로 데이터를 꺼내서 torch.Tensor로 바꿈
    for batch in dummy_batches:
        batches.append(torch.Tensor(batch))
    
    return batches, n_items

def pp_loader(data, batch_size):
    def collate(batch):
        return nn.utils.rnn.pad_sequence(batch)

    t_loader = DataLoader(
        dataset = data,
        batch_size = batch_size,
        shuffle = True,
        collate_fn=collate
    )
    
    return t_loader
