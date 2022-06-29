from copy import deepcopy
import torch.optim as optim
import torch
from sklearn.metrics import roc_curve
from sklearn import metrics
import numpy as np

best_model = None

eps = 1e-8

def loss_function(y_hat, y_real, n_items, device):
    delta = y_real[:,:, :n_items] + y_real[:,:, n_items:]
    mask = delta.sum(axis=-1).type(torch.BoolTensor).to(device)
    y_hat =  y_hat* (1-2*eps) + eps
    #correct는 처음~M까지의 데이터로 여기에는 정답일 경우, 문항 번호를 표기하는 곳임
    correct = y_real[:,:, :n_items].to(device)
    #correct는 정답에 대한 정오답 벡터(n_items 차원)
    #data.log()는 각 문항에 대한 확률값의 로그값(n_items 차원)
    #공식은 BCE 공식 그대로
    bce = - correct*y_hat.log() - (1-correct)*(1-y_hat).log()
    #bce값을 delta(정답쪽 one-hot과 오답쪽 one-hot을 더한 값, n_items 차원)와 곱한 후
    #이것을 axis -1 방향으로 더함
    #그러면 해당 문항의 bce 값을 알 수 있고, 모든 문항에 대한 확률값을 알 수 있음
    bce = (bce*delta).sum(axis=-1)
    #최종 반환 값에서 bce를 mask를 통해 모두 선택하고, 이를 평균내어 반환함
    return torch.masked_select(bce, mask).mean()


def train(model, n_epochs,train_loader,device,optimizer,valid_loader,n_items,auc_history,f1_score_history):
    print_interval = 1
    highest_auc = 0
    highest_epoch = 0
    
    for i in range(n_epochs):
        model.train()

        # train_loader에서 미니배치 반환
        for data in train_loader:
            data = data.to(device)
            y_hat_i = model(data) # 각 값은 문항별 확률값

            optimizer.zero_grad()
            # loss를 구하기 위해서는 반환된 값의 차원(n_items)과 다음 값의 차원(n_items / 2)로 설정해서 비교해야함
            # 따라서 mask를 씌우는 작업이 필요함
            # 해당 기능은 loss_function에서 구현함
            loss = loss_function(y_hat_i[:-1], data[1:], n_items, device)
            loss.backward()
            optimizer.step()

        model.eval()
        # 여기서 y_true와 y_score를 받음
        # 한 epoch을 지나면 다시 초기화됨
        # y_true값은 해당 문항이 정답인지 아닌지에 대한 리스트(실제값)
        # y_score는 해당 문항에 대한 예측 확률을 담고 있는 리스트(예측값)
        y_true, y_score = [], []

        with torch.no_grad():

            for data in valid_loader: 
                data = data.to(device)
                y_hat_i = model(data)
                loss = loss_function(y_hat_i[:-1], data[1:], n_items, device)

                # correc와 mask를 정의
                # correct는 처음부터 M까지의 데이터, 즉 정답값에 속하는 원핫벡터를 의미함
                correct = data[:,:, :n_items].to(device)
                # mask는 정오답에 상관없이 앞의 M개 데이터(정답)와 뒤의 M개 데이터(오답)를 더해서 문항의 위치를 확인하기 위한 용도임
                mask = (data[:,:, :n_items] + data[:,:, n_items:]).type(torch.BoolTensor).to(device)
                # y_true에 들어가는 것은 정답값임
                # 이 중에서 correct[1:]은 가장 첫번째 정답값을 제하고, 두번째 정답부터 끝까지를 담고 있음
                # mask[1:]을 통해 해당 문항이 정답인지 오답인지를 담고 있는 값을 만들 수 있음 -> 값은 정답이면 1, 아니면 0을 담고 있음
                y_true.append(torch.masked_select(correct[1:], mask[1:]) )

                # y_hat_i의 값은 각각 한칸 뒤의 문항에 대한 정답률을 추정하는 확률값임
                # 따라서 y_hat_i[:-1]를 통해 마지막 값은 무시하면 두번째 문항부터 마지막 문항까지 예측 확률값을 알 수 있음
                # 그래서 mask[1:]을 사용하면, 2번 문항부터의 문항번호를 알 수 있기에 해당 문항의 예측 확률값만 얻을 수 있음
                y_score.append(torch.masked_select(y_hat_i[:-1], mask[1:]) )

        # y_true와 y_score를 numpy로 바꿈
        y_true = torch.cat(y_true).detach().cpu().numpy()
        y_score = torch.cat(y_score).detach().cpu().numpy()

        # roc_auc_socre 구하기
        auc_score = metrics.roc_auc_score(y_true, y_score)
        auc_history.append(auc_score)

        # roc_auc_socre 구하기
        f1_score = metrics.f1_score(y_true, np.round(y_score))
        f1_score_history.append(f1_score)
        
        if (i + 1) % print_interval == 0:
            print('Epoch %d: f1_score=%.4f auc_score=%.4f highest_auc=%.4f' % (
                i + 1,
                f1_score,
                auc_score,
                highest_auc,
            ))

        if auc_score >= highest_auc:
            highest_auc = auc_score
            highest_epoch = i
            
            best_model = deepcopy(model.state_dict())
        # else:
        #     if early_stop > 0 and lowest_epoch + early_stop < i + 1:
        #         print("There is no improvement during last %d epochs." % early_stop)
        #         break
        
    print("The best validation roc_auc_score from epoch %d: %.4f" % (highest_epoch + 1, highest_auc))
    model.load_state_dict(best_model)
    return model,f1_score_history,auc_history
