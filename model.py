import torch
import torch.nn as nn
import torch.utils
import numpy as np

class DKT(nn.Module):
    """
    LSTM based model
    """

    def __init__(
        self,
        input_size,
        hidden_size, # rnn은 hidden size == output size
        n_items,
        device,
        #output_size,
        n_layers=4,
        dropout_p=.2,
        
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.n_items = n_items
        self.device = device

        super().__init__()
        
        self.rnn = nn.LSTM(
            # input_size는 총 문항 수의 두배인 2M
            input_size = input_size,
            hidden_size = hidden_size, 
            num_layers = n_layers,
            # (입력 데이터의 가장 첫번째 차원이 batch가 될 수 있도록 batch_first 옵션을 줌)
            # data loader를 통과하니 순서가 바뀌어서 일단 수정함
            # 입력되는 데이터가 (sequence, bs, length) 순서
            batch_first = False,
            dropout = dropout_p,
        )

        self.layers = nn.Sequential(
            # Linear 층에서 hidden_size를 크기로 받고, input_size/2(M개)로 출력
            nn.Linear(hidden_size, int(input_size/2)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # |x| = (batch_size, h, w) = (bs, sequence_length, vector_size)
        z, _ = self.rnn(x)
        # rnn의 output으로 output과 (h_n, c_n)이 나옴
        # 뒷 값은 필요없으므로, _로 무시
        # output인 |z| = (batch_size, sequence_length, hidden_size)
        # z는 모든 타입스텝의 결과를 가져오는 것임

        # 우리는 many to one이므로, 마지막 타임스텝의 결과값만 잘라서 가져오면 됨
        #z = z[:, -1] 
        
        # |z| = (batch_size, hidden_size)
        y = self.layers(z)
        # |y| = (batch_size, output_size)

        return y
    
    def influence_matrix(self):
        matrix = self._conditional_predict_matrix()
        matrix = matrix / np.sum(matrix, axis=0)
        return matrix
    
    def _conditional_predict_matrix(self):
        self.eval()
        x = torch.zeros(size=(1, self.n_items, self.n_items*2)).to(self.device)
        for n in range(self.n_items):
            x[0, n, n] = 1
        with torch.no_grad():
            return self(x)[0].detach().cpu().numpy()