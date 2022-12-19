import torch
import torch.nn as nn
class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size,output_size):
        super(NeuralNet, self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.l2=nn.Linear(hidden_size,hidden_size)
        self.l3=nn.Linear(hidden_size,output_size)
        self.relu=nn.ReLU()

    def forward(self,x):          #forwardpropagation 不斷向前傳遞資料進行神經網路 (與train.py的backpropagation顛倒) 一個進行 一個回去修正 訓練
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out 
        