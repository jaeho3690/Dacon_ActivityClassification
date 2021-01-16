import torch
import torch.nn as nn
import torch.nn.functional as F



class LSTMNet(nn.Module):
    def __init__(self,device,input_size,hidden_size=30,num_layers=1,output_size=61):
        super(LSTMNet, self).__init__()
        self.device=device
        
        self.input_size = input_size
        self.hidden_size= hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax
        
        self.lstm= nn.LSTM(input_size,hidden_size,num_layers)
        self.fc1 = nn.Linear(hidden_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
        
    def init_hidden_and_cell(self, batch_size):
        h0 = torch.zeros(self.num_layers,batch_size, self.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.zeros(self.num_layers,batch_size, self.hidden_size, requires_grad=True).to(self.device)
        return h0, c0
        
    def forward(self,X,batch_size):
        h0,c0 = self.init_hidden_and_cell(batch_size) 
        out,_ = self.lstm(X,(h0,c0))
        out = out[-1,:,:]
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out