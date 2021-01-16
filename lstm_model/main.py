import os
import numpy as np
import pandas as pd 
import argparse 
import time 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import LSTMNet 
from dataset import HealthDataset 

from utils import load_trainfiles,load_testfiles

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--work_type', type=str, default='train', help="choose work type 'train' or 'predict'")

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='for data-loading; right now, only 1 works with h5py')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for lstm')

    # Data arguments
    parser.add_argument('--input_size', type=int, default=7, help='Feature size')
    parser.add_argument('--sequence_length', type=int, default=600, help='Number of samples per id')
    parser.add_argument('--output_size', type=int, default=61, help='Output size')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=30, help='Feature size')
    parser.add_argument('--num_layers', type=int, default=1, help='Feature size')

    # Directory arguments
    parser.add_argument('--data_path', type=str, default='../data/', help='original data path')
    parser.add_argument('--model_save_path', type=str, default='../model_save/lstm', help='model save path')
    parser.add_argument('--model_load_path', type=str, default='../model_save/lstm', help='model load path')
    parser.add_argument('--model_load_num', type=int, default=None, help='epoch number of saved model')
    config = parser.parse_args()

    print(torch.cuda.is_available())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device = 'cpu' if device == 'cpu' else config.device
    print('torch work_type:', config.device)

    model = LSTMNet(config.device,config.input_size)
    #model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if config.work_type=='train':
        train, train_labels = load_trainfiles(config.data_path)
        train_loader = torch.utils.data.DataLoader(
            dataset= HealthDataset(train,train_labels),
            batch_size=config.batch_size, shuffle=True,
            drop_last=True,num_workers=config.workers, pin_memory=True)
    
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) 

        train(config,train_loader,criterion,optimizer)

    elif config.work_type=='test':

        test, submission = load_testfiles(config.data_path)
        test_loader = torch.utils.data.DataLoader(
            dataset= HealthDataset(test,train_mode=False),
            batch_size=1, shuffle=False)
    
    else:
        pass

    print('process time:', time.time() - start_time)
    
def train(config,train_loader,criterion,optimizer):
    n_total_steps = len(train_loader)
    for epoch in range(config.num_epochs):
        for i, (X, labels) in enumerate(train_loader):  
            X = X.float().reshape(config.sequence_length,config.batch_size,-1).to(config.device)
            labels = labels.to(config.device)
            
            # Forward pass
            outputs = model(X,config.batch_size)
            loss = criterion(outputs, labels.view(-1))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 5 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

def test(config,test_loader,submission):
    with torch.no_grad():
        for i, X in enumerate(test_loader):
            X = X.float().reshape(config.sequence_length,1,-1).to(config.device)
            outputs = model(X,1)
            submission.iloc[i,1:]= outputs.cpu().detach().numpy()[0]
            #_, predicted = torch.max(outputs.data, 1)
    
if __name__ == "__main__":
    main()