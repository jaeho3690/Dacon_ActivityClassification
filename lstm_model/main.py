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

from utils import load_trainfiles,load_testfiles,model_save

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--work_type', type=str, default='train_with_val', help="choose work type from 'train' 'train_with_val' 'predict'")
    parser.add_argument('--device', type=str, default='cuda', help='sets device for model and PyTorch tensors')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=300, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='for data-loading; right now, only 1 works with h5py')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for lstm')
    parser.add_argument('--val_ratio',type=float,default=0.2,help='Train validation split ratio')

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


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device = 'cpu' if device == 'cpu' else config.device
    print('torch work_type:', config.device)

    model = LSTMNet(config.device,config.input_size)
    model = model.cuda()

    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = nn.DataParallel(model)

    if config.work_type=='train':
        train_df, train_labels = load_trainfiles(config.data_path)
        train_loader = torch.utils.data.DataLoader(
            dataset= HealthDataset(train_df,train_labels),
            batch_size=config.batch_size, shuffle=True,
            drop_last=True,num_workers=config.workers, pin_memory=True)
    
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) 

        train_step(config,train_loader,criterion,optimizer,model)

    elif config.work_type=='train_with_val':
        train_df, train_labels = load_trainfiles(config.data_path)
        # Starting id of validation set
        val_start_id = int(len(train_df.id.unique())* (1-config.val_ratio))
        # Split
        val_df = train_df[train_df['id']>val_start_id]
        val_labels = train_labels[train_labels['id']>val_start_id]
        train_df = train_df[train_df['id']<=val_start_id]
        train_labels = train_labels[train_labels['id']<=val_start_id]

        train_loader = torch.utils.data.DataLoader(
            dataset= HealthDataset(train_df,train_labels),
            batch_size=config.batch_size, shuffle=True,
            drop_last=True,num_workers=config.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            dataset= HealthDataset(val_df,val_labels),
            batch_size=config.batch_size, shuffle=True,
            drop_last=True,num_workers=config.workers, pin_memory=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) 
        train_step(config,train_loader,criterion,optimizer,model,val_loader,True)
    else:
        test_df, submission = load_testfiles(config.data_path)
        test_loader = torch.utils.data.DataLoader(
            dataset= HealthDataset(test_df,train_mode=False),
            batch_size=1, shuffle=False)


    print('process time:', time.time() - start_time)
    
def train_step(config,train_loader,criterion,optimizer,model,val_loader=None,validation=False):
    n_total_steps = len(train_loader)
    prev_val_loss = 100
    for epoch in range(config.num_epochs):
        for i, (X, labels) in enumerate(train_loader):  
            X = X.float().reshape(config.batch_size,config.sequence_length,-1).to(config.device)
            labels = labels.to(config.device)
            
            # Forward pass
            outputs = model(X,config.batch_size)
            loss = criterion(outputs, labels.view(-1))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 5 == 0 :
                print('-'*50)
                print(f'Epoch [{epoch+1}/{config.num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        if validation:
            with torch.no_grad():
                val_loss_sum=0
                for i, (X, labels) in enumerate(val_loader):
                    X = X.float().reshape(config.batch_size,config.sequence_length,-1).to(config.device)
                    val_labels = labels.to(config.device)
                    val_outputs = model(X,config.batch_size)
                    val_loss_sum += criterion(val_outputs, val_labels.view(-1))
                val_loss_sum = val_loss_sum/len(val_loader)
                print(f'Epoch [{epoch+1}/{config.num_epochs}], Validation Loss: {val_loss_sum.item():.4f}')
                print('')
                if val_loss_sum <prev_val_loss:
                    prev_val_loss= val_loss_sum

                

def test_step(config,test_loader,submission,model):
    with torch.no_grad():
        for i, X in enumerate(test_loader):
            X = X.float().reshape(config.sequence_length,1,-1).to(config.device)
            outputs = model(X,1)
            submission.iloc[i,1:]= outputs.cpu().detach().numpy()[0]
            #_, predicted = torch.max(outputs.data, 1)
    
if __name__ == "__main__":
    main()