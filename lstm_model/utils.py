import os
import csv
import pandas as pd
import torch


def load_trainfiles(data_dir):
    train=pd.read_csv(data_dir+'train_features.csv')
    train_labels=pd.read_csv(data_dir+'train_labels.csv')

    return train, train_labels
    
def load_testfiles(data_dir):
    test=pd.read_csv(data_dir+'test_features.csv')
    submission=pd.read_csv(data_dir+'sample_submission.csv')

    return test,submission

def model_save(model,model_save_path,iter,name=None):
        torch.save(
            model.state_dict(),
            f'{model_save_path}/{name}_{str(iter)}.pkl')
    
def model_load(model,model_save_path,config):
        model.load_state_dict(
            torch.load(f'{model_save_path}/{name}_{str(config.model_load_num)}.pkl',
                       map_location=config.device)
        ) 

def name_maker(config,name=None):
    pass

