import numpy as np 
import pandas as pd 

from torch.utils.data import Dataset
class HealthDataset(Dataset):
    def __init__(self,feature_df,label=None,train_mode=True):
        self.X = feature_df
        self.train_mode= train_mode
        self.first_test_id = self.X['id'].iloc[0]
        if self.train_mode:
            self.y = label
            
        
    def __len__(self):
        return len(self.X.id.unique())
    
    def __getitem__(self,idx):
        if self.train_mode:
            return (np.array(self.X[self.X['id']==idx+self.first_test_id].iloc[:,1:]),np.array(self.y[self.y['id']==idx+self.first_test_id]['label']))
        else:
            return np.array(self.X[self.X['id']==idx+self.first_test_id].iloc[:,1:])