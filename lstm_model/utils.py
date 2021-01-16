import os
import csv
import pandas as pd


def load_trainfiles(data_dir):
    train=pd.read_csv(data_dir+'train_features.csv')
    train_labels=pd.read_csv(data_dir+'train_labels.csv')

    return train, train_labels
    
def load_testfiles(data_dir):
    test=pd.read_csv(data_dir+'test_features.csv')
    submission=pd.read_csv(data_dir+'sample_submission.csv')

    return test,submission
