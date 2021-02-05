from train_test_kfold import train_test_kfold, test_full_model
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='launch some trainings')
parser.add_argument('resolutions_set',type=int, help='resolution set (from 0-2)')
args=parser.parse_args()

res_set = args.resolutions_set

resolutions=[[20,40],[100,200],[300,400,1000]]

kfold=10
max_patience=50
for resolution in resolutions[res_set]:
    print(resolution)
#    train_test_kfold(resolution,kfold,0,None, max_patience=max_patience)
    test_full_model(resolution,kfold, 0, None, max_patience=max_patience)
        



