from train_test_kfold import train_test_kfold
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='launch some trainings')
parser.add_argument('resolutions_set',type=int, help='resolution set (from 0-2)')
args=parser.parse_args()

res_set = args.resolutions_set

resolutions=[[20,40],[100,200],[300,400,1000]]
kfold=10
max_patience=50
#setting k folds. Validation set rotates with test set
pairs=np.array([(np.arange(10)+1)%10,np.arange(10)]).T
for resolution in resolutions[res_set]:
    print(resolution)
 #   pairs=[]
    for pair in pairs:
        train_test_kfold(resolution,kfold,pair[0],pair[1], max_patience=max_patience)




#    while len(pairs)<n_iterations:
#        pair=list(np.random.choice(kfold,2,replace=False))
#        if pair not in pairs:
#            pairs.append(pair)
#            train_test_kfold(resolution,kfold,pair[0],pair[1], max_patience=max_patience)
