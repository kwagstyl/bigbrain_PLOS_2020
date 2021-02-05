import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model', help='Model used for the experiments', default="simple", type=str)
args = parser.parse_args()

# where are the results stored
base_path = '/data/lisatmp4/cucurulg/cortical_layers/6cortical_layers_all/'
filter_kfold = 'kfold=10_val=0_test=1'
filter_simple = args.model+'_model'

experiments = [exp for exp in os.listdir(base_path) if filter_kfold in exp]
experiments = [exp for exp in experiments if filter_simple in exp]

results = []

for exp in experiments:
    exp_path = os.path.join(base_path, exp)

    if 'fcn1D_errors_best.npz' in os.listdir(exp_path):
        errors_best = np.load(os.path.join(exp_path, 'fcn1D_errors_best.npz'))

        res = {
            'name': exp,
            'best_jacc_valid': errors_best['jacc_valid'][-1],
            'best_acc_valid': errors_best['acc_valid'][-1],
            'best_acc_train': errors_best['acc_train'][-1],
        }
    else: # there is no best because the exp was bad
        errors_last = np.load(os.path.join(exp_path, 'fcn1D_errors_last.npz'))

        res = {
            'name': exp,
            'best_jacc_valid': errors_last['jacc_valid'][-1],
            'best_acc_valid': errors_last['acc_valid'][-1],
            'best_acc_train': errors_last['acc_train'][-1]
        }

    results.append(res)

results = sorted(results, key=lambda k: k['best_jacc_valid'], reverse=True)

print("jacc valid, acc valid, acc train")
for res in results:
    print('%f, %f, %f, %s' % (res['best_jacc_valid'], res['best_acc_valid'], res['best_acc_train'], res['name']))
print('Done!')
