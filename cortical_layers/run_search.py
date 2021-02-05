import os
import numpy as np


n_filters = [16, 32, 64, 128]
sizes = [7, 15, 25, 49]
depths = [4,5,6,7,8,9,10]
decays = [0.001, 0.005, 0.0, 0.0005]
spenalties = [0.0, 0.0005, 0.001, 0.005]
lrs = [0.0005, 0.001, 0.002, 0.005, 0.01]
bsizes = [100, 200, 500, 1000]

kfold = 10
val_fold = 0
test_fold = 1

runs = 50 # number of runs

for i in range(runs):
	call = "THEANO_FLAGS='device=cuda,optimizer=fast_compile,optimizer_including=fusion' python train_simple_model_1path.py"
	call += ' -kf '+str(kfold)
	call += ' -vf '+str(val_fold)
	call += ' -tf '+str(test_fold)

	nf = np.random.choice(n_filters)
	fs = np.random.choice(sizes)
	d = np.random.choice(depths)
	wd = np.random.choice(decays)
	sp = np.random.choice(spenalties)
	lr = np.random.choice(lrs)
	bs = np.random.choice(bsizes)

	call += ' -nf '+str(nf)
	call += ' -fs '+str(fs)
	call += ' -d '+str(d)
	call += ' -wd '+str(wd)
	call += ' -sp '+str(sp)
	call += ' -lr '+str(lr)
	call += ' -bs '+str(bs)

	print(call)

	os.system(call)
