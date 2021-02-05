#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=140:00:00

# set name of job
#SBATCH --job-name=test_train_bigbrain_res

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=konrad.wagstyl@gmail.com

# error output file
#SBATCH --error=test.%J.err

#completed output file
#SBATCH --output=test.%J.out

# run the application
dev=2 #$1
source activate deep
export CUDA_VISIBLE_DEVICES=0 # $dev
THEANO_FLAGS="device=cuda0,floatX=float32" python kfold_resolutions.py  $dev
mail -s ""$dev" is complete" kw350@cam.ac.uk <<< ""$dev" is complete"
