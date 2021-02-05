# Code for BigBrain PLOS 2020 layers paper
Code and scripts used for BigBrain cortical layers paper:

Wagstyl, K., Larocque, S., Cucurull, G., Lepage, C., Cohen, J. P., Bludau, S., Palomero-Gallagher, N., Lewis, L. B., Funck, T., Spitzer, H., Dickscheid, T., Fletcher, P. C., Romero, A., Zilles, K., Amunts, K., Bengio, Y., & Evans, A. C. (2020). 

BigBrain 3D atlas of cortical layers: Cortical and laminar thickness gradients diverge in sensory and motor cortices. PLoS Biology, 18(4), e3000678.
https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678


## WARNING
This repository has not been cleaned and has a whole host of horrible dependencies.
It is intended to give a taste of how everything was done. If you want to recreate a particular analysis and need code that appears to be missing, please do get in touch.


For much tidier examples of how to use the output cortical layer data please see:
https://github.com/kwagstyl/cortical_layers_tutorial

Raw BigBrain data and derivatives can be downloaded from:
ftp://bigbrain.loris.ca/BigBrainRelease.2015/


scripts/ BigBrain and mesh processing scripts,
scripts/notebooks: analyses used to generate figures and results for paper
scripts/histology_2d 2D histology processing scripts used to generate training data from manually annotated 2D histological sections
scripts/surfaces mesh operations pre and post layer segmentation
scripts/volume_processing BigBrain volume operations - smoothing volumes and extracting intensity profiles
cortical_layers/ contains deep learning profile code. Another warning is that this is dependent on Theano and lasagne, packages that are no longer maintained.
test_train_full.sh is the main script used to run the training and testing
