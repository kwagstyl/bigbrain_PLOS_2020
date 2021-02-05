# Cortical_layers files

## Models

- fcn_1D_general.py : Defines FCN_1D model. First model tried for segmentation
- simple_model.py : Defines "simple_model" (convolution without pooling, with multiple paths)
- simple_model_1path.py : Defines simple model but with 1 path only
- classif_model.py : Defines classification models (Dense_model class and classif_model class)

## Training Notebooks

- train_simple_model.ipynb : Notebook to train simple_model for cortical layers dataset
- train_simple_model_1path.ipynb : Notebook to train simple_model_1path for cortical layers dataset

## Training python files (to launch directly with python file.py)

- train_fcn1D.py : Train fcn1D for cortical layers dataset
- train_simple_model.py : Train simple_model for cortical layers dataset
- train_simple_model_1path.py : Train simple_model_1path for cortical layers dataset
- train_classif_model.py : Train classif_model for parcellation dataset

## Testing Notebooks

- test_simple_model.ipynb : Output results from simple_model (cortical layers dataset)
- test_classif_model.ipynb : Output results from classif_model (parcellation dataset)


## Utils / Other files

- error_plots.ipynb : Jupyter Notebook to visualize the error/accuracy during training on the training or validation set
- visualize.ipynb : Script to visualize the rays (intensity profiles and ground truth labels)
- metrics.py : Some metrics used
- profile_functions.py : Script that transform intensity profiles to indices to profiles
- tsne_visualization.ipynb : TSNE visualization for input rays/ground truth segmentation
- hyper_parameters.ods : Hyperparameter search and results for simple model (4 cortical layers segmentation)



