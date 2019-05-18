# VLAE
Code for Variational Laplace Autoencoders

## Prerequisite
- Pytorch 1.0 
- Tensorflow 1.13 (for tensorboard utility)

Install required packages
```
pip install -r requirements.txt
```

Make directories for datasets and model checkpoints
```
mkdir checkpoints
mkdir datasets
```

## Running Experiments
To run an experiment
```
python run.py --dataset=MNIST --output_dist=gaussian --model=VLAE --n_epochs=2000 --hidden_dim=500 --z_dim=50 --n_update=4
```
It will train the model for `n_epochs` and then automatically evaluate the log-likelihood using the best checkpoint using importance sampling. 

On the other hand, you can manually evaluate a checkpoint (note that you will have to use the same model setting to load the model)
```
python eval.py --dataset=MNIST --output_dist=gaussian --model=VLAE --hidden_dim=500 z_dim=50 --n_update=4 --checkpoint=<path_to_your_checkpoint>
```

## Features
Model save
Tensorboard

## Notes
- We have slightly changed the implementation for data scale normalization, so the scale of the log-likelihood results is bit different to those reported in the paper.
- The data is normalized so that the reconstruction error `torch.sum((x - mu)**2)` will be initially about d (not 1.0) where d is the data dimension (e.g. 784 for MNIST). The appendix of the paper was incorrect stated on this. 
- Data logit transform

## Results
Coming soon.

## Citation
Coming soon.
