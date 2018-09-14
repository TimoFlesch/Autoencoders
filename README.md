# Implementation of Various Autoencoders
Contains tensorflow code to run
- [Autoencoder](code/ae)
- [Convolutional Autoencoder](code/cae)
- [Variational Autoencoder](code/vae)
- [Convolutional Variational Autoencoder](code/cvae)

All models are trained on the famous and boring MNIST dataset (suboptimal choice for the bVAE) 

### Requirements
Python 3.6 and tensorflow 1.0  

### Structure
Everything you need is contained in the folder [code](code) 
1. Model Selection  
Change the flags in [main.py](code/main.py)  
2. Model Modification  
Each model consists of a specification file  ([e.g. this one for the AE](code/ae/model.py)) and a runner runXY.py ([e.g. this one for the AE](code/ae/runAE.py)). Change the model architecture in the corresponding model.py file.
Some auxiliar functions are defined in the [nntools](code/nntools) subfolder. 

3. Training a model  
Call the respective runXY.py  
4. Evaluating a model  
Change the training flag in main.py to False and call the respective runXY.py  

### Monitoring
The code creates scalar and image summaries during training.

##### Local
to start a monitoring session:
 ```
  tensorboard --logdir=path/to/log-directory --port=6006
 ```

 then, navigate in browser to:

 ```
 0.0.0.0:6006
 ```
to visualize the summaries and monitor the training process

##### Remote
to start monitoring session:
```
ssh -L 16006:127.0.0.1:6006 user@remote.machine
tensorboard --logdir=path/to/log-directory --port=6006
```
then, navigate in browser of local machine (whilst ssh tunnel is open) to:
```
127.0.0.1:16006
```

