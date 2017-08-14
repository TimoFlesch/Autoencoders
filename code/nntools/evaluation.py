import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 

FLAGS = tf.app.flags.FLAGS

def evalModel(sess,nnet,x_train):
    """
    evaluates trained model
    """
    x_sample = x_train[0:40,:]
    # plot a few reconstructions
    plotReconstructions(sess,nnet,x_sample)
    # plot traversal of latent space, if vae:
    if FLAGS.model == 'vae' or FLAGS.model == 'cvae' or FLAGS.model == 'bvae' or FLAGS.model == 'cbvae':
        plotLatentSpace(sess,nnet)
    
    

def plotReconstructions(sess,nnet,x_sample):
    """
    plots reconstructions of inputs
    code heavily inspired by https://jmetzen.github.io/2015-11-27/vae.html
    """

    # x_reconstruct,_ = nnet.inference(x_sample)
    x_reconstruct = nnet.inference(x_sample)
    
    plt.figure(figsize=(12, 6))
    for i in range(5):

        plt.subplot(2, 5, i+1)
        fig = plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")        
        plt.title("Input Image")
        # plt.colorbar()
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(2, 5, i+6)
        fig = plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstructed Image")
        # plt.colorbar()
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

    

def plotLatentSpace(sess,nnet):
    """
    traverses the latent space and plots generated 
    samples
    code heavily inspired by https://jmetzen.github.io/2015-11-27/vae.html
    """
    
    num_x = num_y = 25
    x_vals = np.linspace(-5, 5, num_x)
    y_vals = np.linspace(-5, 5, num_y)

    # create space for image
    img_space = np.empty((28*num_y, 28*num_x))
    for i, yi in enumerate(x_vals):
        for j, xi in enumerate(y_vals):
            # create 100 samples
            z_mu = np.array([[xi, yi]]*100)            
            x_mean = nnet.sample(z_mu)
            # arrange the mean of the samples in image space
            img_space[(num_x-i-1)*28:(num_x-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(10, 10))        
    # Xi, Yi = np.meshgrid(x_vals, y_vals)
    fig = plt.imshow(img_space, origin="upper", cmap="gray")
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title('Traversing the Latent Space')
    plt.tight_layout()
    plt.show()





