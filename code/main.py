"""
Example implementations of various autoencoder flavours
and their application to the well-known and exciting MNIST dataset

Timo Flesch, 2017
"""

# external
import tensorflow as tf 
import numpy      as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
# custom
from ae.runAE    import    runAE # autoencoder
# from cae   import   runCAE # convolutional autoencoder
# from vae   import   runVAE # variational autoencoder
# from vcae  import  runVCAE # variational convolutional autoencoder
# from bvae  import  runbVAE # beta-VAE
# from bvcae import runbVCAE # beta-VCAE


# -- define flags --
FLAGS = tf.app.flags.FLAGS

# directories
tf.app.flags.DEFINE_string('data_dir',        './data/',  """ (string) data directory       """)

tf.app.flags.DEFINE_string('ckpt_dir', './checkpoints/', 
                            """ (string) checkpoint directory """)

tf.app.flags.DEFINE_string('log_dir','./log/', """ (string) log/summary directory """)


# dataset
tf.app.flags.DEFINE_integer('n_samples_train', 50000, 
                           """ (int) number of training samples """)

tf.app.flags.DEFINE_integer('n_samples_test',  10000,  
                           """ (int) number of test samples     """)


# model
tf.app.flags.DEFINE_string('model',                'ae', 
                            """ (string)  chosen model         """)

tf.app.flags.DEFINE_bool('do_training',               1, 
                            """ (boolean) train or not         """)

tf.app.flags.DEFINE_float('weight_init_mu',         0.0, 
                            """ (float)   initial weight mean  """)

tf.app.flags.DEFINE_float('weight_init_std',        .05, 
                            """ (float)   initial weight std   """)

tf.app.flags.DEFINE_string('nonlinearity',       'relu', 
                            """ (string)  activation function  """)


# training
tf.app.flags.DEFINE_float('learning_rate',     0.001, 
                            """ (float)   learning rate              """)

tf.app.flags.DEFINE_integer('n_training_episodes',   20, 
                            """ (int)    number of training episodes  """)

tf.app.flags.DEFINE_integer('n_training_batches',   int(FLAGS.n_samples_train/FLAGS.n_training_episodes), 
                            """    number of training batches per ep  """)

tf.app.flags.DEFINE_integer('display_step',         1, 
                            """(int) episodes until training log """)

tf.app.flags.DEFINE_integer('batch_size',         128, 
                            """ (int)     training batch size        """)

tf.app.flags.DEFINE_string('optimizer',       'RMSProp', 
                            """ (string)   optimisation procedure     """)




def normalizeData(X_train, X_test):
    """ helper function, normalizes input """
    preproc = prep.StandardScaler().fit(X_train)
    x_train = preproc.transform(X_train)
    x_test = preproc.transform(X_test)
    return x_train, x_test


def shuffleData(x,y=None):
    """ helper function, shuffles data """
    ii_shuff = np.random.permutation(x.shape[0])
    # shuffle data
    x = x[ii_shuff,:]
    if y is not None:
        y = y[ii_shuff,:]
    return x, y


def main(argv=None): 
    """ here starts the magic """
    # import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True) 
    # scale data appropriately
    x_train, x_test = normalizeData(mnist.train.images, mnist.test.images)
    x_train,_ = shuffleData(x_train)
    x_test,_  = shuffleData(x_test)

    # run selected model
    if FLAGS.model=='ae':
        runAE(x_train,x_test)
    elif FLAGS.model=='cae':
        runCAE(x_train,x_test)
    elif FLAGS.model=='vae':
        runVAE(x_train,x_test)
    elif FLAGS.model=='vcae':
        runVCAE(x_train,x_test)
    elif FLAGS.model=='bvae':
        runbVAE(x_train,x_test)
    elif FLAGS.model=='bvcae':
        runbVCAE(x_train,x_test)




if __name__ == '__main__':
    """ take care of flags on load """
    tf.app.run()
