"""
runner function for autoencoder 
Timo Flesch, 2017
"""
# external
import numpy      as np
import tensorflow as tf

# custom
from cae.model          import         myModel
from nntools.trainer    import      trainModel
from nntools.evaluation import       evalModel
from nntools.io         import     loadMyModel

FLAGS = tf.app.flags.FLAGS

def runCAE(x_train,x_test):
    # checkpoint run model folder
    ckpt_dir_run = FLAGS.ckpt_dir + 'model_' + FLAGS.model
    log_dir_run  = FLAGS.log_dir+'model_'+FLAGS.model

    if not(tf.gfile.Exists(ckpt_dir_run)):
        tf.gfile.MakeDirs(ckpt_dir_run) 

    if not(tf.gfile.Exists(log_dir_run)):
        tf.gfile.MakeDirs(log_dir_run) 

    with tf.Session() as sess:      

        if FLAGS.do_training:
            nnet = myModel(lr           = FLAGS.learning_rate,
                           optimizer    =     FLAGS.optimizer,
                           nonlinearity =  FLAGS.nonlinearity,
                           )
            print("Now training Convolutional Autoencoder, LR: {} , EPs: {}, BS: {}"
                .format(FLAGS.learning_rate,FLAGS.n_training_episodes, FLAGS.batch_size))
            # initialize all variables
            nnet.init_graph_vars(sess,log_dir=log_dir_run)
            # train model
            results = trainModel(sess,nnet,x_train,x_test,
                n_episodes  =  FLAGS.n_training_episodes,
                n_batches   =   FLAGS.n_training_batches,
                batch_size  =           FLAGS.batch_size,
                model_dir   =               ckpt_dir_run)

        else:           
            nnet = myModel(is_trained=True)
            print("Now evaluating Convolutional Autoencoder")
            ops = loadMyModel(sess,['nnet'],ckpt_dir_run)
            print(ops)
            nnet.y_hat = ops[0]

            results = evalModel(sess,nnet)

        return results





    



