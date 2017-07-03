"""
runner function for autoencoder 
Timo Flesch, 2017
"""
import numpy as np
import tensorflow as tf

from ae.model   import         myModel
from trainer    import      trainModel
from evaluation import       evalModel
from tfio.io    import     loadMyModel

FLAGS = tf.app.flags.FLAGS

def runAE(x_train,x_test):
    # checkpoint run model folder
    ckpt_dir_run = FLAGS.ckpt_dir + 'model_' + FLAGS.model

    if not(tf.gfile.Exists(ckpt_dir_run)):
        tf.gfile.MakeDirs(ckpt_dir_run) 

    with tf.Session() as sess:      

        if FLAGS.do_training:
            nnet = myModel(lr           = FLAGS.learning_rate,
                           optimizer    =     FLAGS.optimizer,
                           nonlinearity =  FLAGS.nonlinearity,
                           )
            print("Now training Autoencoder")
            # initialize all variables
            nnet.init_graph_vars(sess,log_dir=FLAGS.log_dir+'model_'+FLAGS.model)
            # train model
            results = trainModel(sess,nnet,x_train,x_test,
                n_steps=FLAGS.n_training_steps,
                batch_size=FLAGS.batch_size,
                model_dir= ckpt_dir_run)

        else:           
            nnet = myModel(is_trained=True)
            print("Now evaluating Autoencoder")
            ops = loadModel(sess,ckpt_dir_run)
            nnet.y_ = ops[0]

            results = evalModel(sess,nnet)





    



