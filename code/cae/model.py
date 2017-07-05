"""
implements a simple autoencoder
Timo Flesch, 2017
"""

import os
import numpy      as np 
import tensorflow as tf 

from nntools.io     import saveMyModel
from nntools.layers import *

FLAGS = tf.app.flags.FLAGS

class myModel(object):
    """
    the autoencoder class
    """

    def __init__(self,
        dim_inputs   =          [None,28,28,1],
        dim_outputs  =          [None,28,28,1],
        n_hidden     =                 200,
        lr           =               0.005,
        optimizer    =           'RMSProp',
        nonlinearity =              'relu',
        is_trained   =              False):
        """ initializes the model with parameters """

        self.ITER           =     0
        self.session        =  None
        self.learning_rate  =    lr 
        
        self.dim_inputs     =  dim_inputs
        self.dim_outputs    = dim_outputs
        self.n_hidden       =    n_hidden
        
        self.nonlinearity   =             getattr(tf.nn,nonlinearity)
        self.initializer    = tf.truncated_normal_initializer(FLAGS.weight_init_mu, 
                                FLAGS.weight_init_std)        

        # dictionary for all parameters (weights + biases)
        self.params        =  {}

        self.init_done     = False

        if not(is_trained):
            # input placeholder
            with tf.name_scope('input'):
                self.x = tf.placeholder(tf.float32,[None,784],name='x_in')
                self.y_true = tf.placeholder(tf.float32,[None,784],name='y_true')
            
            # the neural network and label placeholder
            with tf.variable_scope('nnet'):
                self.nnet_builder()

            # optimizer
            with tf.name_scope('optimisation'):
                # loss function
                with tf.name_scope('loss-function'):
                    self.y_true_rs = tf.reshape(self.y_true,[-1,self.dim_inputs[1],self.dim_inputs[2],self.dim_inputs[3]],name="true_y")
                    self.loss = tf.reduce_mean(tf.pow(self.y_true_rs-self.y_hat,2),name="loss")

                # optimisation procedure
                with tf.name_scope('optimizer'):
                    self.optimizer = getattr(tf.train,optimizer+'Optimizer')(learning_rate=self.learning_rate)
                    self.train_step = self.optimizer.minimize(self.loss)


            # image reshaper
            with tf.name_scope('reshaper'):
                self.visIMG_orig  = self.x_img # tf.reshape(self.x,[1,28,28,1],name='reshape_input')
                self.visIMG_recon = self.y_hat # tf.reshape(self.y_hat,[1,28,28,1],name='reshape_output')

        else:
            self.init_done = True 


    def nnet_builder(self):
        """ creates neural network function approximator """
        
        # encoder 
        with tf.variable_scope('encoder'):
            self.x_img = tf.reshape(self.x,[-1,self.dim_inputs[1],self.dim_inputs[2],self.dim_inputs[3]])
            # for compatibility reasons, convert uint8 to float32 and rescale values
            self.x_imgF = tf.to_float(self.x_img) / 255.0
            # first convolutional layer
            self.l1_conv, self.params['l11_conv_weights'], self.params['l1_conv_biases'], self.params['l1_conv_shape'] = layer_conv2d(self.x_imgF,
                32,(5,5),name='l1_conv',nonlinearity=self.nonlinearity,stride=(2,2),padding='SAME')
            print(self.params['l1_conv_shape'])
            # second convolutional layer
            self.l2_conv, self.params['l12_conv_weights'], self.params['l2_conv_biases'], self.params['l2_conv_shape'] = layer_conv2d(self.l1_conv,
                64,(5,5),name='l2_conv',nonlinearity=self.nonlinearity,stride=(2,2),padding='SAME')
            print(self.params['l2_conv_shape'])

        # decoder 
        with tf.variable_scope('decoder'):            
            self.l3_transconv, self.params['l3_transconv_weights'], self.params['l3_transconv_biases'] = layer_transpose_conv2d(self.l2_conv,
                64,(5,5),shape=self.params['l2_conv_shape'],name='l3_transconv',nonlinearity=self.nonlinearity,stride=(2,2),padding='SAME')

            self.y_hat, self.params['l4_transconv_weights'], self.params['l4_transconv_biases'] = layer_transpose_conv2d(self.l3_transconv,
                32,(5,5),shape=self.params['l1_conv_shape'],name='l4_transconv',nonlinearity=self.nonlinearity,stride=(2,2),padding='SAME')

        return


    def init_graph_vars(self,sess,log_dir='.'):
        """ initializes graph variables """
        # set session
        self.session = sess
        # initialize all variables
        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)

        # define saver object AFTER variable initialisation
        self.saver = tf.train.Saver()

        # define summaries
        self.summaryTraining = tf.summary.scalar("loss_training",self.loss)        
        self.summaryTest     = tf.summary.scalar("loss_test",self.loss)        
        
        self.summaryImgOrig   = tf.summary.image("input image",self.visIMG_orig)
        self.summaryImgRecon  = tf.summary.image("reconstructed image",self.visIMG_recon)
        self.merged_summary = tf.summary.merge_all()


        self.writer = tf.summary.FileWriter(log_dir,self.session.graph)

        self.init_done = True


    def save_ckpt(self,modelDir):
        """ saves model checkpoint """
        # save the whole statistical model
        saved_ops = [('nnet',self.y_hat)] 

        saveMyModel(self.session,
                    self.saver,
                    saved_ops,
                    globalStep=self.ITER,
                    modelName=modelDir+FLAGS.model)


    def inference(self,x):
        """ forward pass of x through myModel """
        if x.ndim==1:
            x = np.expand_dims(x,axis=0)            
        y_hat = self.session.run(self.y_hat,feed_dict={self.x: x})
        return y_hat


    def train(self,x,y_true):
        """ training step """ 
        _,loss,summary_train = self.session.run([self.train_step,self.loss,self.summaryTraining],
            feed_dict={self.x: x,self.y_true: y_true})
        self.writer.add_summary(summary_train,self.ITER)
        self.ITER +=1 # count one iteration up
        return loss


    def eval(self,x,y_true):
        """ evaluation step """
        y_hat,loss,summary_test = self.session.run([self.y_hat,self.loss,self.summaryTest],
            feed_dict={self.x: x,self.y_true: y_true})
        self.writer.add_summary(summary_test,self.ITER)
        return loss

    def reconstruct(self,x):
        if x.ndim==1:
            x = np.expand_dims(x,axis=0)
        imgOrig,imgRecon = self.session.run([self.summaryImgOrig,self.summaryImgRecon],
            feed_dict={self.x: x})
        self.writer.add_summary(imgOrig,self.ITER)
        self.writer.add_summary(imgRecon,self.ITER)


