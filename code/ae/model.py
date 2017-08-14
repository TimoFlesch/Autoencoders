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
        dim_inputs   =          [None,784],
        dim_outputs  =          [None,784],
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
                self.x = tf.placeholder(tf.float32,self.dim_inputs,name='x_in')
                self.y_true = tf.placeholder(tf.float32,self.dim_outputs,name='y_true')
            
            # the neural network and label placeholder
            with tf.variable_scope('nnet'):
                self.nnet_builder()

            # optimizer
            with tf.name_scope('optimisation'):
                # loss function
                with tf.name_scope('loss-function'):
                    self.loss = tf.reduce_mean(tf.pow(self.y_true-self.y_hat,2),name="loss")

                # optimisation procedure
                with tf.name_scope('optimizer'):
                    self.optimizer = getattr(tf.train,optimizer+'Optimizer')(learning_rate=self.learning_rate)
                    self.train_step = self.optimizer.minimize(self.loss)


            # image reshaper
            with tf.name_scope('reshaper'):
                self.visIMG_orig  = tf.reshape(self.x,[1,28,28,1],name='reshape_input')
                self.visIMG_recon = tf.reshape(self.y_hat,[1,28,28,1],name='reshape_output')

        else:
            self.init_done = True 


    def nnet_builder(self):
        """ creates neural network function approximator """
        
        # encoder 
        with tf.variable_scope('encoder'):
            self.layer1, self.params['layer1_weights'], self.params['layer1_biases'] = layer_fc(self.x,
                self.n_hidden,0.0,'layer1',self.initializer,self.nonlinearity)

            self.layer2, self.params['layer2_weights'], self.params['layer2_biases'] = layer_fc(self.layer1,
                self.n_hidden,0.0,'layer2',self.initializer,self.nonlinearity)

            self.layer3, self.params['layer3_weights'], self.params['layer3_biases'] = layer_fc(self.layer2,
                10,0.0,'layer3',self.initializer,self.nonlinearity)
        
        # decoder 
        with tf.variable_scope('decoder'):            
            self.layer4, self.params['layer4_weights'], self.params['layer4_biases'] = layer_fc(self.layer3,
                self.n_hidden,0.0,'layer4',self.initializer,self.nonlinearity)
                    
            self.layer5, self.params['layer5_weights'], self.params['layer5_biases'] = layer_fc(self.layer4,
                self.dim_outputs[1],0.0,'layer5',self.initializer,self.nonlinearity) 

            self.y_hat, self.params['y_hat_weights'], self.params['y_hat_biases'] = layer_fc(self.layer5,
                self.dim_outputs[1],0.0,'y_hat',self.initializer,tf.nn.sigmoid) 

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


