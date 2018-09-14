"""
implements a beta-variational autoencoder
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
    the variational autoencoder class
    """

    def __init__(self,
        dim_inputs   =          [None,784],
        dim_outputs  =          [None,784],
        n_hidden     =                 200,
        n_latent     =                   2,
        lr           =               0.001,
        optimizer    =                None,
        nonlinearity =                None,
        is_trained   =              False):
        """ initializes the model with parameters """

        self.ITER           =     0
        self.session        =  None
        self.learning_rate  =    lr 
        
        
        self.dim_inputs     =  dim_inputs
        self.dim_outputs    = dim_outputs
        self.n_hidden       =    n_hidden
        self.n_latent       =    n_latent
        
        if nonlinearity != None:
            self.nonlinearity   =  getattr(tf.nn,nonlinearity)
        else:
            self.nonlinearity   = nonlinearity

        self.initializer    = tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False)      # tf.contrib.layers.xavier_initializer() #

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
               self.optimizer_builder(optimizer)

         

        else:
            self.init_done = True 


    def nnet_builder(self):
        """ creates vae function approximator """
        
        # encoder 
        with tf.variable_scope('encoder'):
            # input to hidden layer 1
            self.layer_enc_hidden1, self.params['layer_enc_hidden1_weights'], self.params['layer_enc_hidden1_biases'] = layer_fc(self.x,
                self.n_hidden,0.5,'layer_enc_hidden1',self.initializer,self.nonlinearity)

            self.layer_enc_hidden2, self.params['layer_enc_hidden2_weights'], self.params['layer_enc_hidden2_biases'] = layer_fc(self.layer_enc_hidden1,
                self.n_hidden,0.5,'layer_enc_hidden2',self.initializer,self.nonlinearity)


        # latent sampler
        with tf.variable_scope('latent'):       
            self.layer_z_mu, self.params['layer_z_mu_weights'], self.params['layer_z_mu_biases'] = layer_fc(self.layer_enc_hidden1,self.n_latent,0.5,'z_mu',self.initializer)

            self.layer_z_logvar, self.params['layer_z_logvar_weights'], self.params['layer_z_logvar_biases'] = layer_fc(self.layer_enc_hidden1,self.n_latent,0.5,'z_logvar',self.initializer)

            self.epsilon = tf.random_normal(tf.stack([tf.shape(self.layer_enc_hidden1)[0], self.n_latent]),mean=0.0,stddev=1.0, dtype = tf.float32)
            self.layer_z = tf.add(self.layer_z_mu,tf.multiply(tf.sqrt(tf.exp(self.layer_z_logvar)),self.epsilon))   

            
        # decoder 
        with tf.variable_scope('decoder'):                        
            self.layer_dec_hidden1, self.params['layer_dec_hidden1_weights'], self.params['layer_dec_hidden1_biases'] = layer_fc(self.layer_z,
                    self.n_hidden,0.5,'layer_dec_hidden1',self.initializer,self.nonlinearity)
            
            self.y_hat, self.params['y_hat_weights'], self.params['y_hat_biases'] = layer_fc(self.layer_dec_hidden1,self.dim_outputs[1],0.5,'y_hat',nonlinearity=tf.nn.sigmoid)
        
        return


    def optimizer_builder(self,optimizer):
        """
        creates optimiser 
        """
         # loss function
        with tf.name_scope('loss-function'):
            # "normal" loss term:
          
            with tf.name_scope('reconstruction_loss'):
                # self.reconstr_loss = tf.reduce_sum(tf.pow(self.y_true-self.y_hat,2),1)     
                self.reconstr_loss = -tf.reduce_sum(self.y_true*tf.log(1e-10+self.y_hat)+(1-self.y_true)*tf.log(1e-10+1-self.y_hat),1)
          
            with tf.name_scope('latent_loss'):
                self.latent_loss  = 0* -0.5 * tf.reduce_sum(1 + self.layer_z_logvar - tf.square(self.layer_z_mu) - tf.exp(self.layer_z_logvar),1)
            with tf.name_scope('combined_loss'):
                self.total_loss  = tf.reduce_mean(self.reconstr_loss+self.latent_loss,name="loss")

        # optimisation procedure
        with tf.name_scope('optimizer'):
            # old approach, based on batch norm:
            # self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(self.update_ops):
            #     self.optimizer = getattr(tf.train,optimizer+'Optimizer')(learning_rate=self.learning_rate)
            #     self.train_step = self.optimizer.minimize(self.total_loss)
            self.optimizer = getattr(tf.train,optimizer+'Optimizer')(learning_rate=self.learning_rate)
            self.train_step = self.optimizer.minimize(self.total_loss)
        return


    def init_graph_vars(self,sess,log_dir='.'):
        """ 
        initializes graph variables and adds summaries
        """
        # set session
        self.session = sess
        # initialize all variables
        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)
        # define saver object AFTER variable initialisation
        self.saver = tf.train.Saver()
        # set up all the summaries
        # self.setup_tensorboard_summaries(log_dir)
        self.init_done = True

    def setup_tensorboard_summaries(self,log_dir):
        """
        define summaries
        """            
       
        self.summaryLossRecon = tf.summary.scalar("loss_recon",tf.reduce_mean(self.reconstruction_loss))
        self.summaryLossLatent = tf.summary.scalar("loss_latent",tf.reduce_mean(self.latent_loss))
        self.summaryTraining = tf.summary.scalar("loss_training",self.total_loss)        
        self.summaryTest     = tf.summary.scalar("loss_test",self.total_loss)        
                
        self.merged_summary = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(log_dir,self.session.graph)


    def save_ckpt(self,modelDir):
        """ 
        saves model checkpoint 
        """
        # save the whole statistical model
        saved_ops = [('nnet',self.y_hat)] 

        saveMyModel(self.session,
                    self.saver,
                    saved_ops,
                    globalStep=self.ITER,
                    modelName=modelDir+FLAGS.model)


    def train(self,x,y_true):
        """ training step """ 
        # print(str(x.shape()) + ',  ' + str(y_true.shape()))
        _,loss = self.session.run([self.train_step,self.total_loss],feed_dict={self.x:x,
                                                                        self.y_true:y_true})
        return loss


    def inference(self,x):
        """ 
        forward pass of x through myModel,
        return decoder output and latent output
        """
        if x.ndim==1:
            x = np.expand_dims(x,axis=0)            
        y_hat,z_hat = self.session.run([self.y_hat,self.layer_z],feed_dict={self.x:x})
        # y_hat = self.session.run(self.y_hat,feed_dict={self.x:x})
        # return y_hat,z_hat
        return y_hat



    def eval(self,x,y_true):
        """ 
        evaluation step, returns only calculated loss 
        """
        loss = self.session.run(self.total_loss,feed_dict={self.x:x, self.y_true:y_true})        
        return loss



    def sample(self,z=None):
        """
        samples from the latent space and returns image output
        if z is provided, the generator uses this parameter.
        otherwise, a z is sampled from a random normal distribution
        """
        # sample latent values
        if z is None:
            z = self.session.run(tf.random_normal([1, self.n_latent]))
        if z.ndim==1:
            z = np.expand_dims(z,axis=0)
        # retrieve value
        y_hat = self.session.run(y_hat,feed_dict={self.layer_z:z})
        return y_hat
        


    def reconstruct(self,x):
        """
        slilghtly unnecessary function for logging purposes
        adds reconstructed images to tensorboard summary  
        """
        if x.ndim==1:
            x = np.expand_dims(x,axis=0)
        # self.writer.add_summary(imgOrig,self.ITER)
        # self.writer.add_summary(imgRecon,self.ITER)
        return
