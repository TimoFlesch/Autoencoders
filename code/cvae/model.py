"""
implements a convolutional variational autoencoder
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
    the convolutional autoencoder class
    """

    def __init__(self,
        dim_inputs   =          [None,28,28,1],
        dim_outputs  =          [None,28,28,1],
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
        
        self.nonlinearity   =             getattr(tf.nn,nonlinearity)
        self.initializer    = tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False) 
        #tf.truncated_normal_initializer(FLAGS.weight_init_mu, FLAGS.weight_init_std)        

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
               self.optimizer_builder(optimizer)

            # image reshaper
            with tf.name_scope('reshaper'):
                self.visIMG_orig  = tf.reshape(self.x_img,[1,28,28,1],name='reshape_input')
                self.visIMG_recon =tf.reshape(self.y_hat,[1,28,28,1],name='reshape_output')

        else:
            self.init_done = True 


    def nnet_builder(self):
        """ creates neural network function approximator """
        
        # encoder 
        with tf.variable_scope('encoder'):
            self.x_img = tf.reshape(self.x,[-1,self.dim_inputs[1],self.dim_inputs[2],self.dim_inputs[3]])
            # for compatibility reasons, convert uint8 to float32 and rescale values            
            # first convolutional layer
            self.l1_conv, self.params['l11_conv_weights'], self.params['l1_conv_biases'], self.params['l1_conv_shape'] = layer_conv2d(self.x_img,
                16,(5,5),name='l1_conv',nonlinearity=self.nonlinearity,stride=(2,2),padding='SAME')
            print(self.params['l1_conv_shape'])
            # second convolutional layer
            self.l2_conv, self.params['l12_conv_weights'], self.params['l2_conv_biases'], self.params['l2_conv_shape'] = layer_conv2d(self.l1_conv,
                32,(3,3),name='l2_conv',nonlinearity=self.nonlinearity,stride=(2,2),padding='SAME')
            print(self.params['l2_conv_shape'])
            # third convolutional layer
            self.l3_conv, self.params['l13_conv_weights'], self.params['l3_conv_biases'], self.params['l3_conv_shape'] = layer_conv2d(self.l2_conv,
                32,(3,3),name='l3_conv',nonlinearity=self.nonlinearity,stride=(2,2),padding='SAME')
            print(self.params['l3_conv_shape'])

        # latent sampler
        with tf.variable_scope('latent'):  
            self.enc_flatten = layer_flatten(self.l3_conv)     
            self.layer_z_mu, self.params['layer_z_mu_weights'], self.params['layer_z_mu_biases'] = layer_fc(self.enc_flatten,self.n_latent,0.5,'z_mu',self.initializer)

            self.layer_z_logvar, self.params['layer_z_logvar_weights'], self.params['layer_z_logvar_biases'] = layer_fc(self.enc_flatten,self.n_latent,0.5,'z_logvar',self.initializer)

            self.epsilon = tf.random_normal(tf.stack([tf.shape(self.enc_flatten)[0], self.n_latent]),mean=0.0,stddev=1.0, dtype = tf.float32)
            self.layer_z = tf.add(self.layer_z_mu,tf.multiply(tf.sqrt(tf.exp(self.layer_z_logvar)),self.epsilon))   
            

        # decoder 
        with tf.variable_scope('decoder'):   
            enc_shape = self.l3_conv.get_shape().as_list()
            self.z_reshaped, self.params['z_reshaped_weights'],self.params['z_reshaped_biases'] =  layer_fc(self.layer_z,
                    self.enc_flatten.get_shape().as_list()[1],0.5,'z_reshaped',self.initializer,self.nonlinearity) 
            self.z_img  =  tf.reshape(self.z_reshaped,[-1,enc_shape[1],enc_shape[2],enc_shape[3]],name="z_img")    

            self.l4_transconv, self.params['l4_transconv_weights'], self.params['l4_transconv_biases'], self.params['l4_transconv_shape'] = layer_transpose_conv2d(self.z_img,
                32,(3,3),shape=self.params['l3_conv_shape'],name='l4_transconv',nonlinearity=self.nonlinearity,stride=(2,2),padding='SAME')
            print(self.params['l4_transconv_shape'])

            self.l5_transconv, self.params['l5_transconv_weights'], self.params['l5_transconv_biases'], self.params['l5_transconv_shape'] = layer_transpose_conv2d(self.l4_transconv,
                32,(3,3),shape=self.params['l2_conv_shape'],name='l5_transconv',nonlinearity=self.nonlinearity,stride=(2,2),padding='SAME')
            print(self.params['l5_transconv_shape'])

            self.l6_transconv, self.params['l6_transconv_weights'], self.params['l6_transconv_biases'], self.params['l6_transconv_shape'] = layer_transpose_conv2d(self.l5_transconv,
                16,(5,5),shape=self.params['l1_conv_shape'],name='l6_transconv',nonlinearity=tf.nn.sigmoid,stride=(2,2),padding='SAME')

            self.y_hat = layer_flatten(self.l6_transconv)
            

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
        """ initializes graph variables """
        # set session
        self.session = sess
        # initialize all variables
        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)

        # define saver object AFTER variable initialisation
        self.saver = tf.train.Saver()

        # define summaries
        self.summaryTraining = tf.summary.scalar("loss_training",self.total_loss)        
        self.summaryTest     = tf.summary.scalar("loss_test",self.total_loss)        
        
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
        # print(np.asarray(y_hat).shape)
        return y_hat

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
        y_hat = self.session.run(self.y_hat,feed_dict={self.layer_z:z})
        return y_hat

    def train(self,x,y_true):
        """ training step """ 
        _,loss,summary_train = self.session.run([self.train_step,self.total_loss,self.summaryTraining],
            feed_dict={self.x: x,self.y_true: y_true})
        self.writer.add_summary(summary_train,self.ITER)
        self.ITER +=1 # count one iteration up
        return loss


    def eval(self,x,y_true):
        """ evaluation step """
        y_hat,loss,summary_test = self.session.run([self.y_hat,self.total_loss,self.summaryTest],
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


