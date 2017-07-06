"""
implements a variational autoencoder
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
        n_hidden     =                 100,
        n_latent     =                  2,
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
        self.n_latent       =    n_latent
        
        if nonlinearity != None:
            self.nonlinearity   =  getattr(tf.nn,nonlinearity)
        else:
            self.nonlinearity   = nonlinearity

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
                self.phase  = tf.placeholder(tf.bool, name='training_phase')
            
            # the neural network and label placeholder
            with tf.variable_scope('nnet'):
                self.nnet_builder()

            # optimizer
            with tf.name_scope('optimisation'):
                # loss function
                with tf.name_scope('loss-function'):
                    # "normal" loss term:
                    #reconstruction_loss = 0.5* tf.reduce_sum(tf.pow(self.y_hat-self.y_true,2))
                    with tf.name_scope('xent_loss'):
                        self.y_hat =  tf.clip_by_value(self.y_hat, 1e-10, 1 - 1e-10,name='avoidNaN') # try to avoid nans
                        self.reconstruction_loss = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_hat, labels=self.y_true), reduction_indices=1)
                    # KL-divergence term:
                    with tf.name_scope('latent_loss'):
                        self.latent_loss  = -0.5 * tf.reduce_sum(1 + 2*self.layer_latent_logsd
                        -tf.square(self.layer_latent_mu)-tf.exp(2*self.layer_latent_logsd), 1)
                    with tf.name_scope('combined_loss'):
                        self.loss = tf.reduce_mean(self.reconstruction_loss+self.latent_loss,name="loss")

                # optimisation procedure
                with tf.name_scope('optimizer'):
                    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(self.update_ops):
                        self.optimizer = getattr(tf.train,optimizer+'Optimizer')(learning_rate=self.learning_rate)
                        self.train_step = self.optimizer.minimize(self.loss)


            # image reshaper
            with tf.name_scope('reshaper'):
                self.visIMG_orig  = tf.reshape(self.x,[1,28,28,1],name='reshape_input')
                self.visIMG_recon = tf.reshape(self.y_hat_logit,[1,28,28,1],name='reshape_output')

        else:
            self.init_done = True 


    def nnet_builder(self):
        """ creates neural network function approximator """
        
        # encoder 
        with tf.variable_scope('encoder'):
            # input to hidden layer 1, with batch normalisation
            self.layer_enc_hidden1, self.params['layer_enc_hidden1_weights'], self.params['layer_enc_hidden1_biases'] = layer_fc_bn(self.x,
                self.n_hidden,0.1,'layer_enc_hidden1','layer_enc_hidden1_bn',self.phase,self.initializer,self.nonlinearity)
            # hidden 1 to hidden 2, with batch normalisation
            self.layer_enc_hidden2, self.params['layer_enc_hidden2_weights'], self.params['layer_enc_hidden2_biases'] = layer_fc_bn(self.layer_enc_hidden1,
                self.n_hidden,0.1,'layer_enc_hidden2_fc','layer_enc_hidden2_bn',self.phase,self.initializer,self.nonlinearity)

        # sampler 
        with tf.variable_scope('latent'):
            with tf.variable_scope('latent_mu'):
                # latent mean, hidden to latent layer
                self.layer_latent_mu, self.params['layer_latent_mu_weights'], self.params['layer_latent_mu_biases'] = layer_fc(self.layer_enc_hidden2,
                    self.n_latent,0.1,'layer_latent_mu',self.initializer)

            with tf.variable_scope('latent_logsd'):
                # latent logsd, hidden to latent layer
                self.layer_latent_logsd, self.params['layer_latent_logsd_weights'], self.params['layer_latent_logsd_biases'] = layer_fc(self.layer_enc_hidden2,
                    self.n_latent,0.1,'layer_latent_logsd',self.initializer)

            with tf.variable_scope('latent_sampler'):
                # sample values from standard normal (first dim=batchsize)  
                epsilon = tf.random_normal(tf.stack([tf.shape(self.layer_enc_hidden2)[0], self.n_latent]), 0, 1, dtype = tf.float32)
                # reparametrization trick
                self.z = tf.add(self.layer_latent_mu, tf.multiply(tf.sqrt(tf.exp(self.layer_latent_logsd)), epsilon),name="latent_z")

            
        # decoder 
        with tf.variable_scope('decoder'):            
            # latent to hidden, with batch normalisation
            self.layer_dec_hidden1, self.params['layer_dec_hidden1_weights'], self.params['layer_dec_hidden1_biases'] = layer_fc_bn(self.z,
                self.n_hidden,0.1,'layer_dec_hidden1_fc','layer_dec_hidden1_bn',self.phase,self.initializer,self.nonlinearity) 
            # hidden to output, with batch normalisation
            self.y_hat_logit, self.params['layer_dec_hidden2_weights'], self.params['layer_dec_hidden2_biases'] = layer_fc_bn(self.layer_dec_hidden1,
                self.dim_outputs[1],0.1,'layer_dec_hidden2_fc','layer_dec_hidden2_bn',self.phase,self.initializer)

            # squashed
            with tf.variable_scope('sigmoid_squashing'):
                self.y_hat = tf.nn.sigmoid(self.y_hat_logit)

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
        self.summaryLossRecon = tf.summary.scalar("loss_recon",tf.reduce_mean(self.reconstruction_loss))
        self.summaryLossLatent = tf.summary.scalar("loss_latent",tf.reduce_mean(self.latent_loss))
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
        y_hat = self.session.run(self.y_hat,feed_dict={self.x: x,self.phase:0})
        return y_hat


    def train(self,x,y_true):
        """ training step """ 
        _,loss,summary_train,summary_recon,summary_latent = self.session.run([self.train_step,self.loss,self.summaryTraining,self.summaryLossRecon,self.summaryLossLatent],
            feed_dict={self.x: x,self.y_true: y_true,self.phase:1})
        self.writer.add_summary(summary_train,self.ITER)
        self.writer.add_summary(summary_recon,self.ITER)
        self.writer.add_summary(summary_latent,self.ITER)
        self.ITER +=1 # count one iteration up
        return loss


    def eval(self,x,y_true):
        """ evaluation step """
        y_hat,loss,summary_test = self.session.run([self.y_hat,self.loss,self.summaryTest],
            feed_dict={self.x: x,self.y_true: y_true,self.phase:0})
        self.writer.add_summary(summary_test,self.ITER)
        return loss

    def reconstruct(self,x):
        if x.ndim==1:
            x = np.expand_dims(x,axis=0)
        imgOrig,imgRecon = self.session.run([self.summaryImgOrig,self.summaryImgRecon],
            feed_dict={self.x: x,self.phase:0})
        self.writer.add_summary(imgOrig,self.ITER)
        self.writer.add_summary(imgRecon,self.ITER)

    def sample(self):
        """
        samples from the latent space and returns image output
        """
        # sample latent values
        z = self.session.run(tf.random_normal([1, self.n_latent]))
        if z.ndim==1:
            z = np.expand_dims(z,axis=0)
        # add to summary
        imgRecon = self.session.run(self.summaryImgRecon,feed_dict={self.z: z,self.phase:0})
        self.writer.add_summary(imgRecon,self.ITER)
        y_hat = self.session.run(self.y_hat,feed_dict={self.z: z,self.phase:0})
        return y_hat
        


