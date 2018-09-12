import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import os

class liteRBM:
    def __init__(self, n_visible, n_hidden, 
                 lr=0.01, momentum=0.95):
        # num. of visible units
        self.n_visible = n_visible
        # num. of hidden units
        self.n_hidden = n_hidden
        # learning rate and momentum
        self.lr = lr
        self.momentum = momentum
        # visible (data) layer
        self.v = tf.placeholder(tf.float32, [None, self.n_visible])
        # weight matrix
        self.W = tf.Variable(tf.truncated_normal([n_visible, n_hidden],
                                                 stddev=0.1),
                                                 dtype=tf.float32)
        # visible bias
        self.v_bias = tf.Variable(tf.zeros([self.n_visible]), 
                                  dtype=tf.float32)
        # hidden bias
        self.h_bias = tf.Variable(tf.zeros([self.n_hidden]), 
                                  dtype=tf.float32)
        # initialize gradients variables for Weights and biases
        self.grad_W = tf.Variable(tf.truncated_normal([n_visible, n_hidden],
                                                 stddev=0.1),
                                                 dtype=tf.float32)
        self.grad_v_bias = tf.Variable(tf.zeros([self.n_visible]), 
                                       dtype=tf.float32)
        self.grad_h_bias = tf.Variable(tf.zeros([self.n_hidden]),
                                       dtype=tf.float32)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.init_graph()
        self.saver = tf.train.Saver({'W': self.W,
                                     'v_bias': self.v_bias,
                                     'h_bias': self.h_bias})
        self.free_energy = - (tf.matmul(self.v, tf.reshape(self.v_bias, [-1,1])) + \
                              tf.reshape(tf.reduce_sum
                                        (tf.log
                                        (tf.exp(tf.matmul( self.v, self.W ) + \
                                                self.h_bias) + 1), 1), [1,-1]))
        
    # main optimization process
    def init_graph(self):
        # visible to hidden pass
        v_2_h_1 = tf.nn.sigmoid(tf.matmul(self.v, self.W)
                                + self.h_bias)
        # hidden to visible pass
        # sample h ~ P(h|v) and feed it to visible
        h_2_v_1 = tf.nn.sigmoid(tf.matmul(self.gibbs(v_2_h_1),
                                          tf.transpose(self.W))
                                + self.v_bias)
        # second visible to hidden pass
        # sample v ~ P(v|h) and feed it to hidden
        v_2_h_2 = tf.nn.sigmoid(tf.matmul(self.gibbs(h_2_v_1), self.W)
                                + self.h_bias)
        
        # contrastive divergence
        positive_grad = tf.matmul(tf.transpose(self.v), v_2_h_1)
        negative_grad = tf.matmul(tf.transpose(h_2_v_1), v_2_h_2)
        
        # merge current gradients with momentum
        W_grad_raw = positive_grad - negative_grad
        curr_W_grad = self.c_momentum(self.grad_W, W_grad_raw) / tf.to_float(tf.shape(self.v)[0])
        
        v_bias_grad_raw = tf.reduce_mean(self.v - h_2_v_1, 0)
        curr_v_bias_grad = self.c_momentum(self.grad_v_bias,
                                    v_bias_grad_raw)
        
        h_bias_grad_raw = tf.reduce_mean(v_2_h_1 - v_2_h_2, 0)
        curr_h_bias_grad = self.c_momentum(self.grad_h_bias,
                                    h_bias_grad_raw)
        
        # for parameters and gradient updating
        # see function 'batch_train'
        self.update_grad_W = self.grad_W.assign(curr_W_grad)
        self.update_grad_v_bias = self.grad_v_bias.assign(curr_v_bias_grad)
        self.update_grad_h_bias = self.grad_h_bias.assign(curr_h_bias_grad)
        self.update_W = self.W.assign(self.W + curr_W_grad)
        self.update_v_bias = self.v_bias.assign(self.v_bias + curr_v_bias_grad)
        self.update_h_bias = self.h_bias.assign(self.h_bias + curr_h_bias_grad)
        
                # visible to hidden pass
        self.v_2_h = tf.nn.sigmoid(tf.matmul(self.v, self.W) 
                              + self.h_bias)
        # get reconstructed data from hidden layer
        self.h_2_v = tf.nn.sigmoid(tf.matmul(self.v_2_h, tf.transpose(self.W)) 
                              + self.v_bias)
        
    def free_energy(self, data):
        return self.sess.run(self.free_energy, feed_dict={self.v: data})        
    
    
    # reconstruct data or get root mean square error
    def reconstruct(self, data, get_loss=False):
        if get_loss:
            return self.sess.run(tf.sqrt
                             ( tf.reduce_mean( tf.square(data-self.h_2_v) ) )
                                , feed_dict={self.v: data})
        return self.sess.run(self.h_2_v, feed_dict={self.v: data})
    
    # train 1 mini batch and update all parameters
    def batch_train(self, mini_batch):
        #updating all parameters and gradients
        updates = [
            self.update_grad_W, self.update_grad_v_bias, 
            self.update_grad_h_bias,
            self.update_W, self.update_v_bias, 
            self.update_h_bias   
        ] 
        self.sess.run(updates, feed_dict = {self.v: mini_batch})
        
    def train(self, data, validation_data, 
              n_epoches=10, batch_size=32, validation_steps=500):
        n = data.shape[0]//batch_size
        if data.shape[0] % batch_size==0:
            n_batches = n
        else:
            n_batches = n + 1
        errors = []
        for epoch in range(n_epoches):
            print('training epoche {}/{}:'.format(epoch,n_epoches), end='')
            for i in tqdm(range(n_batches)):
                mini_batch = data[i * batch_size :
                                  (i+1) * batch_size]
                self.batch_train(mini_batch)
            error = self.reconstruct(validation_data, get_loss=True)
            print('validation loss: {:.9f}'.format(error))
            errors.append(error)
        return errors
    
    def save_params(self, directory, filename):
        # save parameters with filename
        # to directory (folder)
        try:
            os.mkdir(directory)
        except FileExistsError as e:
            print(e)
        self.saver.save(self.sess, './' + directory + '/' + filename)
        print('model has been successfully saved!')
    
    def load_params(self, directory, filename):
        self.saver.restore(self.sess, './' + directory + '/' + filename)
        print('model has been successfully loaded')
    
    def gibbs(self, proba):
        # gibbs sampling: e.g. given proba=0.3
        # sample u ~ uniform[0, 1]
        # if u < 0.3 then return 1; else 0
        return tf.nn.relu(tf.sign(proba 
                        - tf.random_uniform(tf.shape(proba))))
    
    def c_momentum(self, prev_grad, curr_grad):
        # merging gradient with momentum
        # momentum * previous gradient + learning rate * current gradient
        return self.momentum * prev_grad + self.lr * curr_grad
