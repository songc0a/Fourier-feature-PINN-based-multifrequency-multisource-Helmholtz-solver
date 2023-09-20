"""
@author: Chao Song
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
import cmath

np.random.seed(1234)
tf.set_random_seed(1234)

PI = 3.1415926
niter = 100000
misfit = []
misfit1 = []

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, z, sx, fre, u0_real, u0_imag, m, m0, layers):
        
        self.iter=0
        self.start_time=0

        X = np.concatenate([x, z, sx, fre], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.x = X[:,0:1]
        self.z = X[:,1:2]
        self.sx = X[:,2:3]
        self.fre = X[:,3:4]

        self.layers = layers
        self.u0_real = u0_real
        self.u0_imag = u0_imag
        self.m = m
        self.m0 = m0

        # Initialize multi-scale Fourier features
        self.W = tf.Variable(tf.random_uniform([4, layers[0] //2  ], minval=-0.04, maxval=0.04, dtype=tf.float32) ,
                               dtype=tf.float32, trainable=False)

       # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers) 

        self.saver = tf.train.Saver() ##saver initialization 

        # tf placeholders 
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.sx_tf = tf.placeholder(tf.float32, shape=[None, self.sx.shape[1]])
        self.fre_tf = tf.placeholder(tf.float32, shape=[None, self.fre.shape[1]])
       
        self.du_real_pred, self.du_imag_pred, self.f_real_pred, self.f_imag_pred = self.net_NS(self.x_tf, self.z_tf, self.sx_tf, self.fre_tf)

        # loss function we define  
        self.loss = tf.reduce_sum(tf.square(self.f_real_pred)) + \
                    tf.reduce_sum(tf.square(self.f_imag_pred))
        
        # optimizer used by default (in original paper)        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        10000, 0.9, staircase=False)
        
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
   #     self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.00025)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.random_normal([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
 
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
       # xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        xavier_stddev = np.sqrt(1./in_dim)
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)

        X = H
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0

        # Multi-scale Fourier feature encodings
        H = tf.concat([tf.sin(tf.matmul(H, self.W)),
                        tf.cos(tf.matmul(H, self.W))], 1)
        
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.atan(tf.add(tf.matmul(H, W), b))
        # Merge the outputs by concatenation
      
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)

        return H

    # Forward pass for u
    def net_u(self, x, z, sx, fre):
        u = self.forward_pass(tf.concat([x, z, sx, fre], 1))
        return u


    def net_NS(self, x, z, sx, fre):
    
        # output scattered wavefield: du_real du_imag, loss function: L du+omega^2 dm u0 
        omega = 2.0*PI*fre
        m = self.m
        m0 = self.m0
        u0_real = self.u0_real
        u0_imag = self.u0_imag

        dureal_and_duimag =  self.net_u(x,z,sx,fre)
        du_real = dureal_and_duimag[:,0:1]
        du_imag = dureal_and_duimag[:,1:2]

        du_real_x = tf.gradients(du_real, x)[0]
        du_real_z = tf.gradients(du_real, z)[0]
        du_real_xx = tf.gradients(du_real_x, x)[0]
        du_real_zz = tf.gradients(du_real_z, z)[0]

        du_imag_x = tf.gradients(du_imag, x)[0]
        du_imag_z = tf.gradients(du_imag, z)[0]
        du_imag_xx = tf.gradients(du_imag_x, x)[0]
        du_imag_zz = tf.gradients(du_imag_z, z)[0]

        f_real =  omega*omega*m*du_real + du_real_xx + du_real_zz + omega*omega*(m-m0)*u0_real #  L du + omega^2 dm u0 
        f_imag =  omega*omega*m*du_imag + du_imag_xx + du_imag_zz + omega*omega*(m-m0)*u0_imag #  L du + omega^2 dm u0
 
        return du_real, du_imag, f_real, f_imag 

    def callback(self, loss):
        #print('Loss: %.3e' % (loss))
        misfit1.append(loss)
        self.iter=self.iter+1
        if self.iter % 10 == 0:
                elapsed = time.time() - self.start_time
                print('It: %d, LBFGS Loss: %.3e,Time: %.2f' %
                      (self.iter, loss, elapsed))
                self.start_time = time.time()

      
    def train(self, nIter): 
        
        start_time = time.time()
        tf_dict = {self.x_tf: self.x, self.z_tf: self.z, self.sx_tf: self.sx, self.fre_tf: self.fre}

        for it in range(niter):

            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            misfit.append(loss_value)         
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e,Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()

            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
    
        self.saver.save(self.sess,'./checkpoint_dir/Mymodel_ff_two32_004') # model save

    def predict(self, x_star, z_star, sx_star, fre_star):
        
        tf_dict = {self.x_tf: x_star, self.z_tf: z_star, self.sx_tf: sx_star, self.fre_tf: fre_star}
        
        du_real_star = self.sess.run(self.du_real_pred, tf_dict)
        du_imag_star = self.sess.run(self.du_imag_pred, tf_dict)

        return du_real_star, du_imag_star
       
if __name__ == "__main__": 

    layers = [128,128,64, 64, 32, 32, 32, 32, 2]
    # Load Data
    data = scipy.io.loadmat('sigsbee_5Hz_testdata_fre.mat')

    x_star = data['x_star'] 
    z_star = data['z_star']  
    sx_star = data['sx_star']  

    train_data = scipy.io.loadmat('sigsbee_train_data_fre_N50000_f510.mat')
           
    u0_real_train = train_data['U0_real_train'] 
    u0_imag_train = train_data['U0_imag_train'] 

    x_train = train_data['x_train'] 
    z_train = train_data['z_train'] 
    sx_train = train_data['sx_train'] 
    fre_train = train_data['f_train']

    m_train = train_data['m_train']
    m0_train = train_data['m0_train']

    #N = x.shape[0]
    # Training

    model = PhysicsInformedNN(x_train, z_train, sx_train, fre_train, u0_real_train, u0_imag_train, m_train, m0_train, layers)
    model.train(niter)

    scipy.io.savemat('misfit_fre_two32_from5_ff_uniform004_xavier1.mat',{'misfit':misfit})
    scipy.io.savemat('misfit1_fre_two32_from5_ff_uniform004_xavier1.mat',{'misfit1':misfit1}) 
    

    N = x_star.shape[0]
    frequency_band = range(4,11)

    for fre in frequency_band:

        fre_star = fre*np.ones((N,1),dtype=np.float)
        du_real_pred, du_imag_pred = model.predict(x_star, z_star, sx_star, fre_star)
        scipy.io.savemat('du_real_pred_atan_%dhz_fre_two32_from5_ff_uniform004_xavier1.mat'%(fre),{'du_real_pred':du_real_pred})
        scipy.io.savemat('du_imag_pred_atan_%dhz_fre_two32_from5_ff_uniform004_xavier1.mat'%(fre),{'du_imag_pred':du_imag_pred})




