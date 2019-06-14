import tensorflow as tf
import numpy as np
class NN:
    def __init__(self,batchsize,channelsize,width,height):
        self.sess = tf.Session()
        msg = tf.constant("V1.0")
        self.sess.run(msg)
        self.batchsize = batchsize
        self.channelsize = channelsize
        self.width = width
        self.height = height
        self.inputs = tf.placeholder(dtype = tf.float32,shape=[self.batchsize,self.channelsize,self.width,self.height],name="Qt0")
        self.act = self.create_network(self.inputs)
        #loss component
        self.inputs_next_step = tf.placeholder(dtype = tf.float32,shape=[self.batchsize,self.channelsize,self.width,self.height],name="Qt1")
        self.gamma = tf.placeholder(dtype=tf.float32,shape=(),name="gamma")
        separ = np.zeros(self.batchsize,np.float32)
        self.rr = tf.constant(separ)
        self.r = tf.placeholder(dtype=tf.float32,shape=[self.batchsize],name="r")
        y = self.create_network(self.inputs_next_step,isTraining=False,reuse=True)
        y = tf.reduce_max(y,axis = 1)
        mask = tf.less(self.r,self.rr)
        self.y = tf.where(mask,self.r,self.r+y*self.gamma)
        self.action_index = tf.placeholder(dtype=tf.int32,shape=[self.batchsize],name="action_index")
        self.qvalue_now = tf.slice(self.y,self.action_index,[1])
        self.loss = tf.losses.mean_squared_error(self.y,self.qvalue_now)
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.sess.run(tf.initializers.global_variables())

    def create_network(self,inputs,isTraining=True,reuse=False):
        #channels first
        net = tf.layers.conv2d(inputs,32,3,padding='same',activation=tf.nn.relu,data_format='channels_first')
        net = tf.layers.conv2d(net,32,5,strides=(5,5),padding='same',activation=tf.nn.relu,data_format='channels_first')
        net = tf.layers.conv2d(net,32,3,padding='same',activation=tf.nn.relu,data_format='channels_first')
        net = tf.layers.conv2d(net,32,5,strides=(5,5),padding='same',activation=tf.nn.relu,data_format='channels_first')
        net = tf.layers.conv2d(net,32,3,padding='same',activation=tf.nn.relu,data_format='channels_first')
        net = tf.reshape(net,shape=[self.batchsize,-1])
        net = tf.layers.dense(net,32,activation=tf.nn.relu)
        net= tf.layers.dense(net,32,activation=tf.nn.relu) 
        net= tf.layers.dense(net,5,activation=tf.nn.relu) #queue value of no move,up,down,left,right
        act = tf.nn.softmax(net)
        return act

    def cal_loss(self,np_images_previous,np_image_after,reward,action_index,gamma):
        l,_ = self.sess.run([self.train_op,self.loss],feed_dict={self.inputs:np_images_previous,self.inputs_next_step:np_image_after,self.r:reward,self.gamma:gamma,self.action_index:action_index})

    def forward(self,np_images):
        act = self.sess.run(self.act,feed_dict={self.inputs:np_images})
        return act
