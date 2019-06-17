import tensorflow as tf
import numpy as np
class NN:
    def __init__(self,batchsize,channelsize,width,height):
        self.sess= tf.Session()
        self.batchsize = batchsize
        self.channelsize = channelsize
        self.width = width
        self.height = height
        #used for
        #         (1) inference and store in the replay memory 
        #         (2) loss calculation
        self.inputs = tf.placeholder(dtype = tf.float32,shape=[1,self.channelsize,self.width,self.height],name="Qt0")
        self.act = self.create_network(self.inputs,isTraining = False)

        #loss component
        #@1 sample (St, at, rt, St+1) outside tensorflow
        #@2 calculate scores using St+1, inside tensorflow
        #@3 calculate scores using St, inside tensorflow
        #@4 calculate loss between  @2 and @3

        '''step 2, only calculate max q value of next step, no action calculation involved'''
        '''(2-1) inputs of next time step '''
        self.inputs_previous_step = tf.placeholder(dtype = tf.float32,shape=[self.batchsize,self.channelsize,self.width,self.height],name="Qt1")
        self.inputs_next_step = tf.placeholder(dtype = tf.float32,shape=[self.batchsize,self.channelsize,self.width,self.height],name="Qt2")
        '''(2-2) gamma, shape = scalar'''
        self.gamma = tf.placeholder(dtype=tf.float32,shape=(),name="gamma")
        '''(2-3) reward, shape = (batchsize,1) '''
        self.r = tf.placeholder(dtype = tf.float32,shape = [self.batchsize,1],name='r') 
        '''(2-4) y, real reward,shape = (batchsize,number_actions) '''
        #    batch index            act0, act1, act2,act3,act4
        #      0                     0.1   0.2  0.2   0.3 0.2
        #      1                     0.2   0.3  0.1   0.1 0.3
        #    ...                     ...   ...  ...   ... ...
        y_pre = self.create_network(self.inputs_next_step, isTraining = True , reuse = True)
        y = self.create_network(self.inputs_next_step, isTraining = False, reuse = True)
        y = tf.reduce_max(y,axis = 1,keepdims = True)
        #   batch index            max_score
        #      0                    0.3
        #      1                    0.3
        #     ...                   ...
        tmp = np.zeros([self.batchsize,1],np.float32)
        tmp = tf.constant(tmp)
        mask = tf.less(self.r,tmp)
        self.y = tf.where(mask,self.r,self.r+y*self.gamma)

        '''step 3, calculate the q value corresponding to a selected action'''
        '''(3-1) action index, selected'''
        #   batch index             value
        #      0                     1
        #      1                     2
        #      2                     0
        #     ...                   ... (0~4)
        self.action_index = tf.placeholder(dtype=tf.int64,shape=[self.batchsize,1],name="action_index")
        #tf.gather_nd, index defines (batch_index, chosen_element_in_batch)
        #so combine batch indices and action_index to (batch_index, action_index)
        batch_indices = np.reshape(np.arange(self.batchsize),[self.batchsize,1])
        batch_indices = tf.constant(batch_indices)
        action_index = tf.concat([batch_indices,self.action_index],axis=1)
        '''(3-2) get q-value corresponding to chosen action '''
        self.qvalue_action = tf.gather_nd(y_pre,action_index)
        self.qvalue_action = tf.expand_dims(self.qvalue_action,-1)
        self.loss = tf.losses.mean_squared_error(self.y,self.qvalue_action)
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.sess.run(tf.initializers.global_variables())


        




      

    def create_network(self,inputs,isTraining=True,reuse=False):
        #channels first
        net = tf.layers.conv2d(inputs,32,3,padding='same',activation=tf.nn.relu,data_format='channels_first',trainable = isTraining)
        net = tf.layers.conv2d(net,32,5,strides=(5,5),padding='same',activation=tf.nn.relu,data_format='channels_first',trainable = isTraining)
        net = tf.layers.conv2d(net,32,3,padding='same',activation=tf.nn.relu,data_format='channels_first',trainable = isTraining)
        net = tf.layers.conv2d(net,32,5,strides=(5,5),padding='same',activation=tf.nn.relu,data_format='channels_first',trainable = isTraining)
        net = tf.layers.conv2d(net,32,3,padding='same',activation=tf.nn.relu,data_format='channels_first',trainable = isTraining)
        net = tf.reshape(net,shape=[-1,20*16*32])
        net = tf.layers.dense(net,32,activation=tf.nn.relu,trainable = isTraining)
        net= tf.layers.dense(net,32,activation=tf.nn.relu,trainable = isTraining) 
        net= tf.layers.dense(net,5,activation=tf.nn.relu,trainable = isTraining) #queue value of no move,up,down,left,right
        act = tf.nn.softmax(net)
        return act

    def cal_loss(self,np_images_previous,np_image_after,reward,action_index,gamma):
        _,l = self.sess.run([self.train_op,self.loss],feed_dict={self.inputs_previous_step:np_images_previous,self.inputs_next_step:np_image_after,self.r:reward,self.gamma:gamma,self.action_index:action_index})
        print(l)

    def forward(self,np_images):
        act = self.sess.run(self.act,feed_dict={self.inputs:np_images})
        return act
