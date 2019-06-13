import tensorflow as tf
class NN:
    def __init__(self,batchsize,channelsize,width,height):
        self.sess = tf.Session()
        msg = tf.constant("V1.0")
        self.sess.run(msg)
        self.batchsize = batchsize
        self.channelsize = channelsize
        self.width = width
        self.height = height
        self.inputs = tf.placeholder(dtype = tf.float32,shape=[self.batchsize,self.channelsize,self.width,self.height])
        self.act = self.create_network(self.inputs)
        self.inputs_next_step = tf.placeholder(dtype = tf.float32,shape=[self.batchsize,self.channelsize,self.width,self.height])
        self.gamma = tf.placeholder(dtype=tf.float32)
        self.r = tf.placeholder(dtype=tf.float32)
        self.y = self.create_network(self.inputs_next_step,isTraining=False,reuse=True)
        self.y = tf.cond(self.r<0,lambda:self.r,lambda:self.r+self.y*self.gamma)
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

    def loss(self,np_images_previous,np_image_after,reward,gamma):
        loss = tf.losses.mean_squared_error(self.y,self.act)
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        l,_ = self.sess.run([train_op,loss],feed_dict={self.inputs:np_images_previous,self.inputs_next_step:np_image_after,self.r:reward,self.gamma:gamma})

    def forward(self,np_images):
        act = self.sess.run(self.act,feed_dict={self.inputs:np_images})
        return act
