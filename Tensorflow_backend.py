import tensorflow as tf
class NN:
    def __init__(self,batchsize,channelsize,width,height):
        self.sess = tf.Session()
        self.batchsize = batchsize
        self.channelsize = channelsize
        self.width = width
        self.height = height
        self.act1 = None
        self.act2 = None

    def create_network(self,isTraining=True):
        #channels first
        self.inputs = tf.placeholder(dtype = tf.float32,shape=[self.batchsize,self.channelsize,self.width,self.height])
        net = tf.layers.conv2d(self.inputs,32,3,padding='same',activation=tf.nn.relu,data_format='channels_first')
        net = tf.layers.conv2d(net,32,5,strides=(5,5),padding='same',activation=tf.nn.relu,data_format='channels_first')
        net = tf.layers.conv2d(net,32,3,padding='same',activation=tf.nn.relu,data_format='channels_first')
        net = tf.layers.conv2d(net,32,5,strides=(5,5),padding='same',activation=tf.nn.relu,data_format='channels_first')
        net = tf.layers.conv2d(net,32,3,padding='same',activation=tf.nn.relu,data_format='channels_first')
        net = tf.reshape(net,shape=[self.batchsize,-1])
        net = tf.layers.dense(net,32,activation=tf.nn.relu)
        net_1= tf.layers.dense(net,32,activation=tf.nn.relu) 
        net_1= tf.layers.dense(net_1,4,activation=tf.nn.relu) #up,down,left,right
        net_2= tf.layers.dense(net,32,activation=tf.nn.relu) 
        net_2= tf.layers.dense(net_2,2,activation=tf.nn.relu) #press/release
        act1 = tf.nn.softmax(net_1)
        act2 = tf.nn.softmax(net_2)

        self.act1 = act1
        self.act2 = act2
        self.sess.run(tf.initializers.global_variables())


    def forward(self,np_images):
        act1,act2 = self.sess.run([self.act1,self.act2],feed_dict={self.inputs:np_images})
        return act1,act2
