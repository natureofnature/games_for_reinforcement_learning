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
        net= tf.layers.dense(net,32,activation=tf.nn.relu) 
        net= tf.layers.dense(net,5,activation=tf.nn.relu) #queue value of no move,up,down,left,right
        act = tf.nn.softmax(net)
        self.act = act
        self.sess.run(tf.initializers.global_variables())


    def forward(self,np_images):
        act = self.sess.run(self.act,feed_dict={self.inputs:np_images})
        return act
