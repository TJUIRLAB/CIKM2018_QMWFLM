import tensorflow as tf
import numpy as np
tf.set_random_seed(1234)

# model_type :apn or qacnn
class QA_CNN_quantum_extend(object):
    def __init__(self,max_input_left,max_input_right,batch_size,vocab_size,embedding_size,filter_sizes,num_filters,
        dropout_keep_prob = 1,learning_rate = 0.001,embeddings = None,l2_reg_lambda = 0.0,overlap_needed = False,trainable = True,extend_feature_dim = 10,pooling = 'attentive',position_needed = True,conv = 'narrow',margin = 0.05):
        print("QA_CNN_quantum_extend")
        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.filter_sizes = filter_sizes
        self.l2_reg_lambda = l2_reg_lambda
        self.para = []
        self.extend_feature_dim = extend_feature_dim
        self.max_input_left = max_input_left
        self.max_input_right = max_input_right
        self.overlap_needed = overlap_needed
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.trainable = trainable
        self.vocab_size = vocab_size
        self.pooling = pooling
        self.position_needed = position_needed
        self.conv = conv
        self.learning_rate = learning_rate
        self.margin = margin
    def create_placeholder(self):
        print('Create placeholders')
        self.question = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'input_answer')
        self.answer_negative = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'input_right')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name = 'dropout_prob')
    def add_embeddings(self):
        print('add_embeddings')
        if self.embeddings is not None:
            print ("load embedding")
            W = tf.Variable(np.array(self.embeddings),name = "W" ,dtype="float32",trainable = self.trainable)
            
        else:
            print ("random embedding")
            W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
        self.embedding_W = W
        # self.overlap_W = tf.Variable(a,name="W",trainable = True)
        self.para.append(self.embedding_W)
         #get embedding
        self.q_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_W,self.question),-1)
        self.a_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_W,self.answer),-1)
        self.a_neg_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_W,self.answer_negative),-1)
        self.see = self.q_embedding
    def convolution(self):

        print ('convolution:wide_convolution')
        self.kernels = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-max-pool-%s' % filter_size):
                filter_shape = [filter_size,self.embedding_size,1,self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)
        #convolution

        embeddings = [self.q_embedding,self.a_embedding,self.a_neg_embedding]
        self.q_feature_map,self.a_pos_feature_map,self.a_neg_feature_map = \
        [self.wide_convolution(embedding) for embedding in embeddings]

    def product_pooling(self,conv):
        s = tf.squeeze(conv,2)

        s_represent = tf.reduce_mean(tf.log(tf.maximum(s + 1,1e-12)),1)
        return s_represent
    def mean_pooling(self,conv):
        s = tf.squeeze(conv,2)
        s_represent = tf.reduce_mean(s,1)
        return s_represent



    def pooling_graph(self):
        print ('pooling: max pooling or attentive pooling or product')
        print (self.pooling)
        
        if self.pooling == 'product':
            self.q_pos_pooling = self.product_pooling(self.q_feature_map)
            self.q_neg_pooling = self.product_pooling(self.q_feature_map)
            self.a_pos_pooling = self.product_pooling(self.a_pos_feature_map)
            self.a_neg_pooling = self.product_pooling(self.a_neg_feature_map)
        elif self.pooling == 'mean':
            self.q_pos_pooling = self.mean_pooling(self.q_feature_map)
            self.q_neg_pooling = self.mean_pooling(self.q_feature_map)
            self.a_pos_pooling = self.mean_pooling(self.a_pos_feature_map)
            self.a_neg_pooling = self.mean_pooling(self.a_neg_feature_map)
        else:
            print ('no implement')
            exit(0)  
    def create_loss(self):
        
        with tf.name_scope('score'):
            self.score12 = self.getCosine(self.q_pos_pooling,self.a_pos_pooling)
            self.score13 = self.getCosine(self.q_neg_pooling,self.a_neg_pooling)
          
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(0.0, tf.subtract(self.margin, tf.subtract(self.score12, self.score13)))
            self.loss = tf.reduce_sum(self.losses) + self.l2_reg_lambda * l2_loss
        tf.summary.scalar('loss', self.loss)
        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)
    def create_op(self):
        self.global_step = tf.Variable(0, name="global_step", trainable = False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step = self.global_step)

    def concat_embedding(self,words_indice,overlap_indice,position_indice,conv_position):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        all_embedding = tf.expand_dims(embedded_chars_q,-1)

        return all_embedding
    def max_pooling(self,conv,input_length):
        pooled = tf.nn.max_pool(
                    conv,
                    ksize = [1, input_length, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name="pool")
        return pooled
    def getCosine(self,q,a):

        q = tf.nn.l2_normalize(q,1)
        a = tf.nn.l2_normalize(a,1)
        self.see = q
        score = tf.reduce_sum(tf.multiply(q,a),1)
        return score
    
    def wide_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, self.embedding_size, 1],
                    padding='SAME',
                    name="conv-1"
            )
            h = tf.nn.elu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped
    def narrow_convolution_pooling(self):
        print ('narrow pooling')
        self.kernels = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-max-pool-%s' % filter_size):
                filter_shape = [filter_size,self.total_embedding_dim,1,self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name="W")
                b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)
        embeddings = [self.q_pos_embedding,self.q_neg_embedding,self.a_pos_embedding,self.a_neg_embedding]
        self.q_pos_pooling,self.q_neg_pooling,self.a_pos_pooling,self.a_neg_pooling = [self.getFeatureMap(embedding,right = i / 2) for i,embedding in enumerate(embeddings) ]
    def variable_summaries(self,var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        if self.conv == 'narrow':
            self.narrow_convolution_pooling()
        else:
            self.convolution()
            # self.out_product()
            self.pooling_graph()
        self.create_loss()
        self.create_op()
        self.merged = tf.summary.merge_all()

    
if __name__ == '__main__':
    cnn = QA_CNN_quantum_extend(max_input_left = 33,
        max_input_right = 40,
        batch_size = 3,
        vocab_size = 5000,
        embedding_size = 100,
        filter_sizes = [3,4,5],
        num_filters = 64, 
        dropout_keep_prob = 1.0,
        embeddings = None,
        l2_reg_lambda = 0.0,
        overlap_needed = False,
        trainable = True,
        extend_feature_dim = 10,
        position_needed = False,
        pooling = 'product',
        conv = 'wide',
        margin = 0.05)
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_x_3 = np.reshape(np.arange(3 * 40),[3,40])

    # q_pos_embedding = np.ones((3,33))
    # q_neg_embedding = np.ones((3,33))
    # a_pos_embedding = np.ones((3,40))
    # a_neg_embedding = np.ones((3,40)) 

    # q_position = np.ones((3,33))
    # a_pos_position = np.ones((3,40))
    # a_neg_position = np.ones((3,40))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.answer_negative:input_x_3,
            cnn.dropout_keep_prob:1.0
            # cnn.q_pos_overlap:q_pos_embedding,
            # cnn.q_neg_overlap:q_neg_embedding,
            # cnn.a_pos_overlap:a_pos_embedding,
            # cnn.a_neg_overlap:a_neg_embedding,
            # cnn.q_position:q_position,
            # cnn.a_pos_position:a_pos_position,
            # cnn.a_neg_position:a_neg_position
        }
        question,answer,score,see = sess.run([cnn.question,cnn.answer,cnn.score12,cnn.see],feed_dict)
        print(score)
        # print(see)


