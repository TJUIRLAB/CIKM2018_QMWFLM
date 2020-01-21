#coding=utf-8
#! /usr/bin/env python3.4
import tensorflow as tf
import numpy as np
import os
import time
import datetime

from data_helper import batch_gen_with_pair,load,prepare,batch_gen_with_single
import operator

from QA_CNN_quantum_pairwise import QA_CNN_quantum_extend
# from QA_RNN_pairwise import QA_RNN_extend
import random
import evaluation
import pickle as pickle
import config
from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.set_random_seed(1234)
now = int(time.time()) 
    
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)

from functools import wraps
#print( tf.__version__)
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco

FLAGS = config.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
log_dir = 'log/'+ timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

data_file = log_dir + '/test_' + FLAGS.data + timeStamp
precision = data_file + 'precise'
attention = []

@log_time_delta
def predict(sess,cnn,test,alphabet,batch_size,q_len,a_len):
    scores = []

    for data in batch_gen_with_single(test,alphabet,batch_size,q_len,a_len,overlap_dict = None): 
        feed_dict = {
                    cnn.question: data[0],
                    cnn.answer: data[1],
         
                    cnn.dropout_keep_prob:1.0
                    }
        score = sess.run(cnn.score12, feed_dict)
        scores.extend(score)
    return np.array(scores[:len(test)])

@log_time_delta
def test_pair_wise(dns = FLAGS.dns):
    train,test,dev = load(FLAGS.data,filter = FLAGS.clean)
    test = test.reindex(np.random.permutation(test.index))

    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print ('q_question_length:{} a_question_length:{}'.format(q_max_sent_length,a_max_sent_length))
    print ('train question unique:{}'.format(len(train['question'].unique())))
    print ('train length',len(train))
    print ('test length', len(test))
    print ('dev length', len(dev))
    alphabet,embeddings = prepare([train,test,dev],dim = FLAGS.embedding_dim,is_embedding_needed = True,fresh = FLAGS.fresh)
    # alphabet,embeddings = prepare_300([train,test,dev])
    print ('alphabet:',len(alphabet))
    with tf.Graph().as_default(), tf.device("/gpu:" + str(FLAGS.gpu)):
        # with tf.device("/cpu:0"):
        session_conf = tf.ConfigProto()
        session_conf.allow_soft_placement = FLAGS.allow_soft_placement
        session_conf.log_device_placement = FLAGS.log_device_placement
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default(),open(precision,"w") as log:
            log.write(str(FLAGS.__flags) + '\n')
            folder = 'runs/' + timeDay + '/' + timeStamp + '/'
            out_dir = folder + FLAGS.data
            if not os.path.exists(folder):
                os.makedirs(folder)
            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            print ("start build model")
            cnn = QA_CNN_quantum_extend                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                (
                max_input_left = q_max_sent_length,
                max_input_right = a_max_sent_length,
                batch_size = FLAGS.batch_size,
                vocab_size = len(alphabet),
                embedding_size = FLAGS.embedding_dim,
                filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters = FLAGS.num_filters,                
                dropout_keep_prob = FLAGS.dropout_keep_prob,
                embeddings = embeddings,                
                l2_reg_lambda = FLAGS.l2_reg_lambda,
                overlap_needed = FLAGS.overlap_needed,
                learning_rate=FLAGS.learning_rate,
                trainable = FLAGS.trainable,
                extend_feature_dim = FLAGS.extend_feature_dim,
                pooling = FLAGS.pooling,
                position_needed = FLAGS.position_needed,
                conv = FLAGS.conv,
                margin = FLAGS.margin)
            cnn.build_graph()
           
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 20)
            train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(log_dir + '/test')
            # Initialize all variables
            print ("build over")
            sess.run(tf.global_variables_initializer())
            print ("variables_initializer")

            # saver.restore(sess, 'runs/20170910/20170910154937/wiki')
            map_max = 0.65
            for i in range(FLAGS.num_epochs):
                
                datas = batch_gen_with_pair(train,alphabet,FLAGS.batch_size,
                    q_len = q_max_sent_length,a_len = a_max_sent_length,fresh = FLAGS.fresh,overlap_dict = None)        
                print ("load data")
                for data in datas:
                    feed_dict = {
                        cnn.question: data[0],
                        cnn.answer: data[1],
                        cnn.answer_negative:data[2],
                        cnn.dropout_keep_prob:FLAGS.dropout_keep_prob
                    }
                    _, summary,step,loss, accuracy,score12,score13,see = sess.run(
                    [cnn.train_op, cnn.merged,cnn.global_step,cnn.loss, cnn.accuracy,cnn.score12,cnn.score13,cnn.see],
                    feed_dict)

                    train_writer.add_summary(summary, i)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))
                    line = "{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13))
                    # print loss
                if i % 1 == 0:
                    predicted_dev = predict(sess,cnn,dev,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                    map_mrr_dev = evaluation.evaluationBypandas(dev,predicted_dev)
                    predicted_test = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                    map_mrr_test = evaluation.evaluationBypandas(test,predicted_test)

                    precise_test = evaluation.precision(test,predicted_test)
                    
                    print("test precise : {}".format(precise_test))
                    print ("{}:epoch:dev map mrr {}".format(i,map_mrr_dev))
                    print ("{}:epoch:test map mrr {}".format(i,map_mrr_test))
                    line = " {}:epoch: precise: {}--- map_dev{}-------map_mrr_test{}".format(i,precise_test,map_mrr_dev[0],map_mrr_test)
                    if map_mrr_dev[0] > map_max:
                        map_max = map_mrr_dev[0]
                    
                        
                        save_path = saver.save(sess, out_dir)
                        print ("Model saved in file: ", save_path)

                log.write(line + '\n')
                log.flush()
            print ('train over')
            saver.restore(sess, out_dir)
            predicted = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
            train['predicted'] = predicted      
            map_mrr_train = evaluation.evaluationBypandas(train,predicted)
            predicted_dev = predict(sess,cnn,dev,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
            dev['predicted'] = predicted_dev           
            map_mrr_dev = evaluation.evaluationBypandas(dev,predicted_dev)
            predicted_test = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
            test['predicted'] = predicted_test           
            map_mrr_test = evaluation.evaluationBypandas(test,predicted_test)
    
            ap = evaluation.get_ap(test,predicted_test)
            ap.to_csv('ap_score_qlm_wiki',header = None,sep = '\t')
            print ('map_mrr train',map_mrr_train)
            print ('map_mrr dev',map_mrr_dev)
            print ('map_mrr test',map_mrr_test)
            log.write(str(map_mrr_train) + '\n')
            log.write(str(map_mrr_test) + '\n')
            log.write(str(map_mrr_dev) + '\n')

if __name__ == '__main__':
    test_pair_wise()

