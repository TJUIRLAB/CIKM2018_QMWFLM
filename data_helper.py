# -*- coding:utf-8-*-
from __future__ import print_function
import numpy as np
import random,os,math
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import sklearn
import multiprocessing
import time
import pickle as pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import evaluation
import string

from nltk import stem
from tqdm import tqdm
import chardet
import re
import config
from functools import wraps

FLAGS = config.flags.FLAGS
FLAGS.flag_values_dict()
dataset = FLAGS.data
isEnglish = FLAGS.isEnglish
UNKNOWN_WORD_IDX = 0
is_stemmed_needed = False

rng = np.random.RandomState(23455)
# ner_dict = pickle.load(open('ner_dict'))
if is_stemmed_needed:
    stemmer = stem.lancaster.LancasterStemmer()
def cut(sentence,isEnglish = isEnglish):
    if isEnglish:
        tokens = sentence.lower().split()
    else:
        # words = jieba.cut(str(sentence))
        tokens = [word for word in sentence.split() if word not in stopwords]
    return tokens

################ recording time for function

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

############### get the word dict ##########

class Alphabet(dict):
    def __init__(self, start_feature_id = 1):
        self.fid = start_feature_id

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
      # self[idx] = item
            self.fid += 1
        return idx

    def dump(self, fname):
        with open(fname, "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))
############## prepare the dataset ############
@log_time_delta
def prepare(cropuses,is_embedding_needed = False,dim = 50,fresh = False):
    vocab_file = 'model/voc'
    
    if os.path.exists(vocab_file) and not fresh:
        alphabet = pickle.load(open(vocab_file,'r'))
    else:   
        alphabet = Alphabet(start_feature_id=0)
        alphabet.add('[UNKNOW]')  
        alphabet.add('END') 
        count = 0
        for corpus in cropuses:
            for texts in [corpus["question"].unique(),corpus["answer"]]:
                for sentence in tqdm(texts):   
                    count += 1
                    if count % 10000 == 0:
                        print (count)
                    tokens = cut(sentence)
                    for token in set(tokens):
                        alphabet.add(token)

    if is_embedding_needed:
        sub_vec_file = 'embedding/sub_vector'
        if os.path.exists(sub_vec_file) and not fresh:
            sub_embeddings = pickle.load(open(sub_vec_file,'r'))
        else:    
            if isEnglish:        
                fname = '../../embedding/glove.6B/glove.6B.{}d.txt'.format(dim)
                embeddings = load_text_vec(alphabet,fname,embedding_size = dim)
                sub_embeddings = getSubVectorsFromDict(embeddings,alphabet,dim)
            else:
                fname = 'model/wiki.ch.text.vector'
                embeddings = load_text_vec(alphabet,fname,embedding_size = dim)
                sub_embeddings = getSubVectorsFromDict(embeddings,alphabet,dim)

        return alphabet,sub_embeddings
    else:
        return alphabet


def load_text_vec(alphabet,filename="",embedding_size = 100):
    vectors = {}
    with open(filename) as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print ('epch %d' % i)
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size= items[0],items[1]
                print ( vocab_size, embedding_size)
            else:
                word = items[0]
                if word in alphabet:
                    vectors[word] = items[1:]
    print ('embedding_size',embedding_size)
    print ('done')
    print ('words found in wor2vec embedding ',len(vectors.keys()))
    return vectors
def getSubVectorsFromDict(vectors,vocab,dim = 300):

    embedding = np.zeros((len(vocab),dim))
    count = 1
    for word in vocab:
        
        if word in vectors:
            count += 1
            embedding[vocab[word]]= vectors[word]
        else:
   
 
            embedding[vocab[word]]= rng.uniform(-0.5,+0.5,dim)#vectors['[UNKNOW]'] #.tolist()
    print ('word in embedding',count)
    return embedding

@log_time_delta
def batch_gen_with_pair(df,alphabet, batch_size = 10,q_len = 40,a_len = 40,fresh = True,overlap_dict = None):
    pairs = []
    start = time.time()
    for question in df["question"].unique():
        group = df[df["question"]==question]
        pos_answers = group[df["flag"] == 1]
        pos_answers = pos_answers['answer']
        neg_answers = group[df["flag"] == 0]["answer"].reset_index()
        question_indices = encode_to_split(question,alphabet,max_sentence = q_len)
        for pos in pos_answers:
            if len(neg_answers.index) > 0:
                neg_index = np.random.choice(neg_answers.index)
                neg = neg_answers.loc[neg_index,]["answer"]
                pairs.append((question_indices,encode_to_split(pos,alphabet,max_sentence = a_len),encode_to_split(neg,alphabet,max_sentence = a_len)))

    print ('pairs:{}'.format(len(pairs)))
    end = time.time()
    delta = end - start
    print( "batch_gen_with_pair_runed %.2f seconds" % (delta))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches= int(len(pairs)*1.0/batch_size)
    pairs = sklearn.utils.shuffle(pairs,random_state =132)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield [[pair[i] for pair in batch]  for i in range(3)]

def encode_to_split(sentence,alphabet,max_sentence = 40):
    indices = []    
    tokens = cut(sentence)
    for word in tokens:
        if word in alphabet:
            indices.append(alphabet[word])
        else:
            continue
    results = indices+[alphabet["END"]]*(max_sentence-len(indices))
    return results[:max_sentence]
def transform(flag):
    if flag == 1:
        return [0,1]
    else:
        return [1,0]
@log_time_delta
def batch_gen_with_single(df,alphabet,batch_size = 10,q_len = 33,a_len = 40,overlap_dict = None):
    pairs=[]
    for index,row in df.iterrows():
        question = encode_to_split(row["question"],alphabet,max_sentence = q_len)
        answer = encode_to_split(row["answer"],alphabet,max_sentence = a_len)
        pairs.append((question,answer))
    
    num_batches_per_epoch = int((len(pairs)-1)/ batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(pairs))
            batch = pairs[start_index:end_index]
            yield [[pair[j] for pair in batch]  for j in range(2)]

def load(dataset = dataset, filter = False):
    data_dir = "data/" + dataset
    datas = []

    names = ['train.txt','test.txt','dev.txt']
    if dataset == 'trec':
        names = ['train-all.txt','test.txt','dev.txt']
    for data_name in names:
        data_file = os.path.join(data_dir,data_name)
        data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna('')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    return tuple(datas)
def removeUnanswerdQuestion(df):
    counter= df.groupby("question").apply(lambda group: sum(group["flag"]))
    questions_have_correct=counter[counter>0].index
    counter= df.groupby("question").apply(lambda group: sum(group["flag"]==0))
    questions_have_uncorrect=counter[counter>0].index
    counter=df.groupby("question").apply(lambda group: len(group["flag"]))
    questions_multi=counter[counter>1].index

    return df[df["question"].isin(questions_have_correct) &  df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()


