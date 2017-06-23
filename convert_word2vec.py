#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This code takes a pickled object containing the labelled sentences and converts
the sentences  to their word representations and saves that as a pickled object
"""
from gensim.models.keyedvectors import KeyedVectors as kv
import pickle
import numpy as np

# load pubmed word2vec model
# wv = kv.load_word2vec_format("PubMed-and-PMC-w2v.bin", binary=True)

with open('labelled','rb') as f:
    data = pickle.load(f)

    
def return_word2vec_padded(x, padding):
    """
    Return a numpy array containing the word vectors of a sentence
    Padding is the 
    """
    splitted = x[0].split()
    #wv is the word2vec model that has been trained on the PubMed db
    vector = [wv.word_vec(y) for y in splitted]
    split_len = len(splitted)
    if padding <= split_len:
        vector = vector[0:padding]
    else:
        difference  = padding - split_len
        pad = np.array([0]*200)
        for _ in range(difference):
            vector.append(pad)
    arr = np.concatenate(vector).reshape([padding,200])
    return arr

vectorized = [return_word2vec_padded(t,29) for t in data]
labels = [x[1] for x in data]

pairs = zip(vectorized, labels)

with open('sentence_wordvecs','wb') as f:
    pickle.dump(pairs,f)
    
    
wv = kv.load_word2vec_format("PubMed-and-PMC-w2v.bin", binary=True)
with open('labelled','rb') as f:
    data = pickle.load(f)

def return_word2vec_padded(x, padding):
    """
    Return a numpy array containing the word vectors of a sentence
    Padding is the length of input vector
    """
    splitted = x[0].split()
    #wv is the word2vec model that has been trained on the PubMed db
    vector = [wv.word_vec(y) for y in splitted]
    split_len = len(splitted)
    if padding <= split_len:
        vector = vector[0:padding]
    else:
        difference  = padding - split_len
        pad = np.array([0]*200)
        for _ in range(difference):
            vector.append(pad)
    arr = np.concatenate(vector).reshape([padding,200])
    return arr

padding_list = range(20,50)
vectorized = [[return_word2vec_padded(t,i) for t in data] for i in padding_list]
labels = [x[1] for x in data]