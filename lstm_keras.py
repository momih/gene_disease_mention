import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from gensim.models.keyedvectors import KeyedVectors as kv
from random import sample
import random

# load pubmed word2vec model
wv = kv.load_word2vec_format("PubMed-and-PMC-w2v.bin", binary=True)

# labelled data is saved in pickled object called labelled
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

#with open('sentence_wordvecs','rb') as f:
#    a = pickle.load(f)
#a has 1000 tuples consisting of (sentence, label) pair
# where sentence is in vector representation

def create_validation_split(data, labels, percent, seed):
    random.seed(seed)
    nrows = len(data)
    train_len = int(percent/100.0 * nrows)
    test_len = nrows-train_len
    train_indices = sorted(sample(range(nrows),train_len))
    test_indices = list(set(range(nrows)) - set(train_indices))
    X_train = [data[x] for x in train_indices]
    Y_train = [labels[x] for x in train_indices]
    X_test = [data[x] for x in test_indices]
    Y_test = [labels[x] for x in test_indices]
    return train_len, test_len, X_train, Y_train, X_test, Y_test

def get_accuracy(X_train, Y_train, X_test, Y_test, padding,
                 train_len, test_len, epochs):
    model = Sequential()
    model.add(LSTM(27,input_shape=(padding,200)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split=0.2,
              shuffle=True)
    # Final evaluation of the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return scores[1]*100

acc_list = []
for i in range(40,41):
    vectorized = [return_word2vec_padded(t,i) for t in data]
    labels = [x[1] for x in data]
    train_len, test_len, X_train, Y_train, X_test, Y_test = create_validation_split(vectorized,labels, 70, 2)
    # reshaping the sentence vectors into 
    # (no_of_sentences,sentence_length,word_vector_size) using numpy
    X_train = np.concatenate(X_train).reshape([train_len,i,200])
    X_test = np.concatenate(X_test).reshape([test_len,i,200])
    acc = get_accuracy(X_train, Y_train, X_test, Y_test, i,
                       train_len, test_len, 10)
    acc_list.append((i,acc))
    
with open('accuracies_one','wb') as f:
    pickle.dump(acc_list, f)
