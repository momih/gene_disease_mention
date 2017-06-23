import pickle
import nltk
from gensim.models.keyedvectors import KeyedVectors as kv
import numpy as np
with open('sentences_filtered_final', 'rb') as f:
    sentences = pickle.load(f) 

def label(x):
    splitted = x[2].split()
    gp_i = splitted.index('gene_placeholder')
    dp_i = splitted.index('disease_placeholder')
    phrase_list = splitted[gp_i:dp_i]
    y = x[2].replace('gene_placeholder',x[0])\
                .replace('disease_placeholder',x[1])
    
    # Label 0 for lack of clear conclusions
    clarity = 'unclear unsure inconclusive unknown'.split()
    if any(word in x[2] for word in clarity):
        return (y,0)
    # Label 0 for too many words between gene and disease
    elif len(phrase_list) > 20:
        return (y,0)
    # Label 1 if target, bind etc is in between gene and disease
    elif len(phrase_list) < 12:
        bw_terms = 'target inhibit bind express associat influenc'.split()
        phrase = ' '.join(phrase_list)
        if any(word in phrase for word in bw_terms):
            return (y, 1)
    # Label 1 if no of words bw gene and disease <5 and any word is a verb
    elif len(phrase_list) < 5:
        tagged = nltk.pos_tag(phrase_list)
        if any('V' in x for x in tagged[1]):
            return (y, 1)
    else:
        pass
    
labelled = [label(x) for x in sentences if label(x) is not None]

#removing sentences which dont have words in word2vec vocab
#word2vec vocab is saved in pmw2v_vocab
with open('pmw2v_vocab', 'rb') as f:
    pmvocab = pickle.load(f)
corpus_vocab = ' '.join([x[0] for x in labelled]).split()
difference = list(set(corpus_vocab) - set(pmvocab))

#removing sentences which dont have words in the w2v vocab
labelled = [x for x in labelled if not any(word in x[0] for word in difference)]

#saving
with open('labelled','wb') as f:
    pickle.dump(labelled,f)

    
#saving word2vec representations of the data
with open('labelled','rb') as f:
    data = pickle.load(f)
    
def return_word2vec_padded(x, padding):
    splitted = x.split()
    vector = [wv.word_vec(y) for y in splitted]
    difference  = padding - len(splitted)
    pad = np.array([0]*200)
    for _ in range(difference):
        vector.append(pad)
    arr = np.concatenate(vector).reshape([padding,200])
    return arr
    
vectorized = [return_word2vec_padded(t,50) for t in data]
              
with open('t','wb') as f:
    pickle.dump(w,f)