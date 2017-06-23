import pickle
import re
import nltk
#from gensim.models.keyedvectors import KeyedVectors as kv

with open('mentions', 'rb') as f:
    mentions = pickle.load(f)
    
filtered = [x for x in mentions if len(x[2].split(" ")) < 100 
                       and not re.search("\s{3,}", x[2])]
filtered = [(x[0], x[1],re.sub("\\[[0-9]{1,}\\]", "", x[2])) for x in filtered]

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
        return [y,0]
    # Label 0 for too many words between gene and disease
    elif len(phrase_list) > 20:
        return [y,0]
    # Label 1 if target, bind etc is in between gene and disease
    elif len(phrase_list) < 12:
        bw_terms = 'target inhibit bind express associat influenc'.split()
        phrase = ' '.join(phrase_list)
        if any(word in phrase for word in bw_terms):
            return [y, 1]
    # Label 1 if no of words bw gene and disease <5 and any word is a verb
    elif len(phrase_list) < 5:
        tagged = nltk.pos_tag(phrase_list)
        if any('V' in x for x in tagged[1]):
            return [y, 1]
    else:
        pass
    
labelled = [label(x) for x in filtered if label(x) is not None]
            
with open('pmw2v_vocab', 'rb') as f:
    pmvocab = pickle.load(f)
corpus_vocab = ' '.join([x[0] for x in labelled]).split()
difference = list(set(corpus_vocab) - set(pmvocab))

#removing sentences which dont have words in the w2v vocab
labelled = [x for x in labelled if not any(word in x[0] for word in difference)]

with open('labelled', 'wb') as f:
    pickle.dump(labelled, f)     
