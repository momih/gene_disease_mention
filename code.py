MY_EMAIL = 'youremail@gmail.com'
from Bio import Entrez
from lxml import etree as et
import re
import nltk
import time 
from os import listdir, getcwd, chdir
import pickle

###########################################################
# Define functions to access PubMed through Entrez and 
# download papers
###########################################################

def entrez_search(gene, database, number):
    """
    Takes two strings as input:
    *gene - name of gene
    *database - NCBI database (for this project 
    *only pmc and pubmed considered)
    Returns a list containing the PubMed or PMC ID's for the result
    """
    Entrez.email = MY_EMAIL
    search = Entrez.esearch(db=database, 
                            sort='relevance', 
                            retmax=number,
                            retmode='xml', 
                            term="open access[filter]" + gene,
                            #rettype='full'
                            )
    results = Entrez.read(search)
    id_list = results['IdList']
    return id_list
    
def fetch_pubmed_abstract(ids):
    """
    ONLY FOR PUBMED DATABASE
    Takes a list of ids and returns only the sentences containing the term in 
    the article
    """
    fetch = Entrez.efetch(db='pubmed',
                         resetmode='xml',
                         id=ids,
                         rettype='full')
    papers = Entrez.read(fetch)
    article = papers['PubmedArticle'][0]['MedlineCitation']['Article']
    abstract = article['Abstract']['AbstractText'][0]
    return abstract

def save_as_xml(id_list, batch_number):
    """
    ONLY FOR PMC DATABASE
    Takes a list of ids returned by Entrez and downloads the article to an
    XML file
    """
    ids = ','.join(id_list)
    fetch = Entrez.efetch(db='pmc',
                         resetmode='xml',
                         id=ids)
    filename = "PMC" + str(batch_number) + ".nxml"
    with open(filename, "w") as f:
        f.write(fetch.read())
        
def download_pmc(gene_list, no_of_papers):
    """
    Takes a list of genes and downloads a specified number of papers with
    those genes in them
    """
    id_list = []   
    for i in gene_list:
        paperID = entrez_search(i, 'pmc', no_of_papers)
        id_list.extend(paperID)
    id_batches = [id_list[i:i+100] for i in range(0, len(id_list), 100)]  
    batch_index = 0
    for id_sublist in id_batches:
        start = time.time()
        batch_index += 1
        save_as_xml(id_sublist, batch_index)
        end = time.time() - start
        print "Time for batch %i is -  %f \n" % (batch_index, end) 
 
###########################################################
# Functions to parse the XML files (a single one 
# or in batches) and extract relevant sentences
###########################################################

def parse_pmc_xml(file):
    """
    Takes an XML file of the and returns the full text of the article
    """
    xml = et.parse(file)
    xml_article = xml.findall("article")
    #article = ''.join([x for x in xml_article.itertext()]).strip()\
                #.replace("\n",' ') 
    article = []
    for i in xml_article:
        j = i.find("body")
        if (j == None):
            continue
        else:
            article.append(''.join([x for x in j.itertext()]).strip()\
                                             .replace("\n",' '))
    print "\t" + str(file)
    return article

def parse_batches():
    """
    Takes all xml files in a folder and creates batches to parse and saves them
    as pickled objects named after their batch
    """
    list_of_files = listdir(getcwd())
    file_batches = [list_of_files[i:i+100] for i in range(0, 
                len(list_of_files), 100)]
    b_i = 0
    for batch in file_batches:
        article_texts = [parse_pmc_xml(x) for x in batch]
        names = '/home/momi/Documents/526/Project/PMC_papers/Pickled/batch_'+str(b_i)
        with open(names, 'wb') as fp:
            pickle.dump(article_texts, fp)                
        b_i += 1
        print b_i    

def extract_sentences(text, term_list):
    """
    Takes an article/abstract and a search term and returns the sentences in
    the text that contain the term.
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    splitted = tokenizer.tokenize(text)
    sentences = [x for x in splitted if any(word in x for word in term_list)]
    return sentences
  
def extract_batches(gene_list):
    """
    When run from a directory containing pickled parsed PMC articles, it takes
    each pickled file and loads it into memory and extracts sentences 
    containing the gene mentions
    """
    list_of_files = listdir(getcwd())
    batch_i = 0
    for pickled in list_of_files: 
        #disease_list.extend([x + "s" for x in disease_list])
        with open (pickled, 'rb') as f:
            article = pickle.load(f)
        article_texts = [item for sublist in article for item in sublist if sublist]
        texts = [extract_sentences(x, gene_list) for x in article_texts]
        sentences = [item for sublist in texts for item in sublist if sublist]
        with open('sentences'+ str(batch_i), 'wb') as f:
            pickle.dump(sentences, f)  
        batch_i += 1
        print batch_i

###########################################################
# Functions to annotate the sentences with the gene and 
# disease
###########################################################
        
def annotate_labels(text, gene_list, disease_list):
    """
    Finds the first mentioned gene and disease in the sentence and
    replaces them with a placeholder word so labelling is easier.
    Returns a tuple with (gene, disease, replaced_text)
    """
    try:
        gene = list(set(text.split(" ")).intersection(gene_list))[0]
        text = text.replace(gene, "gene_placeholder")
    except:
        gene = 'None'
    try:
        disease = list(set(text.split(" ")).intersection(disease_list))[0]
        text = text.replace(disease, "disease_placeholder")
    except:
        disease = 'None'
    return gene, disease, text
    
    
def gene_disease(gene_list, disease_list):
    """
    Takes all the files in the directory that contains pickled objects 
    containing the sentences and returns an annotated version
    """
    list_of_files = listdir(getcwd())
    gene_mentions = []
    for files in list_of_files:
        with open(files, 'rb') as f:
            sentences = pickle.load(f)
        sentences = [re.sub(ur"[.,\(\)?!;]","",x.lower()) for x in sentences]
        gene_mentions.extend(sentences)
    annotated = [annotate_labels(x, gene_list, disease_list) for x in gene_mentions]
    pairs = [x for x in annotated 
                 if x[0] is not 'None' and x[1] is not 'None']
    return pairs

def get_names():
    with open("cancer_genes.txt", "r") as f:
        gene_list = list(set(f.read().lower().split(" ")))
    with open('cancer_names.txt', 'r') as f:
        cancer = [x.strip() for x in f.read().lower().split("\n")]    
                  
    names = [x.split(" ")[-1].lower() for x in cancer]
    names.extend([x for x in cancer if len(x.split(" ")) <= 1])
    disease_list = list(set(names))
    #removing meaningless values
    disease_list.remove('')
    disease_list.remove('ii')
    #adding plural terms to original disease list
    disease_list.extend([x + "s" for x in disease_list])
    return gene_list, cancer

