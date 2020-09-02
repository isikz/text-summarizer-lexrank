import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import math



## Method to lowercase the terms, remove stop words, remove punctuations
def remove_stop_words(abstract):
    stop_words = set(stopwords.words('english'))
    refined_abs = []
    uniq_words = set(RegexpTokenizer('\w+').tokenize(abstract))
    for w in uniq_words:
        if w.casefold() not in stop_words:
            refined_abs.append(w.lower())
    
    return refined_abs

## Create document frequency of a term dictionary
def calculate_dict(abstracts):
    nan=0
    word_dict = {}
    for abstract in abstracts:
        if pd.isnull(abstract):
            nan+=1
        else:
            abstract = remove_stop_words(abstract)
            for word in abstract:
                if word in word_dict:
                    word_dict[word] +=1
                else:
                    word_dict[word] = 1
    
    return word_dict, nan, len(abstracts)

## Calculate idf scores and create dictionary of it
def idf(word_dict, nan, len_abstracts):
    idf = {}
    for key, value in word_dict.items():
        idf[key] = math.log10((len_abstracts-nan)/value)
    return idf

## Get ID and abstracts of relevant documents for topic 1,3,13
def get_relevance(infile, id_abstracts):
    inp = open(infile, "r", encoding="utf-8")
    inp_lines = inp.readlines()
    
    topic_1 = {}
    topic_3 = {} ## topic that I have chosen
    topic_13 = {} 
    
    for line in inp_lines:
        elements = line.split()
       
        if elements[3] == "2": ## fully relevant
            if elements[0] == "1": ## relevant abstract is about topic 1
                ind = np.where(id_abstracts[:,0] == elements[2])
                topic_1[elements[2]] = id_abstracts[ind[0][0],1]
            if elements[0] == "3": ## relevant abstract is about topic 3, my choice
                ind = np.where(id_abstracts[:,0] == elements[2])
                topic_3[elements[2]] = id_abstracts[ind[0][0],1]
            if elements[0] == "13": ## relevant abstract is about topic 13
                ind = np.where(id_abstracts[:,0] == elements[2])
                topic_13[elements[2]] = id_abstracts[ind[0][0],1]
                
    return topic_1, topic_3, topic_13

## Create documenst vectors for similarity calculation
def create_document_vectors(topic_x, word_dict, idf):
    doc_vectors = []
    for abstract in topic_x:
        sub_vector = []
        if isinstance(abstract, str) is False:
            sub_vector = [0]*len(idf)
        else: 
            doc_words = remove_stop_words(abstract)
            doc_length = len(doc_words)
            for key in word_dict.keys():
                if(doc_words.count(key)==0):
                    sub_vector.append(0)
                else:
                    tf = doc_words.count(key)/doc_length
                    sub_vector.append(tf*idf[key])
                
            
        doc_vectors.append(sub_vector)
    
    return doc_vectors

## Create link matrix of documents
def link_matrix(doc_vectors,thr):
    link_matrix = np.zeros((len(doc_vectors), len(doc_vectors)))
    for i in range(len(doc_vectors)):
        for j in range(len(doc_vectors)):
            if i==j:
                link_matrix[i][j] = 0
            else:
                #cos_sim = np.dot(doc_vectors[i],doc_vectors[j])/(np.linalg.norm(doc_vectors[i])*np.linalg.norm(doc_vectors[j]))
                cos_sim = similarity(doc_vectors[i], doc_vectors[j])
                if cos_sim < thr:
                    link_matrix[i][j] = 0
                else:
                    link_matrix[i][j] = 1
    return link_matrix

## Calculate similarity
def similarity(a,b):
    summation = 0
    norm_a = 0
    norm_b = 0
    for x in range(len(a)):
        norm_a += a[x]*a[x]
        norm_b += b[x]*b[x]
        summation += a[x]*b[x]
    
    return summation/(math.sqrt(norm_a)*math.sqrt(norm_b) + 0.000001) 
                
## Create teleportation matrix of documents with 0.15 teleportation rate
def transition_matrix(link_matrix, tel_rate):
    P = np.zeros((link_matrix.shape[0],link_matrix.shape[0]))
    for i in range(link_matrix.shape[0]):
        count_1 = 0
        for j in range(link_matrix.shape[0]):
            if link_matrix[i][j] == 1:
                count_1 = count_1 + 1
                
        for j in range(link_matrix.shape[0]):
            if link_matrix[i][j] == 1:
                P[i][j] = (1-tel_rate)/count_1 + (tel_rate/(link_matrix.shape[0]))
            else:
                P[i][j] =  tel_rate/(link_matrix.shape[0])
    return P
                
# Find steady state a            
def power_method(a, P, err, i):
    a_new = np.dot(a, P)
    comp = np.equal(a, a_new)
    same_rate = np.mean(comp.astype(np.float32))
    
    if i >= 999: ## recursion limit is 1000
        return a_new
    else:
        if 1-same_rate < err:
            return a_new
        else:
            return power_method(a_new, P, err, i+1)
    
# Find top 10 abstracts
def top_10_abstract(a, topic_x):
    abstracts = np.asarray(list(topic_x.values()))
    idx = a.argsort()[::-1]
    a = a[idx]
    abstracts = abstracts[idx]
    abs_10 = abstracts[8:18]
    a = a[8:18]
    return abs_10, a

# Tokenize sentence to abstracts
def abstract_sentences(top_10_abs):
    sentences = []
    for abst in top_10_abs:
        sentences = sentences + sent_tokenize(abst)
    return sentences

# Find top 20 sentences        
def top_20_sent(a, sent):
    sentences = np.asarray(sent)
    idx = a.argsort()[::-1]
    a = a[idx]
    sentences = sentences[idx]
    sent_20 = sentences[0:20]
    a = a[0:20]
    return sent_20, a
        


tel_rate = 0.15 #teleportation rate
err = 0.00001 #error tolerance
thr = 0.10
## READ FILES
meta_data = pd.read_csv('/2020-04-10/metadata.csv')
i_a = meta_data[['cord_uid', 'abstract']]
id_abstracts = pd.DataFrame(i_a).to_numpy()

## RUN METHODS
word_dict, nan, len_abstracts = calculate_dict(id_abstracts[:,1])
idf = idf(word_dict, nan, len_abstracts)
#get_topics()
topic_1, topic_3, topic_13 = get_relevance('qrels-rnd1.txt', id_abstracts[:,:])
## TOPIC 1
i=0
doc_vec_1 = create_document_vectors(topic_1.values(), word_dict, idf)
link_mat_1 = link_matrix(doc_vec_1, thr)
P_1 = transition_matrix(link_mat_1, tel_rate)
a_init_1 = np.asarray([tel_rate/len(link_mat_1)]*len(link_mat_1))
a_final_1 = power_method(a_init_1, P_1, err,i)
abs_top_1, a_top_1 = top_10_abstract(a_final_1, topic_1)

id_list = []
for ab in abs_top_1:
    ind = np.where(id_abstracts[:,1] == ab)
    id_list.append(id_abstracts[ind[0][0],0])
    
i=0
sent_1 = abstract_sentences(abs_top_1)
sent_vec_1 = create_document_vectors(sent_1, word_dict, idf)
link_sent_1 = link_matrix(sent_vec_1, thr)
P_sent_1 = transition_matrix(link_sent_1, tel_rate)
a_sent_init_1 = np.asarray([tel_rate/len(link_sent_1)]*len(link_sent_1))
a_sent_final_1 = power_method(a_sent_init_1, P_sent_1, err, i)
sent_top_1, a_sent_1 = top_20_sent(a_sent_final_1, sent_1)

file_1 = open("top_20_1.txt", "w+", encoding="utf-8")
index=1
for s in sent_top_1:
    w = str(index) + ": " +s + "\n"
    file_1.write(w)
    index=index+1
file_1.close()


## TOPIC 3
"""
i=0
doc_vec_3 = create_document_vectors(topic_3.values(), word_dict, idf)
link_mat_3 = link_matrix(doc_vec_3, thr)
P_3 = transition_matrix(link_mat_3, tel_rate)
a_init_3 = np.asarray([tel_rate/len(link_mat_3)]*len(link_mat_3))
a_final_3 = power_method(a_init_3, P_3,err,i)
abs_top_3, a_top_3 = top_10_abstract(a_final_3, topic_3)

id_list = []
for ab in abs_top_3:
    ind = np.where(id_abstracts[:,1] == ab)
    id_list.append(id_abstracts[ind[0][0],0])
    
i=0
sent_3 = abstract_sentences(abs_top_3)
sent_vec_3 = create_document_vectors(sent_3, word_dict, idf)
link_sent_3 = link_matrix(sent_vec_3, thr)
P_sent_3 = transition_matrix(link_sent_3, tel_rate)
a_sent_init_3 = np.asarray([tel_rate/len(link_sent_3)]*len(link_sent_3))
a_sent_final_3 = power_method(a_sent_init_3, P_sent_3, err, i)
sent_top_3, a_sent_3 = top_20_sent(a_sent_final_3, sent_3)

file_3 = open("top_20_3.txt", "a", encoding="utf-8")
index=1
for s in sent_top_3:
    file_3.write(str(index) + ": " +s + "\n")
    index=index+1
file_3.close()
"""

## TOPIC 13
"""
i=0
doc_vec_13 = create_document_vectors(topic_13.values(), word_dict, idf)
link_mat_13 = link_matrix(doc_vec_13, thr)
P_13 = transition_matrix(link_mat_13, tel_rate)
a_init_13 = np.asarray([tel_rate/len(link_mat_13)]*len(link_mat_13))
a_final_13 = power_method(a_init_13, P_13,err,i)
abs_top_13, a_top_13 = top_10_abstract(a_final_13, topic_13)

id_list = []
for ab in abs_top_13:
    ind = np.where(id_abstracts[:,1] == ab)
    id_list.append(id_abstracts[ind[0][0],0])

i=0
sent_13 = abstract_sentences(abs_top_13)
sent_vec_13 = create_document_vectors(sent_13, word_dict, idf)
link_sent_13 = link_matrix(sent_vec_13, thr)
P_sent_13 = transition_matrix(link_sent_13, tel_rate)
a_sent_init_13 = np.asarray([tel_rate/len(link_sent_13)]*len(link_sent_13))
a_sent_final_13 = power_method(a_sent_init_13, P_sent_13, err, i)
sent_top_13, a_sent_13 = top_20_sent(a_sent_final_13, sent_13)

file_13 = open("top_20_13.txt", "a", encoding="utf-8")
index=1
for s in sent_top_13:
    file_13.write(str(index) + ": " +s + "\n")
    index=index+1
file_13.close()
"""
