#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import csv
import time
import jieba
import pprint
import re, string
import numpy as np
from gensim import corpora
# from threading import Semaphore, Thread, Event # for multi-processing
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
from smart_open import open # for transparently opening compressed files
import multiprocessing as mp # pool, pipe, process for multi-processing
from itertools import repeat, islice # itertools.islice, for slicing generators


# ## Number of Cores

# In[188]:


NCORE = input("Enter Number of Processes")
NUM_FILES = input("Enter Number of Files")


# ## Parallel Process Function

# In[189]:


def calc_given_keywords(words, expanded_nclass, expanded_keywords):
    """Calculate frequency given keywords and return it"""
    freq = np.array([0 for _ in range(expanded_nclass)]) # easy to add up
    for word in words:
        for i, category in enumerate(expanded_keywords, 1):
            if word in category:
                freq[i-1] += 1 # -1 for the right index
    return freq


# In[190]:


def parallel_process(document, city_link, expanded_keywords):
    words = document[0]
    cities = document[1]
    # frequency in the format: array([0, 0, 0, 0, 0, 0, 0]), easy to add up
    _freq = calc_given_keywords(words, len(expanded_keywords), expanded_keywords)
    # combine "A-B" and "B-A" city pairs
    for i in range(len(cities)): 
        for j in range(len(cities)):
            if i != j:
                if (cities[i], cities[j]) in city_link:
                    city_link[(cities[i], cities[j])] += _freq
                if (cities[j], cities[i]) in city_link:
                    city_link[(cities[j], cities[i])] += _freq
    return city_link


# # Main

# #### File List

# In[191]:


file_list = [f for f in os.listdir('../webdata') if f.startswith('part-')][:1]


# #### Dictionary

# In[192]:


print('loading dictionary...')
if 'dict.dict' in os.listdir('../dict'):
    dictionary = corpora.Dictionary().load('../dict/dict.dict')  # already processed from embedding file
else:
    texts = []
    with open('../embedding/Tencent_AILab_ChineseEmbedding.txt') as f:
        skip_head = True
        for line in f:
            if skip_head:
                skip_head = False
                continue
            else:
                texts.append(line.split(' ')[0])
    dictionary = corpora.Dictionary([texts])
    dictionary.save('../dict/dict.dict')
print(dictionary)


# #### Stop List

# In[193]:


stop_list = []
with open('resources/stopwords_zh.txt') as f:
    for line in f:
        stop_list.append(line[:-1])
stop_list = set(stop_list)


# #### City List

# In[194]:


city_list = []
with open('resources/China_Cities_Coordinates_CHN_ENG.csv') as f:
    skip_head = True
    for line in f:
        if skip_head:
            skip_head = False
            continue
        else:
            city_list.append(line.split(',')[0])
city_list = list(set(city_list))


# #### Save 'Bin_Tencent_AILab_ChineseEmbedding.bin' in '../embedding'

# In[195]:


if 'Bin_Tencent_AILab_ChineseEmbedding.bin' not in os.listdir('../embedding'):
    embedding_file = '../embedding/Tencent_AILab_ChineseEmbedding.txt'
    wv = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    wv.init_sims(replace=True)
    wv.save('../embedding/Bin_Tencent_AILab_ChineseEmbedding.bin')


# #### Load Word Embeddings

# In[ ]:


print('loading word embeddings...')
wv = KeyedVectors.load('../embedding/Bin_Tencent_AILab_ChineseEmbedding.bin', mmap='r')
wv.vectors_norm = wv.vectors  # prevent recalc of normed vectors


# #### Save 'expanded_keywords.csv' in 'resources'

# In[ ]:


if 'expanded_keywords.csv' not in os.listdir('resources'):
    # Expand the existing keywords by finding words in the embedding file that are above the threshold
    '''load keywords'''
    nclass = 7
    keywords = [[] for _ in range(nclass)]
    with open('resources/keywords.csv') as f:
        for line in f:
            line = line.replace('\n', '')
            for i, category in enumerate(line.split(',')):
                if category != '' and category in wv:
                    keywords[i].append(category)

    '''save expanded keywords'''
    expanded_nclass = 7
    expanded_keywords = [[] for _ in range(expanded_nclass)]
    for i, category in enumerate(keywords, 1):
        for key in category:
            # get most similar words to keys whose similarity > threshold
            expanded_keywords[i-1].append(key)
            closest = wv.most_similar(key)
            for sim_pair in closest:
                if sim_pair[1] > 0.8:
                    expanded_keywords[i-1].append(sim_pair[0])
                else:
                    break
    with open('resources/expanded_keywords.csv', "w") as f:
        writer = csv.writer(f, delimiter=',')
        for category in expanded_keywords:
            writer.writerow(category)


# #### Load Expanded Keywords

# In[ ]:


print('loading keywords...')
if 'expanded_keywords.csv' in os.listdir('resources'): # already expanded, load from saved
    expanded_nclass = 7
    expanded_keywords = [[] for _ in range(expanded_nclass)]
    with open('resources/expanded_keywords.csv') as f:
        for i, line in enumerate(f):
            line = line.replace('\n', '')
            line = line.split(',')
            for keyword in line:
                expanded_keywords[i].append(keyword)
else:
    print('Error: Expanded keywords not found')


# #### Get Documents

# In[ ]:


print('loading {} files...'.format(NUM_FILES))
documents = [] # [document[[words],[cities]]]
jieba.enable_parallel(NCORE)
for filename in file_list[:NUM_FILES]:
    with open('../webdata/' + filename, encoding='utf-8') as f:
        for line in f:
            document = []
            # drop meta-info
            if line == '' or line.startswith('\r') or line.startswith('WARC') or line.startswith('Content'):
                continue
            # drop alphabetic characters
            line = re.sub(r'[a-zA-Z]', '', line)
            # drop digits and punctuations
            line = re.sub('[%s]' % (string.punctuation + string.digits), '', line)
            # drop empty line
            if line == '\r':
                continue
            # segment the sentence using jieba
            words = ' '.join(jieba.cut(line, cut_all=False)).split(' ')
            # drop stopwords
            words = [word for word in words if word not in stop_list]
            if not words or len(words) < 2:  # less than 2 words won't contain 2 cities
                continue
            cities = []
            indices = []
            for idx, word in enumerate(words):
                if word in city_list:
                    cities.append(word)
                    indices.append(idx)
            # remove cities from the document
            for idx in indices[::-1]:
                del words[idx]
            cities = list(set(cities)) # get unique cities
            if len(cities) < 2:  # less than 2 cities won't composite a link
                continue
            document.append(words)
            document.append(cities)
            documents.append(document)
jieba.disable_parallel()


# #### mp.manager (slow)/mp.pool.apply_async (serial) if used

# In[ ]:


# manager = mp.Manager()
# documents_ref = manager.list(documents)
# city_link_ref = manager.dict(city_link)
# expanded_keywords_ref = manager.list(expanded_keywords)

## Instantiate the pool here
# pool = mp.Pool(processes=NCORE)
# pool.apply_async(parallel_process, (documents_ref[(NCORE-1)*NDOC:], 
#                                     city_link_ref, expanded_keywords_ref))
# for core in range(NCORE-1):
#     pool.apply_async(parallel_process, (documents_ref[core*NDOC:(core+1)*NDOC],
#                                         city_link_ref, expanded_keywords_ref))

# results = [pool.apply_async(parallel_process, (documents[core*NDOC:(core+1)*NDOC], 
#                                                city_link, expanded_keywords)).get() 
#            if core < NCORE-1 else pool.apply_async(parallel_process, (documents[(NCORE-1)*NDOC:], 
#                                     city_link, expanded_keywords)).get()
#           for core in range(NCORE)]


# ### Main Run Part

# In[ ]:


# Initialise City Link
city_link = {}
expanded_nclass = len(expanded_keywords)
for i in range(len(city_list)-1):
    for j in range(i+1,len(city_list)):
        city_link[(city_list[i], city_list[j])] = np.array([0 for _ in range(expanded_nclass)]) # easy to add up

start = time.time()
NDOC = len(documents)//NCORE # num_documents_per_process

print('Start {} processes...'.format(NCORE))
# Instantiate the pool here
pool = mp.Pool(processes=NCORE)
result = pool.starmap_async(parallel_process, zip(documents, repeat(city_link), repeat(expanded_keywords)), NDOC)
pool.close()
pool.join()

result = result.get()

print('Get individual result from each process after {} seconds elapsed.'.format(NCORE, time.time() - start))

for key in city_link.keys():
    saved_previous = np.array([-1 for _ in range(expanded_nclass)])
    for i, res in enumerate(result):
        if not np.array_equal(saved_previous,res[key]):
            saved_previous = res[key]
            city_link[key] += res[key]

print('Get final results after {} seconds elapsed.'.format(time.time() - start))

print('Saving...')
with open('results/city_link_frequency_multiThreads.csv', "w") as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(('City1','City2','经济','科技','法律','文学','娱乐','第二产业','农业')) # first row as header
    for key, value in city_link.items():
        writer.writerow((key[0], key[1], value[0], value[1], value[2], value[3], value[4], value[5], value[6]))
print('Done.')


# In[ ]:




