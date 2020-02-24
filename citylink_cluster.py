#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import time
import jieba
import pprint
import re, string
import numpy as np
from gensim import corpora
from threading import Semaphore
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
from smart_open import open # for transparently opening compressed files


# ### Corpus class for getting one document at a time from the whole dataset

# In[2]:


class MyCorpus:
    """Corpus that handles one document at a time
    """
    
    def __init__(self, root_path, file_list, dictionary, stop_list, city_list):
        """
        Args:
            root_path - root path to the website files
            file_list - list of website files
            dictionary - mapping from words to ids
            stop_list - chinese stopwords set
            city_list - chinese city set
        """
        self.root_path = root_path
        self.file_list = file_list
        self.dictionary = dictionary
        self.stop_list = stop_list
        self.city_list = city_list
    
    def __iter__(self):
        for filename in self.file_list:
            with open(self.root_path + filename, encoding='utf-8') as f:
                for line in f:
                    words = self._process(line)
                    if not words or len(words) < 2:  # less than 2 words won't contain 2 cities
                        continue
                    words, cities = self._retrieve_cities(words)
                    if len(cities) < 2:  # less than 2 cities won't composite a link
                        continue
#                     yield {'words': self.dictionary.doc2bow(words), 'cities': cities}
                    yield {'words': words, 'cities': cities}
                    
    def _process(self, line):
        # drop meta-info
        if line == '' or line.startswith('\r') or line.startswith('WARC') or line.startswith('Content'):
            return
        # drop alphabetic characters
        line = re.sub(r'[a-zA-Z]', '', line)
        # drop digits and punctuations
        line = re.sub('[%s]' % (string.punctuation + string.digits), '', line)
        # drop empty line
        if line == '\r':
            return
        # segment the sentence using jieba
        words = ' '.join(jieba.cut(line, cut_all=False)).split(' ')
        # drop stopwords
        words = [word for word in words if word not in self.stop_list]
        return words
    
    def _retrieve_cities(self, words):
        """Caution: cities are removed from the document because cities are not supposed to related to any category
        """
        cities = []
        indices = []
        for idx, word in enumerate(words):
            if word in self.city_list:
                cities.append(word)
                indices.append(idx)
        # remove cities from the document
        for idx in indices[::-1]:
            del words[idx]
        return words, cities


# ### Class that calculates the frequency of every category in a document

# In[119]:


class FrequencyCalculator:
    """Calculate frequency of every category in a document by compare the similarities betweeen each word and the keyword
    """
    
    def __init__(self, wv, nclass, keywords, th):
        """
        Args:
            wv - word vectors
            nclass - number of categories
            keywords - 文化，经济，体育等. Nested list. [[经济,金融...],[科技,互联网...]...], 一共nclass类
            th - similarity threshold. Similarites Under the threshold will be discarded.
        """
        self.wv = wv
        self.nclass = nclass
        self.keywords = keywords
        self.th = th
    
    def calc(self, words):
        """Calculate frequency and return it"""
        freq = {i: 0 for i in range(1, self.nclass + 1)}
        for word in words:
            if word not in self.wv:
                continue
            for i, category in enumerate(self.keywords, 1):
                for key in category:
                    if self.wv.similarity(word, key) > self.th:
                        freq[i] += 1
        return freq
    
    def calc_given_keywords(self, words, expanded_keywords):
        """Calculate frequency given keywords and return it"""
        freq = np.array([0 for _ in range(expanded_nclass)]) # easy to add up
        for word in words:
            if word not in self.wv:
                continue
            for i, category in enumerate(expanded_keywords, 1):
                if word in category:
                    freq[i-1] += 1 # -1 for the right index
        return freq


# ### Class that converts frequency of every document to frequency of two city link in the whole corpus

# In[4]:


class FrequencyConverter


# ### Load the dictionary

# In[11]:


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


# ### Load the word embeddings

# In[8]:


if 'Bin_Tencent_AILab_ChineseEmbedding.bin' not in os.listdir('../embedding'):
    embedding_file = '../embedding/Tencent_AILab_ChineseEmbedding.txt'
    wv = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    wv.init_sims(replace=True)
    wv.save('../embedding/Bin_Tencent_AILab_ChineseEmbedding.bin')

wv = KeyedVectors.load('../embedding/Bin_Tencent_AILab_ChineseEmbedding.bin', mmap='r')
wv.vectors_norm = wv.vectors  # prevent recalc of normed vectors
wv.most_similar('stuff')  # any word will do: just to page all in


# ### Load the stopwords, city names and keywords

# In[84]:


stop_list = []
with open('resources/stopwords_zh.txt') as f:
    for line in f:
        stop_list.append(line[:-1])
stop_list = set(stop_list)

city_list = []
with open('resources/China_Cities_Coordinates_CHN_ENG.csv') as f:
    skip_head = True
    for line in f:
        if skip_head:
            skip_head = False
            continue
        else:
            city_list.append(line.split(',')[0])
city_list = set(city_list)  

nclass = 7
keywords = [[] for _ in range(nclass)]
with open('resources/keywords.csv') as f:
    for line in f:
        line = line.replace('\n', '')
        for i, category in enumerate(line.split(',')):
            if category != '' and category in wv:
                keywords[i].append(category)


# ### City Links

# In[139]:


# get city links
city_l = list(city_list)
city_link = {}
for i in range(len(city_l)-1):
    for j in range(i+1,len(city_l)):
        city_link[(city_l[i], city_l[j])] = np.array([0 for _ in range(expanded_nclass)]) # easy to add up
city_link[('郴州', '许昌')]


# ### Instantiate the corpus and frequency calculator

# In[120]:


file_list = [f for f in os.listdir('../webdata') if f.startswith('part-')][:1]
my_corpus = MyCorpus('../webdata/', file_list, dictionary, stop_list, city_list)
freq_calc = FrequencyCalculator(wv, nclass=nclass, keywords=keywords, th=0.8)


# ### Main run part

# In[142]:


# frequency = {i: 0 for i in range(1, nclass + 1)}  # final frequency
frequency = np.array([0 for _ in range(expanded_nclass)]) # easy to add up

start = time.time()

cnt = 0
for document in my_corpus:
    if cnt > 18000:
        break
    cnt += 1
    _freq = freq_calc.calc_given_keywords(document["words"], expanded_keywords)
    # update final frequency
    frequency += _freq
    # update city_link
    current_city_list = list(set(document["cities"]))
    for i in range(len(current_city_list)-1):
        for j in range(i+1,len(current_city_list)):
            if (current_city_list[i], current_city_list[j]) in city_link:
                city_link[(current_city_list[i], current_city_list[j])] += _freq
            if (current_city_list[j], current_city_list[i]) in city_link:
                city_link[(current_city_list[j], current_city_list[i])] += _freq

print(frequency)

with open('results/city_link_frequency.csv', "w") as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(('City1','City2','经济','科技','法律','文学','娱乐','第二产业','农业'))
    for key, value in city_link.items():
        writer.writerow((key[0], key[1], value[0], value[1], value[2], value[3], value[4], value[5], value[6]))
    
end = time.time()


# In[143]:


print('{} documents (websites) in total. {} (avg: {}) seconds elapsed.'.format(cnt, end - start, (end - start) / cnt))


# ### Cluster the keywords with k-means

# In[ ]:


keys = []
for cate in keywords:
    keys.extend(cate)
print(keys)

X = np.array([wv[k] for k in keys])
kmeans = KMeans(n_clusters=7, random_state=0, verbose=1).fit(X)
# kmeans.labels_[:]


# ### Use t-SNE to project vectors on a 2-D plane

# In[ ]:


# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# In[ ]:


X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded.shape


# ### Find a Common Threshold

# In[40]:


max_similarity = -1
repeated_keys = []
for i in range(6):
    for key in keywords[i]:
        for j in range(i+1,7):
            for oppo_key in keywords[j]:
                if key != oppo_key:
                    if wv.similarity(key, oppo_key) > max_similarity:
                        max_similarity = wv.similarity(key, oppo_key)
                        key1 = key
                        key2 = oppo_key
                else:
                    repeated_keys.append(key)
repeated_keys = set(repeated_keys) # used for later to filter repeated keywords
midpoint = (wv[key1] + wv[key2])/2 # middle vector of the two words
threshold = wv.most_similar(positive=[midpoint])[0][1]
print(threshold)


# ### Get an Expanded Keywords List Based on the Thresold)

# In[133]:


if 'expanded_keywords.csv' in os.listdir('resources'): # already expanded, load from saved
    expanded_nclass = 7
    expanded_keywords = [[] for _ in range(expanded_nclass)]
    with open('resources/expanded_keywords.csv') as f:
        for i, line in enumerate(f):
            line = line.replace('\n', '')
            line = line.split(',')
            for keyword in line:
                expanded_keywords[i].append(keyword)
else: # Expand the existing keywords by finding words in the embedding file that are above the threshold
    expanded_nclass = 7
    expanded_keywords = [[] for _ in range(expanded_nclass)]
    start = time.time()
    for i, category in enumerate(keywords, 1):
        for key in category:
            if key not in repeated_keys:
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


# #### Scratch (not important)

# <!-- 1. Dictionary -->
# <!-- 2. Stop words -->
# <!-- 3. Remove without city -->
# <!-- 4. Store city links -->
# 5. Count words related with different category

# In[11]:


[i for i in wv.similar_by_word('中国', 100) if '中国' not in i[0]]


# In[15]:


wv.similarity('拓荒者', '中国')


# In[ ]:




