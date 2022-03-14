#Importing Libraries
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import re
from nltk.stem import WordNetLemmatizer # Lemmatization

#Loading dataset CRE
data = pd.read_csv("C:/Users/naimi/OneDrive/Desktop/ECS412Project/crowdre_cleaned-csv/requirements.csv")
data['CRE'] = data['feature'] + ', ' + data['benefit'] + '.'
C_CRE = []
for i in data['CRE']:
    C_CRE.append(i)
C_CRE = [x for x in C_CRE]   
    
#Loading dataset CHA
C_HA =""
with open("C:/Users/naimi/OneDrive/Desktop/ECS412Project/web.txt", 'r', encoding='utf-8') as f:
   C_HA = f.read()



C_HA =  C_HA.split("END")


# NLP Pipeline for data preprocessing
stop_words = stopwords.words('english') #Stopwords
lemmatizer = WordNetLemmatizer() 
def datapreprocess(corpus):
    corpus = re.sub(r'\[[0-9]*\]',' ',corpus)  #remove like [8]
    corpus = re.sub(r'\d',' ',corpus) #remove digits
    corpus = re.sub(r"[^a-zA-Z0-9]",' ',corpus) #removal of alpha numeric characters
    corpus = re.sub(r'\b\w{1}\b', ' ', corpus)
    corpus = re.sub(r'\s+',' ',corpus)
    corpus = word_tokenize(corpus) #word tokenization
    corpus = [word.lower() for word in corpus] #lowering of tokens
    corpus = [word for word in corpus if word not in stop_words] #removal of stopwords
    corpus = [lemmatizer.lemmatize(word) for word in corpus] #lemmatizing the words
    
    return corpus

# Passing data through nlp pipeline
Cdash_CRE = [datapreprocess(x) for x in C_CRE]
Cdash_HA = [datapreprocess(y) for y in C_HA]

cre_tokens = []
for m in Cdash_CRE:
    for l in m:
        cre_tokens.append(l)

cha_tokens = []
for m in Cdash_HA:
    for l in m:
        cha_tokens.append(l)        
        
        
        
# text chunking
grammar = ('''
           NP: {<DT>?<JJ>*<NN>}
           ''')    
a = nltk.pos_tag(cre_tokens)
chunkParser = nltk.RegexpParser(grammar)
tree = chunkParser.parse(a)
gt = []
for subtree in tree.subtrees():
    gt.append(subtree) 
    
    
b = nltk.pos_tag(cha_tokens)
tree2 = chunkParser.parse(b)
tw = []
for subtree in tree2.subtrees():
    tw.append(subtree) 



GT = set()
for i in gt:
    m = ""
    for j in i:
        m = m + " " + str(j[0])
    GT.add(m)   
GT = list(GT)    
GT = [txt.strip() for txt in GT] 

TW = set()
for i in tw:
    m = ""
    for j in i:
        m = m + " " + str(j[0])
    TW.add(m)   
TW = list(TW)    
TW = [txt.strip() for txt in TW]   

CGT = [value for value in GT if value in TW]


Cdoubledash_CRE = Cdash_CRE.copy()
for i in CGT:
    b = i.split()
    length = len(b)
    for k in Cdoubledash_CRE:
        for j in range(len(k)-length+1):
            d = k[j:j+length]
            if d == b:
                k[j] = "$"+i.replace(" ", "_")+"$"
                for w in range(j+1,j+length):
                    k[w] = ""        

#Cdoubledash_CRE = [i for i in Cdoubledash_CRE if i != ""]  

Cdoubledash_HA = Cdash_HA.copy()
for i in CGT:
    b = i.split()
    length = len(b)
    for k in Cdoubledash_HA:
        for j in range(len(k)-length+1):
            d = k[j:j+length]
            if d == b:
                k[j] = i.replace(" ", "_")
                for w in range(j+1,j+length):
                    k[w] = ""      
                
#Cdash_HA = [i for i in Cdash_HA if i != ""] 

corpora =[]
for i in Cdoubledash_CRE:
    corpora.append(i)    
for k in Cdoubledash_HA:
    corpora.append(k)    

import gensim
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# Create Skipgram model
model = Word2Vec(corpora, min_count = 1, 
                              vector_size = 100, window = 10,sg = 1)
print(model)
# Create CBOW model
model1 = Word2Vec(corpora, min_count = 1, 
                              vector_size = 100, window = 10,sg = 0)
print(model)

sortedCGT = sorted(CGT)
CRE_context = ["$"+i.replace(" ", "_")+"$" for i in sortedCGT]
CHA_context = [j.replace(" ", "_") for j in sortedCGT]
cbow_similarity = []
result = []
sc = []
for i in range(len(sortedCGT)):
    if (CRE_context[i] in model1.wv) and (CHA_context[i] in model1.wv) :
        cbow_similarity.append(model1.wv.similarity(w1 = CRE_context[i] , w2 =CHA_context[i]))
r = sorted(cbow_similarity, reverse=True)
print(len([t for t in r if t>0.6]))
#        if model.wv.similarity(w1 = CRE_context[i] , w2 =CHA_context[i]) > 0.85:
 #           a = "print(model1.wv.similarity(w1 = {} , w2 = {})) = {}".format(CRE_context[i],CHA_context[i],model1.wv.similarity(w1 = CRE_context[i] , w2 =CHA_context[i]))
  #          result.append(a)

file = open('C:/Users/naimi/OneDrive/Desktop/ECS412Project/cbow.txt','a')
p = len(result)
for u in result:
    file.write(u)
    file.write("\n")
file.close()



file = open('C:/Users/naimi/OneDrive/Desktop/ECS412Project/new.txt','a')
p = 1
for u in C_CRE[0:100]:
    file.write("R"+"{} ".format(p))
    file.write(u)
    file.write("\n")
    p = p+1
file.close()
















    
    
