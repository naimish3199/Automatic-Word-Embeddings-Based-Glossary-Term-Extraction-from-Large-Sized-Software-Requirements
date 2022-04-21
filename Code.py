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
# =============================================================================
cre100 = C_CRE[0:100]
# =============================================================================
    
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
# =============================================================================
cre100prep = [datapreprocess(v) for v in cre100]
# =============================================================================

cre_tokens = []
for m in Cdash_CRE:
    for l in m:
        cre_tokens.append(l)

#         
# =============================================================================
cre100_tokens = []
for m in cre100prep:
    for l in m:
        cre100_tokens.append(l)        
# 
# =============================================================================

cha_tokens = []
for m in Cdash_HA:
    for l in m:
        cha_tokens.append(l)        
        
def chunker(token):
    # text chunking
    grammar = ('''
               NP: {<DT>?<JJ>*<NN>}
               ''')               
    a = nltk.pos_tag(token)
    chunkParser = nltk.RegexpParser(grammar)
    tree = chunkParser.parse(a)
    g = []
    for subtree in tree.subtrees():
        g.append(subtree) 
    G = set()
    for i in g:
        m = ""
        for j in i:
            m = m + " " + str(j[0])
        G.add(m)   
    G = list(G)    
    G = [txt.strip() for txt in G] 

    nn = ['NN','NNS','NNP','NNPS']
    GT2 = []
    for m in range(len(a)-1):
        if a[m][1] in nn and a[m+1][1] in nn:
            GT2.append(a[m][0]+" "+a[m+1][0])
        

    q = []    
    for m in range(len(a)-2):
        if a[m][1] in nn and a[m+1][1] in nn and a[m+2][1] in nn:
            q.append(a[m][0]+" "+a[m+1][0]+" "+a[m+2][0])  
        
    gtnnjj = []    
    for m in range(len(a)-2):
        if a[m][1] == 'JJ' and a[m+1][1] in nn and a[m+2][1] in nn:
            gtnnjj.append(a[m][0]+" "+a[m+1][0]+" "+a[m+2][0])
    
    result = G + GT2 + q + gtnnjj
    return result        
            
GT = chunker(cre_tokens)
TW = chunker(cha_tokens)
C100 = chunker(cre100_tokens)


CGT = [value for value in GT if value in TW]
CGT = sorted(set(CGT))
# =============================================================================
CGT100 = [value for value in C100 if value in TW]
CGT100 = sorted(set(CGT100))
# 
# =============================================================================

cgtsplit = [x.split() for x in CGT]
Cdoubledash_CRE = Cdash_CRE.copy()

# =============================================================================
cgtsplit100 = [x.split() for x in CGT100]
Cdouble100 = cre100prep.copy()
# =============================================================================
for i in range(len(CGT100)):
    b = CGT100[i].split()
    length = len(b)
    for k in Cdouble100:
        l = len(k)-length+1
        for j in range(l):
            d = k[j:j+length]
            onext = k[j:j+length+1]
            tnext = k[j:j+length+2]
            olast = k[j-1:j+1]
            tlast = k[j-2:j+1]
            if d == b:
                if j < l-2:
                    if onext not in cgtsplit100 and tnext not in cgtsplit100 and olast not in cgtsplit100 and tlast not in cgtsplit100: 
                        k[j] = "$"+CGT100[i].replace(" ", "_")+"$"
                        for w in range(j+1,j+length):
                            k[w] = ""
                        j = j + length - 1    
                elif j == l-2:
                    if onext not in cgtsplit100 and olast not in cgtsplit100 and tlast not in cgtsplit100: 
                        k[j] = "$"+CGT100[i].replace(" ", "_")+"$"
                        for w in range(j+1,j+length):
                            k[w] = ""  
                        j = j + length - 1      
                else:
                    if olast not in cgtsplit100 and tlast not in cgtsplit100:
                        k[j] = "$"+CGT100[i].replace(" ", "_")+"$"
                        for w in range(j+1,j+length):
                            k[w] = ""  
                        j = j + length - 1      
                        
                        
# =============================================================================
# for k in Cdoubledash_CRE:
#    ch = '_'
#    for j in range(len(k)):
#         if k[j].find(ch) != -1:
#             k[j] = "$"+k[j]+"$"
# =============================================================================
            
#Cdoubledash_CRE = [i for i in Cdoubledash_CRE if i != ""]  

Cdoubledash_HA = Cdash_HA.copy()
for i in range(len(CGT)):
    b = CGT[i].split()
    length = len(b)
    for k in Cdoubledash_HA:
        l = len(k)-length+1
        for j in range(l):
            d = k[j:j+length]
            onext = k[j:j+length+1]
            tnext = k[j:j+length+2]
            olast = k[j-1:j+1]
            tlast = k[j-2:j+1]
            if d == b:
                if j < l-2:
                    if onext not in cgtsplit and tnext not in cgtsplit and olast not in cgtsplit and tlast not in cgtsplit: 
                        k[j] = CGT[i].replace(" ", "_")
                        for w in range(j+1,j+length):
                            k[w] = ""
                        j = j + length - 1  
                elif j == l-2:
                    if onext not in cgtsplit and olast not in cgtsplit and tlast not in cgtsplit: 
                        k[j] = CGT[i].replace(" ", "_")
                        for w in range(j+1,j+length):
                            k[w] = ""  
                        j = j + length - 1      
                else:
                    if olast not in cgtsplit and tlast not in cgtsplit:
                        k[j] = CGT[i].replace(" ", "_")
                        for w in range(j+1,j+length):
                            k[w] = ""       
                        j = j + length - 1      
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
mskip = Word2Vec(corpora, min_count = 1, 
                              vector_size = 100, window = 10,sg = 1)
print(mskip)
#Create CBOW model
mcbow = Word2Vec(corpora, min_count = 1, 
                              vector_size = 100, window = 10,sg = 0)
print(mcbow)



sortedCGT = sorted(CGT)
CRE_context = ["$"+i.replace(" ", "_")+"$" for i in sortedCGT]
CHA_context = [i.replace(" ", "_") for i in sortedCGT]
fCRE_context = ["$"+i.replace(" ", "_")+"$" for i in CGT100]
fCHA_context = [i.replace(" ", "_") for i in CGT100]
cbow_similarity = []
skipgram_similarity = []
sresult = []
cresult = []
sc = []
ss= []
f100 = []
term = []
for i in range(len(CGT100)):
    if (fCRE_context[i] in mskip.wv) and (fCHA_context[i] in mskip.wv) :
        ss.append(mskip.wv.similarity(w1 = fCRE_context[i] , w2 =fCHA_context[i]))
# =============================================================================
        if mskip.wv.similarity(w1 = fCRE_context[i] , w2 =fCHA_context[i]) >= 0.6:
            a = "print(model.wv.similarity(w1 = {} , w2 = {})) = {}".format(fCRE_context[i],fCHA_context[i],mskip.wv.similarity(w1 = fCRE_context[i] , w2 =fCHA_context[i]))
            f100.append(a)
            term.append(fCHA_context[i].replace('_',' '))
# =============================================================================
            
for i in range(len(sortedCGT)):
    if (CRE_context[i] in mcbow.wv) and (CHA_context[i] in mcbow.wv) :
        cbow_similarity.append(mcbow.wv.similarity(w1 = CRE_context[i] , w2 =CHA_context[i]))
# =============================================================================
#         if mcbow.wv.similarity(w1 = CRE_context[i] , w2 =CHA_context[i]) >= 0.6:
#             a = "print(model.wv.similarity(w1 = {} , w2 = {})) = {}".format(CRE_context[i],CHA_context[i],mcbow.wv.similarity(w1 = CRE_context[i] , w2 =CHA_context[i]))
#             cresult.append(a) 
# =============================================================================
            
 
#r = sorted(cbow_similarity, reverse=True)
#print(len([t for t in r if t>0.6]))
file1 = open('C:/Users/naimi/OneDrive/Desktop/final/skipgram_results.txt','a')
for u in sresult:
    file1.write(u)
    file1.write("\n")
file1.close()



file2 = open('C:/Users/naimi/OneDrive/Desktop/final/cbow_results.txt','a')
for u in cresult:
    file2.write(u)
    file2.write("\n")
file2.close()
# =============================================================================
# p = 1
# for u in C_CRE[0:100]:
#     file.write("R"+"{} ".format(p))
#     file.write(u)
#     file.write("\n")
#     p = p+1
# file.close()
# =============================================================================

f = open('C:/Users/naimi/OneDrive/Desktop/final/first100.txt','a')
for u in f100:
    f.write(u)
    f.write("\n")
f.close()

f1= open('C:/Users/naimi/OneDrive/Desktop/final/first100terms.txt','a')
for u in term:
    f1.write(u)
    f1.write("\n")
f1.close()













    
    
