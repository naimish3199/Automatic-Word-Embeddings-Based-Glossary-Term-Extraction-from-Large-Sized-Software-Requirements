# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 22:24:13 2022

@author: naimi
"""
import bs4
import requests
import urllib
import re
from nltk.tokenize import word_tokenize
import pandas as pd

df = pd.read_csv("C:/Users/naimi/OneDrive/Desktop/ECS412Project/homeautomation.csv")

x = "" # for storing wikipedia corpus

for i in df['title']:
    url = "https://en.wikipedia.org/wiki/" + i
    response = requests.get(url)
    page = response.text
    soup = bs4.BeautifulSoup(page, 'lxml')
    #texts = ""
    file = open('web.txt','a',encoding='utf-8')
    with file as f:
        for para in soup.find_all('p'):
            f.write(para.text)        
    f.write("END")    
 


