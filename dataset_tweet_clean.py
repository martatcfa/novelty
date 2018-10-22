#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:33:08 2018

@author: marta

Limpando os dados do tweet
"""
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from os import listdir
from nltk.corpus import stopwords

from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

stopwords_en =set(stopwords.words('english'))
import re

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


def remove_stopwords(words, stem):
    pat=r'([^a-zA-Zs?":çáàãâéèêẽëíîóôûúü#.!,@_-…-)/)();+*$])' 
    
    stop_words = list(get_stop_words('en'))         #About 900 stopwords
    nltk_words = list(stopwords.words('english')) #About 150 stopwords
    stop_words.extend(nltk_words)
    #element = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', element, flags=re.MULTILINE)
    if stem==1:
        output = [porter_stemmer.stem(re.sub(pat, '', w, flags=re.MULTILINE)).replace("’", "").replace("‘","").replace("…","").replace("“","").replace("”","") for w in words if not w in stop_words]
    else:
        output = [(re.sub(pat, '', w, flags=re.MULTILINE)).replace("’", "").replace("‘","").replace("…","").replace("“","").replace("”","") for w in words if not w in stop_words]
    #output = [wordnet_lemmatizer.lemmatize(re.sub(pat, '', w, flags=re.MULTILINE)).replace("’", "") for w in words if not w in stop_words]
    #output = [re.sub(pat, '', w, flags=re.MULTILINE) for w in words if not w in stop_words]
    output = [w for w in output if len(w)>3]
    output_str = " ".join(str(x) for x in output)
    return output_str
    
fname='/home/marta/meu/projfinal/dataset_tweet_en/csv'
docLabels = []
docLabels = [f for f in sorted(listdir(fname)) if  f.endswith('.csv')]



for i,doc in enumerate(docLabels):
    df = pd.read_csv(fname +'/' + doc)
    
    if i==0:
        df_total = df
    else:        
        df_total=pd.concat([df_total,df],ignore_index=True)


#reading each line and cleaning
listatext_str=[]

df_total['text_normal'] = ''
for index, row in df_total.iterrows():
    df_total.iloc[index, 7] =  row['text']
    if not type(row['text']) is float:
        if len((row['text']).strip())>10:
            output_str= remove_stopwords(text_to_word_sequence((row['text']).strip()),1)    
            listatext_str.append(output_str)
            df_total.iloc[index, df.columns.get_loc('text')] = output_str
    else:
        print("empty text")
    


tokenizer = Tokenizer()
tokenizer.fit_on_texts(listatext_str)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


print('printing file')
fname='/home/marta/meu/projfinal/dataset_tweet_en'
df_total.to_csv(fname + '/clean_csv/tweets_clean.csv', encoding='utf-8', index=False)
