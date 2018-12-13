#!/usr/bin/env python3
#Labeling Code
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:07:52 2018

@author: marta
https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
https://nlpforhackers.io/wordnet-sentence-similarity/
https://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf
"""
import pandas as pd
from datetime import datetime
from datetime import timedelta
import re
import math
import networkx as nx
import pylab as plt
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import numpy as np


from collections import Counter
threshold_1=0.2
threshold_2=0.4
window_days=40
WORD = re.compile(r'\w+')
type_similirity_words=0
#Generate graph
G = nx.Graph()

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
    
 
def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0
 
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        
        val= [0 if synset.path_similarity(ss) is None else synset.path_similarity(ss) for ss in synsets2]
        
        if len(val)==0:
            best_score=0
        else:
            best_score = max(val)
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
 
    # Average the values
    if count==0:
        score=0
    else:    
        score /= count
        
        
    return score

    
def create_edged(list_node, node_central):
    for node,w in list_node:
        if (str(node_central)!= str(node) and w!=0):
            if G.has_edge(node_central,node):
                G[node_central][node]['weight'] = w
            else:
                G.add_edge(node_central, node, weight=w)
                
def generate_graph(df_graph,namefile,typefile):
   
    G.clear()
    plt.cla()
    plt.clf()
    plt.close()  
    node_color = []
    
    for index,row in df_graph.iterrows():         
        if row['audience']==1:
            c='pink'
        elif row['audience']==2:    
            c='orange'
        elif row['audience']==3:    
            c='yellow'
        elif row['audience']==4:    
            c='red'
        else:
            c='blue'
            
        G.add_node(row['id'])                   
        node_color.append(c)             
        similarity_list_present=row['similarity_list_present']        
        create_edged(similarity_list_present,row['id'])
        
        
      
    fig=plt.figure( figsize=(30, 30))   
    pos=nx.spring_layout(G, k=0.7, iterations=20)
        
    if len(nx.get_edge_attributes(G,'weight').items())!=0:
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        nx.draw(G,pos, with_labels = True, node_color=node_color,node_size=1000, edgelist=edges, edge_color=weights,width=5.0, edge_cmap=plt.cm.Reds)
    else:
        nx.draw(G,pos, with_labels = True, node_color=node_color,node_size=1000,width=5.0, edge_cmap=plt.cm.Reds)
            
           
    print('graph generating - period: ' + namefile)
    fig.savefig("graph/" + namefile + "_"+ typefile + " .png", format="PNG")
    plt.cla()
    plt.clf()
    plt.close()    
    
    G.clear()

def calcule_audience(quartile,df_weight_tweet,df_all):
    
    
    for index, row in df_weight_tweet.iterrows():   
        audience=1
        if row['weight']> quartile['weight'][4] and  row['weight']< quartile['weight'][5]:
            audience=2
        else:
            if row['weight']> quartile['weight'][5] and  row['weight']< quartile['weight'][6]:
                audience=3
            else:
             if  row['weight']>= quartile['weight'][6]:
                 audience=4
        df_all.loc[df_all['id'] == row['id'], 'audience'] = audience
    
       
def similarity_betwee_tweet_df(id_tweet,txt_tweet,df_present,favoritecount, retweetcount):
    List_similarity=[]
    List_fav=[]
    List_ret=[]
    weigh_total=0
    weigh=0
    weigh_total_mean=0
    
    for index, row in df_present.iterrows():    
        
        if type_similirity_words==1:
            vector1 = txt_tweet
            vector2 = row['text']
            val_similarity= sentence_similarity(vector1, vector2)
        else:
            vector1 = text_to_vector(txt_tweet)
            vector2 = text_to_vector(row['text'])
            val_similarity = get_cosine(vector1, vector2)
         


        
        #jaccard = get_jaccard_sim(str_tweet_actual,tweet[1])
        #mean_distance= (cosine+ jaccard)/2
        List_similarity.append(val_similarity)
        List_ret.append(row['retweetcount'])
        List_fav.append(row['favoritecount'])
        
        
        weigh=val_similarity+weigh
        
    #weigh_total= weigh_total* ((favoritecount/window_days) + retweetcount/window_days)  
    if len(List_similarity)!=0:
        weigh_total_mean=weigh/len(List_similarity)
        
    f=(favoritecount - np.min(List_fav))/ (np.max(List_fav) - np.min(List_fav))
    r=(retweetcount - np.min(List_ret))/ (np.max(List_ret) - np.min(List_ret))
    
    weigh_total= weigh_total_mean + (f + r)  
      
    return weigh_total,weigh_total_mean,f, r

        
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

       
def similarity_betwee_tweet(id_tweet_actual, str_tweet_actual, Lista_tweet):
    List_similarity=[]
    for tweet in Lista_tweet:
        
        if type_similirity_words==1:
            vector1 = str_tweet_actual
            vector2 = tweet[1]
            val_similarity= sentence_similarity(vector1, vector2)
        else:
            vector1 = text_to_vector(str_tweet_actual)
            vector2 = text_to_vector(tweet[1])
            val_similarity = get_cosine(vector1, vector2)
            
            
        #jaccard = get_jaccard_sim(str_tweet_actual,tweet[1])
        #mean_distance= (cosine+ jaccard)/2
        List_similarity.append([tweet[0], val_similarity])
        
    return List_similarity
    
def tag_rarity_tweet(List_present,List_past):   
    past_NP=0
    past_N=0
    past_NN=0
    
    for tweet in List_past:
        if threshold_1 < tweet[1] < threshold_2:
             past_NP=past_NP+1
             
        else:
            if tweet[1] >= threshold_2:
                past_NN=past_NN+1
             
            if tweet[1] <= threshold_1:
                past_N=past_N+1
                
                
                
    if (past_NN>0):
        tag="NN"
    else:
        if past_NP>past_N:
            tag= "NP"
        else:
            tag= "N"
    
    
    
    
    pre_NP=0
    pre_N=0
    pre_NN=0
    
    for tweet in List_present:
        if threshold_1 < tweet[1] < threshold_2:
             pre_NP=pre_NP+1
             
        else:
            if tweet[1] >= threshold_2:
                pre_NN=pre_NN+1
             
            if tweet[1] <= threshold_1:
                pre_N=pre_N+1
                
                
                
    if (pre_NN>0 or past_NN>0):
        tag="NN"
    else:
        if (pre_NP+ past_NP) > (pre_N+past_N):
            tag= "NP"
        else:
            tag= "N"

        
    return tag
    
fname='/home/marta/meu/projfinal/dataset_tweet_en/clean_csv'
df = pd.read_csv(fname + '/tweets_clean.csv')

#Labeling: Novel or Not Novel
df=df.sort_values(by=['created_at'],ascending=True )

count=0
list_present=[]
list_past=[]
df_all= pd.DataFrame( columns=['id', 'author','text','created_at','created_at_least','retweetcount','favoritecount','similarity_list_present','similarity_list_past','name_window','tag','audience','text_normal','weight_tweet','weigh_tweet_mean','weight_retweetcount','weight_favoritecount'])

 
print("Starting tweet labeling.")
for index, row in df.iterrows():    
    
   
    if (count==0):  
        date_tweet_ini=datetime.strptime(row['created_at'], "%Y-%m-%d %H:%M:%S")
        date_tweet_end = date_tweet_ini + timedelta(days=window_days)
        print('period: ' + str(date_tweet_ini) + " - " + str(date_tweet_end)  )
        count=1
        
        
    
    
    
    d1=datetime.strptime(row['created_at'], "%Y-%m-%d %H:%M:%S")
    if not (date_tweet_ini <= d1 <= date_tweet_end):        
              
        date_tweet_ini=datetime.strptime(row['created_at'], "%Y-%m-%d %H:%M:%S")
        date_tweet_end = date_tweet_ini + timedelta(days=window_days)
        print('period: ' + str(date_tweet_ini) + " - " + str(date_tweet_end)  )
        
        list_past=list_present
        list_present=[]
    
    namewindow=str(date_tweet_ini.year) + str(date_tweet_ini.month).zfill(2)  + str(date_tweet_ini.day).zfill(2)  + "_" + str(date_tweet_end.year) + str(date_tweet_end.month).zfill(2)  + str(date_tweet_end.day).zfill(2)  
    
     
    
    text="" if type(row['text'])==float else row['text']
    
    similarity_list_present=similarity_betwee_tweet(row['id'],text,list_present)
    similarity_list_past=similarity_betwee_tweet(row['id'],text,list_past)
    list_present.append( [row['id'],text,row['created_at'],row['retweetcount'],row['favoritecount'],similarity_list_present,similarity_list_past,tag_rarity_tweet(similarity_list_present,similarity_list_past),similarity_list_present,similarity_list_past])
    tag=tag_rarity_tweet(similarity_list_present,similarity_list_past)
    
    
     
    print('labeling:' + str(row['id']) + ' - tag:' + tag)      
    newrow=[row['id'],row['author'],text,row['created_at'],datetime.strptime(row['created_at'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d"),row['retweetcount'],row['favoritecount'],similarity_list_present,similarity_list_past,namewindow,tag,0, row['text_normal'],0,0,0,0]
    df_all = df_all.append(pd.DataFrame([newrow],index=[index],columns=df_all.columns))
   


#Audience: 1 a 4
print("Starting calculate audience.")
df_all=df_all.sort_values(by=['created_at'],ascending=True )
df_filter=df_all.loc[df_all['tag'].isin(['N','NP'])].sort_values(by=['created_at'],ascending=True )
count=0


weight_tweet=0
df_weight_tweet= pd.DataFrame( columns=['id', 'weight','audience'])

 
for index,row in df_filter.iterrows():        
     
    if (count==0):  
        date_tweet_ini=  datetime.strptime(row['created_at_least'], "%Y-%m-%d")
        date_tweet_end = date_tweet_ini + timedelta(days=window_days)        
        
        # Select the rows between two dates       
        df_date=df_all[(df_all['created_at_least'] > str(date_tweet_ini)) & (df_all['created_at_least'] <= str(date_tweet_end))]
        print('period: ' + str(date_tweet_ini) + " - " + str(date_tweet_ini)  )
        count=1
    
    d1=datetime.strptime(row['created_at_least'], "%Y-%m-%d")

    if not (date_tweet_ini <= d1 <= date_tweet_end):   
        
       
        
        quartile=df_weight_tweet.describe()
        
        calcule_audience(quartile,df_weight_tweet,df_all)
        
        namefile=str(date_tweet_ini.year) + str(date_tweet_ini.month).zfill(2)  + str(date_tweet_ini.day).zfill(2)  + "_" + str(date_tweet_end.year) + str(date_tweet_end.month).zfill(2)  + str(date_tweet_end.day).zfill(2)  
        df_graph=df_all[(df_all['created_at_least'] > str(date_tweet_ini)) & (df_all['created_at_least'] <= str(date_tweet_end))]
        #generate_graph(df_graph,namefile,"present")

        
        df_graph=df_all[(df_all['created_at_least'] > str(date_tweet_ini - timedelta(days=window_days+1)  )) & (df_all['created_at_least'] <= str(date_tweet_end ))]
        #generate_graph(df_graph,namefile,"past")
        
        
        
        
        date_tweet_ini=  datetime.strptime(row['created_at_least'], "%Y-%m-%d")
        date_tweet_end = date_tweet_ini + timedelta(days=window_days)        
     
        
        df_weight_tweet=df_weight_tweet.iloc[0:0]
        weight_tweet=0
        print('period: ' + str(date_tweet_ini) + " - " + str(date_tweet_end)  )
        
         # Select the rows between two dates       
        df_date=df_all[(df_all['created_at_least'] > str(date_tweet_ini)) & (df_all['created_at_least'] <= str(date_tweet_end))]
        
    
    
    weight_tweet,weigh_tweet_mean,weight_retweetcount,weight_favoritecount=similarity_betwee_tweet_df(row['id'],row['text'],df_date,row['favoritecount'],row['retweetcount'])
    
    df_all.loc[df_all['id'] == row['id'], 'weight_tweet'] = weight_tweet
    df_all.loc[df_all['id'] == row['id'], 'weigh_tweet_mean'] = weigh_tweet_mean
    df_all.loc[df_all['id'] == row['id'], 'weight_retweetcount'] = weight_retweetcount
    df_all.loc[df_all['id'] == row['id'], 'weight_favoritecount'] = weight_favoritecount
     
   
    
    
    newrow=[row['id'], weight_tweet,0]
    
    df_weight_tweet= df_weight_tweet.append(pd.DataFrame([newrow],index=[index],columns=df_weight_tweet.columns))
    




quartile=df_weight_tweet.describe()        
df_weight_tweet=calcule_audience(quartile,df_weight_tweet,df_all)
namefile=str(date_tweet_ini.year) + str(date_tweet_ini.month).zfill(2)  + str(date_tweet_ini.day).zfill(2)  + "_" + str(date_tweet_end.year) + str(date_tweet_end.month).zfill(2)  + str(date_tweet_end.day).zfill(2)  
df_graph=df_all[(df_all['created_at_least'] > str(date_tweet_ini)) & (df_all['created_at_least'] <= str(date_tweet_end))]
#generate_graph(df_graph,namefile,"present")


  
df_graph=df_all[(df_all['created_at_least'] > str(date_tweet_ini - timedelta(days=window_days+1)  )) & (df_all['created_at_least'] <= str(date_tweet_end ))]
#generate_graph(df_graph,namefile,"past")
       



    

    
    
#https://nlpforhackers.io/wordnet-sentence-similarity/
#https://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf

print('printing file labeling')
fname='/home/marta/marta/projfinal/dataset_tweet_en'
df_all.to_csv(fname + '/clean_csv/tweets_labeling_en_w' + str(window_days) + '.csv', encoding='utf-8', index=False)

print('Finished processing')
    
        
        








