#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:18:39 2018

@author: marta
"""
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
import os
import imageio
from PIL import Image
from scipy.misc import imresize
import pandas as pd
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Dense,Flatten,BatchNormalization
from keras import regularizers
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer 
from nltk.corpus import stopwords
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import codecs
from keras.layers import Embedding, Conv1D, MaxPool1D,  Dropout
from keras import optimizers

#from sklearn.utils import class_weight


#from pyod.models.knn import KNN
#from pyod.models.lof import LOF
#from pyod.models.mcd import MCD
#from pyod.models.iforest import IForest
#from pyod.models.abod import ABOD
#from pyod.models.combination import aom, moa, average, maximization
#from pyod.utils.utility import standardizer
#from pyod.models.feature_bagging import FeatureBagging
   

from pyod.models.auto_encoder import AutoEncoder
#from pyod.models.mcd import MCD

#from pyod.models.hbos import HBOS

sns.set_style("whitegrid")
np.random.seed(0)
encoding_dim = 150
inChannel = 3
h, w = 90, 90
input_img = Input(shape = (h, w, inChannel))
fname='/home/marta/meu/projfinal/dataset_tweet_en/clean_csv'
fname_result='/home/marta/meu/projfinal/dataset_tweet_en/result'
fname_img='/home/marta/meu/projfinal/dataset_tweet_en/jpg'
weight_decay = 1e-4
MAX_NB_WORDS = 100000

K_neighbor=2
#n_bins=100

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

embeddings_index = {}

def normalizeclass(y):
        
    ########################### ENCODING CLASSES ################################
    ### Change any labels to sequential integer labels
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    Yn = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    Yi = np_utils.to_categorical(Yn)
    uniques, ids = np.unique(Yn, return_inverse=True)
  
    ########################### NORMALIZATION ####################################
    return Yi,Yn



def readimage(list_item):
   
    
    j=0
    n_imgs = len(list_item) # number of images
    imgs = np.zeros((n_imgs,h,w,inChannel), dtype=np.float32)
    
      
                
    for i, val in enumerate(list_item):  
        if os.path.isfile(fname_img + '/' + val.lower() + '.jpg'):
            val_final=fname_img + '/' + val.lower() + '.jpg'
        else:
            if os.path.isfile(fname_img + '/' + val.lower() + '.png'):
                val_final=fname_img + '/' + val.lower() + '.png'
            else:
                
                print('*******ERROR: FILE NOT EXISTS*******')
                            
        print('read files:' + val_final)   
        img = imageio.imread(val_final)
                
        if len(img.shape)==2:
             img = Image.open(val_final).convert('RGB')
             
        if img.shape[2]==4:            
              img = Image.open(val_final).convert('RGB')
            
            
        img = imresize(img,(h,w))    
                
        imgs[j, ...] = img
        j=j+1
         
            
         
         
        
    return imgs


def enconding_img(lst_img,Yn):
    

    #activity_regularizer=regularizers.l1(10e-5)
    
    x = Conv2D(16, (2, 2), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x )
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dense(25, activation='relu', kernel_initializer='lecun_normal')(x)
    x = Flatten()(x) 
    
    encoded = Dense(encoding_dim, activation='relu')(x)
    encoder = Model(input_img, encoded)
    
    
    x = Dense(encoding_dim, activation='relu', kernel_initializer='lecun_normal')(encoded)
    x = BatchNormalization()(x)
    x = Dense(50, activation='relu',kernel_initializer='lecun_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(20, activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    decoded = Dense(1, activation='sigmoid')(x)
    
    
    
    autoencoder = Model(input_img, decoded)   

    #autoencoder.summary()
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
    
    X = readimage(lst_img)
    
    
  
       
    # reshape, normalization and type conversation
    X = np.reshape(X, (len(X), h, w,inChannel))  # adapt this if using `channels_first` image data format
    X =  X.astype('float32') / np.max(X)    
    X = X.astype('float32')
    
     #training params
    batch_size = int(len(lst_img)/8) 
    if batch_size<1:
        batch_size = 1
    

    
    num_epochs = int(len(lst_img)/4)-1     
    if num_epochs<1: 
        num_epochs = 1
    
        
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(Yn),  Yn)
    
    autoencoder.fit(X, Yn, epochs=num_epochs, batch_size=batch_size,verbose=1,validation_split=0.1,class_weight = 'auto')

    encoded_imgs_train = encoder.predict(X)

    return encoded_imgs_train


def enconding_img1(lst_img,Yn):
    

    #activity_regularizer=regularizers.l1(10e-5)
    
    x = Conv2D(16, (2, 2), activation='relu', padding='same')(input_img)
    x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x )
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x )
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Dense(25, activation='relu', kernel_initializer='lecun_normal')(x)
    x = Flatten()(x) 
    
    encoded = Dense(encoding_dim, activation='relu')(x)
    encoder = Model(input_img, encoded)
    
    
    x = Dense(encoding_dim, activation='relu', kernel_initializer='lecun_normal')(encoded)
    x = Dense(encoding_dim, activation='relu', kernel_initializer='lecun_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(50, activation='relu',kernel_initializer='lecun_normal')(x)
    x = Dense(50, activation='relu',kernel_initializer='lecun_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(20, activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    decoded = Dense(1, activation='sigmoid')(x)
    
    
    
    autoencoder = Model(input_img, decoded)   

    #autoencoder.summary()
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
    
    X = readimage(lst_img)
    
    
  
       
    # reshape, normalization and type conversation
    X = np.reshape(X, (len(X), h, w,inChannel))  # adapt this if using `channels_first` image data format
    X =  X.astype('float32') / np.max(X)    
    X = X.astype('float32')
    
     #training params
    batch_size = int(len(lst_img)/8) 
    if batch_size<1:
        batch_size = 1
    

    
    num_epochs = int(len(lst_img)/4)-1     
    #num_epochs = int(len(lst_img)/4)
    if num_epochs<1: 
        num_epochs = 1
    
        
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(Yn),  Yn)
    
    autoencoder.fit(X, Yn, epochs=num_epochs, batch_size=batch_size,verbose=1,validation_split=0.1,class_weight = 'auto')

    encoded_imgs_train = encoder.predict(X)

    return encoded_imgs_train



def enconding_txt(lst_txt,Yn):
    tokenizer = RegexpTokenizer(r'\w+')
    
    #load data
    train_df = pd.DataFrame(lst_txt,columns=['text'])    
    train_df = train_df.fillna('_NA_')
    
    
    
    #visualize word distribution
    train_df['doc_len'] = train_df['text'].apply(lambda words: len(words.split(" ")))
    mean_doc=train_df['doc_len'].mean()
    max_seq_len = np.round(mean_doc + train_df['doc_len'].std()).astype(int)
    
    #word quatity per doc
    sns.distplot(train_df['doc_len'], hist=True, kde=True, color='b', label='doc len')
    plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
    plt.title('comment length'); plt.legend()
    plt.show()
    
    
    
    raw_docs_train = train_df['text'].tolist()
    
        
    print("Pre-processing train data...")
    processed_docs_train = []
    for doc in tqdm(raw_docs_train):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs_train.append(" ".join(filtered))
    
    
    print("Tokenizing input data...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    tokenizer.fit_on_texts(processed_docs_train )  #leaky
    word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)    
    word_index = tokenizer.word_index
    print("Dictionary size: ", len(word_index))
    
    #pad sequences
    word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
    
    
    #training params
     #training params
    batch_size = int(len(lst_txt)/8) 
    if batch_size<1:
        batch_size = 1
    

    
    num_epochs = int(len(lst_txt)/2)+1     
    if num_epochs<1: 
        num_epochs = 2
        
    #batch_size = 2 
    #num_epochs = 10
    
    #model parameters
    num_filters = 64 
    embed_dim = 300 
    weight_decay = 1e-4
    
    
    #embedding matrix
    print('Preparing embedding matrix...')
    words_not_found = []
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('Number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print("Sample words not found: ", np.random.choice(words_not_found, 10))
    #number of null word embeddings: 3421
    
    
    print('Building model...')
    
    
    input_txt = Input(shape = (max_seq_len,))
    x = Embedding(nb_words,embed_dim,weights=[embedding_matrix], input_length=max_seq_len, trainable=False)(input_txt)
    x = Conv1D(num_filters, 7, activation='relu', padding='same')(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(num_filters, 7, activation='relu', padding='same') (x)
    x = MaxPool1D()(x)
    
    x = Flatten()(x) 
    
    
    encoded = Dense(encoding_dim, activation='relu')(x)
    encoder = Model(input_txt, encoded)
    
    
    x = Dense(encoding_dim, activation='relu', kernel_initializer='random_uniform')(encoded)
    x = Dense(50, activation='relu', kernel_initializer='random_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    #x = Dense(32, activation='relu')(x)
    decoded = Dense(1, activation='sigmoid') (x)
    
    
    
    autoencoder = Model(input_txt, decoded)
    
    #autoencoder.summary()
    
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(Yn),  Yn)
    
    autoencoder.fit(word_seq_train, Yn,batch_size=batch_size,epochs=num_epochs, verbose=1,validation_split=0.1,class_weight = 'auto')
    encoded_txt = encoder.predict(word_seq_train) 
    return encoded_txt,mean_doc,max_seq_len

def readGoogle():    
    
     #load embeddings GOOGLE
    print('Loading Google word embeddings...')
    
    f = codecs.open('fasttext/wiki-news-300d-1M.vec', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors Google' % len(embeddings_index))


def fusionTxt_Img( y_test_pred_txt,y_prob_txt,y_test_pred_img,y_prob_img):
    

    bol_nov='1'
    type_fusion='txt'
    if (y_test_pred_txt[0]==1 and y_test_pred_img[0]==0): #1 = Novidade / 0 = Não Novidade
        prob_txt=y_prob_txt[0][1]
        prob_img=y_prob_img[0][0]
        
        probN=y_prob_txt[0][1]
        probNN=y_prob_txt[0][0]
        
        if prob_txt< prob_img:
            
            bol_nov='0'
            probN=y_prob_img[0][1]
            probNN=y_prob_img[0][0]
            type_fusion='img'
            
    else:
        if (y_test_pred_txt[0]==0 and y_test_pred_img[0]==1):
            prob_txt=y_prob_txt[0][0]
            prob_img=y_prob_img[0][1]
            
            probN=y_prob_img[0][1]
            probNN=y_prob_img[0][0]
            type_fusion='img'
            
            if  (prob_img < prob_txt):
                bol_nov='0'    
                probN=y_prob_txt[0][1]
                probNN=y_prob_txt[0][0]
                type_fusion='txt'
        
        else:
            if (y_test_pred_txt[0]==0 and y_test_pred_img[0]==0):
                if np.max([y_prob_txt[0][0],y_prob_img[0][0]])== y_prob_img[0][0]:                    
                    probN=y_prob_img[0][1]
                    probNN=y_prob_img[0][0]
                    type_fusion='img'
                else:
                    probN=y_prob_txt[0][1]
                    probNN=y_prob_txt[0][0]
                    type_fusion='txt'
            else:    
                if np.max([y_prob_txt[0][1],y_prob_img[0][1]])== y_prob_img[0][1]:                    
                    probN=y_prob_img[0][1]
                    probNN=y_prob_img[0][0]
                    type_fusion='img'
                else:
                    probN=y_prob_txt[0][1]
                    probNN=y_prob_txt[0][0]
                    type_fusion='txt'
                    
            bol_nov=y_test_pred_txt[0]
    
    return bol_nov,probN,probNN,type_fusion




def algorithmFusion(X_txt,Y,X_img, df_filter_pre,param):
    Y_txt=[1 if x==0 else 0 for x in Y]
    Y_img=[1 if x==0 else 0 for x in Y]
    
    X_train_txt=X_txt[0:len(X_txt) -len(df_filter_pre),:]
    Y_train_txt=Y_txt[0:len(Y_txt) -len(df_filter_pre)]
    X_test_txt=X_txt[len(X_txt) -len(df_filter_pre):len(X_txt),:]
    Y_test_txt=Y_txt[len(Y_txt) -len(df_filter_pre):len(Y_txt)]
    
    
    
    X_train_img=X_img[0:len(X_img) -len(df_filter_pre),:]
    Y_train_img=Y_img[0:len(Y_img) -len(df_filter_pre)]
    X_test_img=X_img[len(X_img) -len(df_filter_pre):len(X_img),:]
    Y_test_img=Y_img[len(Y_img) -len(df_filter_pre):len(Y_img)]
    
    list_roc=[]
    list_precision=[]
    
    rows=[]
    for i, row in enumerate(df_filter_pre.values):    
         # train txt kNN detector
        #clf_txt = KNN(n_neighbors=1)
        #clf_txt = IForest(n_estimators=param)
         
        #clf_txt = ABOD(contamination=0.1,n_neighbors=param,method='fast')
        clf_txt = AutoEncoder(epochs=param,verbose=0)    
        #clf_txt = FeatureBagging(n_estimators=param)
        
        #clf_txt = MCD(contamination=0.1)
        #clf_txt = HBOS(n_bins=param)
        
        clf_txt.fit(X_train_txt)
        
         # train img kNN detector
        #clf_img = KNN(n_neighbors=4)
        #clf_img = IForest(n_estimators=param)
        #clf_img = ABOD(contamination=0.1,n_neighbors=param,method='fast')
        clf_img = AutoEncoder(epochs=param,verbose=0)
        #clf_img = FeatureBagging(n_estimators=param)
        
        
        
        #clf_img = MCD(contamination=0.1)
        #clf_img = HBOS(n_bins=2)
        
        clf_img.fit(X_train_img)
        
        
        y_test_pred_txt = clf_txt.predict(X_test_txt[i].reshape(1,-1))  # outlier labels (0 or 1)
        y_prob_txt = clf_txt.predict_proba(X_test_txt[i].reshape(1,-1), method='unify')  
        
        
        y_test_pred_img = clf_img.predict(X_test_img[i].reshape(1,-1))  # outlier labels (0 or 1)
        y_prob_img = clf_img.predict_proba(X_test_img[i].reshape(1,-1), method='unify')  
        
        bol_nov,probN,probNN,type_fusion=fusionTxt_Img(y_test_pred_txt,y_prob_txt,y_test_pred_img,y_prob_img)
    
    
    
        from pyod.utils.utility import standardizer
        from pyod.utils.data import evaluate_print
        
        train_scores=np.zeros([len(clf_txt.decision_scores_), 2])
        test_scores=np.zeros([len(clf_txt.decision_function(X_test_txt[i].reshape(1,-1))), 2])
        
        train_scores[:, 0] = clf_txt.decision_scores_
        test_scores[:, 0] = clf_txt.decision_function(X_test_txt[i].reshape(1,-1))
        
        
        train_scores[:, 1] = clf_img.decision_scores_
        test_scores[:, 1] = clf_img.decision_function(X_test_txt[i].reshape(1,-1))
        
        # scores have to be normalized before combination
        train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)
        
        
        
       # y_by_average = average(test_scores_norm)
        
        #evaluate_print('Combination by AOM txt', y_test_pred_txt, y_by_average)
        #evaluate_print('Combination by AOM img', y_test_pred_img, y_by_average)
            
       # y_by_maximization = maximization(test_scores_norm)
        
        #y_by_aom = aom(test_scores_norm) # 5 groups
        #y_by_moa = moa(test_scores_norm, 1) # 5 groups
        
        
        

    
        cont_txt=len(X_train_txt)
        X_train_txt=np.append(X_train_txt,X_test_txt[0])
        Y_train_txt=np.append(Y_train_txt,Y_test_txt[0])
        X_train_txt = X_train_txt.reshape(cont_txt +1,encoding_dim)
        
        if (type_fusion=='txt'):             
            roc=clf_txt.fit_predict_score (X_train_txt,Y_train_txt,scoring='roc_auc_score')
            precision=clf_txt.fit_predict_score (X_train_txt,Y_train_txt,scoring='prc_n_score')
            y_test_pred_tag= 'NN' if y_test_pred_txt[0] == 0 else 'N'
      
        cont_img=len(X_train_img)
        X_train_img=np.append(X_train_img,X_test_img[0])
        Y_train_img=np.append(Y_train_img,Y_test_img[0])
        X_train_img = X_train_img.reshape(cont_img +1,encoding_dim)
        if (type_fusion=='img'):           
            roc=clf_img.fit_predict_score (X_train_img,Y_train_img,scoring='roc_auc_score')
            precision=clf_img.fit_predict_score (X_train_img,Y_train_img,scoring='prc_n_score')
            y_test_pred_tag= 'NN' if y_test_pred_img[0] == 0 else 'N'
        
       
        list_roc.append(roc)
        list_precision.append(precision)
        
        
        newrow=[row[0],row[1],row[3],row[4],row[11],row[15],row[13],y_test_pred_tag,row[14],roc,precision,y_prob_txt[0][1],y_prob_txt[0][0],type_fusion]
        rows.append(newrow)
   
        #Cada noticia detectada como novidade (idtweet, rotulo verdadeiro, probabilidades) em cada passo no tempo      
        #ver algo no tempo
    
    mean_roc = np.mean(list_roc)
    mean_precision= np.mean(list_precision)
    
    return mean_roc,mean_precision,rows


def algorithm(X,Y, df_filter_pre,param):
    Y=[1 if x==0 else 0 for x in Y]

    
    X_train=X[0:len(X) -len(df_filter_pre),:]
    Y_train=Y[0:len(Y) -len(df_filter_pre)]
    X_test=X[len(X) -len(df_filter_pre):len(X),:]
    Y_test=Y[len(Y) -len(df_filter_pre):len(Y)]
    
    list_roc=[]
    list_precision=[]
    
    rows=[]
    for i, row in enumerate(df_filter_pre.values): 
        
         # train kNN detector
           
        #clf = KNN(n_neighbors=param)
        #clf = IForest(n_estimators=param)
        #clf = HBOS(n_bins=param)
        #clf = ABOD(contamination=0.1,n_neighbors=param,method='fast')
        clf = AutoEncoder(epochs=param,verbose=0)
        #clf = MCD(contamination=0.1)
        #clf = FeatureBagging(n_estimators=param)
        
        
     
        
        
        clf.fit(X_train)
        
        
        y_test_pred = clf.predict(X_test[i].reshape(1,-1))  # outlier labels (0 or 1)
       
        y_prob = clf.predict_proba(X_test[i].reshape(1,-1), method='unify')  
        
    
    
    
        cont=len(X_train)
        X_train=np.append(X_train,X_test[0])
        Y_train=np.append(Y_train,Y_test[0])
        X_train = X_train.reshape(cont +1,encoding_dim)
        
        
        #y_train_scores = clf.decision_function(X_train)  # outlier score
        #y_train_prob = clf.predict_proba(X_train, method='unify') 
        try:
            roc=clf.fit_predict_score (X_train,Y_train,scoring='roc_auc_score')
        except:
            print("ROC - An exception occurred") 
            roc=0
             
        try:     
            precision=clf.fit_predict_score (X_train,Y_train,scoring='prc_n_score')
        except:
            print("Precision - An exception occurred") 
            precision=0
        
        list_roc.append(roc)
        list_precision.append(precision)
        
        y_test_pred_tag= 'NN' if y_test_pred == 0 else 'N'
        newrow=[row[0],row[1],row[3],row[4],row[11],row[15],row[13],y_test_pred_tag,row[14],roc,precision,y_prob[0][1],y_prob[0][0]]
        rows.append(newrow)
   
        #Cada noticia detectada como novidade (idtweet, rotulo verdadeiro, probabilidades) em cada passo no tempo      
        #ver algo no tempo
    
    mean_roc = np.mean(list_roc)
    mean_precision= np.mean(list_precision)
    
    return mean_roc,mean_precision,rows


if __name__ == "__main__":
    
    print('\n Beginning read images and text \n \n')
    readGoogle()
    
    
    df = pd.read_csv(fname + '/tweets_labeling_en_w10_confere_new.csv')    
    df_COCO = pd.read_csv(fname + '/tweets_labeling_en_w10_confere_new_COCO.csv')    
    
    
    #df = pd.read_csv(fname + '/tweets_labeling_en_w20_teste1.csv')    
    #df_COCO = pd.read_csv(fname + '/tweets_labeling_en_w20_teste_COCO1.csv')    
    
    
    
    df=df.sort_values(by=['created_at'],ascending=True )
    df_COCO=df_COCO.sort_values(by=['created_at'],ascending=True )
    
    list_past_txt=[]
    list_pres_txt=[]
    list_total_txt=[]
    
     
    list_past_txt_COCO=[]
    list_pres_txt_COCO=[]
    list_total_txt_COCO=[]
    
    list_past_img=[]
    list_pres_img=[]
    list_total_img=[]
    
    y_past=[]
    y_pres=[]
    y_total=[]
    
    data_now=0
    count_window=0
    name_window_pres=''
    name_window_past=''
    
    df_all=df
    df_all_COCO=df_COCO
    
    list_final_roc_txt=[]
    list_final_precision_txt=[]    
    list_final_roc_txt_COCO=[]
    list_final_precision_txt_COCO=[] 
    list_final_roc_img=[]
    list_final_precision_img=[]
    list_final_roc_fusion=[]
    list_final_precision_fusion=[]
    
    list_final_mean_doc=[]
    list_final_max_seq_len=[]
    list_final_mean_doc_COCO=[]
    list_final_max_seq_len_COCO=[]
    rows_total_txt_COCO=[]
    rows_total_txt=[]
    rows_total_img=[]
    rows_total_fusion=[]
    for index, row in df.iterrows():    
        print('record:' + str(index))
        if data_now!=row.name_window:
            print('Window name:' + str(row.name_window))
            list_total_txt = list_past_txt + list_pres_txt
            list_total_txt_COCO = list_past_txt_COCO + list_pres_txt_COCO
            list_total_img = list_past_img + list_pres_img
            y_total= y_past + y_pres
            
            if len(list_past_txt)!=0:
                #normalizade one hot class
                Yi,Yn=normalizeclass(y_total)
                print('Image coding')
                enc_img = enconding_img1(list_total_img,Yn)
                print('Text coding')
                enc_txt,mean_doc,max_seq_len = enconding_txt(list_total_txt,Yn)
                enc_txt_COCO,mean_doc_COCO,max_seq_len_COCO = enconding_txt(list_total_txt_COCO,Yn)
                
                df_filter=df_all.loc[df_all['name_window'].isin([name_window_pres])]
                df_filter_COCO=df_all_COCO.loc[df_all_COCO['name_window'].isin([name_window_pres])]
                
                
                print('Finished coding')
                #knn40 1,4,coco 3
                print('Begging TXT Algorithm')
                txt_mean_roc,txt_mean_precision,rows_txt= algorithm(enc_txt,Yn,df_filter,5)
                print('Finished TXT Algorithm')
                
                #y_txt_hat =t=[row[7] for row in rows_txt]
                print('Begging IMG Algorithm')
                img_mean_roc,img_mean_precision,rows_img= algorithm(enc_img,Yn,df_filter,5)
                print('Finished IMG Algorithm')
                
                #y_img_hat =[row[7] for row in rows_img]
                print('Begging FUSION Algorithm')
                fusion_mean_roc,fusion_mean_precision,rows_fusion = algorithmFusion(enc_txt,Yn,enc_img, df_filter,5)
                print('Finished FUSION Algorithm')
                
                #https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
                print('Begging COCO Algorithm')
                txt_mean_roc_COCO,txt_mean_precision_COCO,rows_txt_COCO= algorithm(enc_txt_COCO,Yn,df_filter_COCO,5)
                print('Finished COCO Algorithm')
                
                
                rows_total_txt.extend(rows_txt)
                rows_total_txt_COCO.extend(rows_txt_COCO)
                rows_total_img.extend(rows_img)
                rows_total_fusion.extend(rows_fusion)
                
                list_final_roc_txt.append(txt_mean_roc)
                list_final_precision_txt.append(txt_mean_precision)
                list_final_roc_txt_COCO.append(txt_mean_roc_COCO)
                list_final_precision_txt_COCO.append(txt_mean_precision_COCO)
                list_final_roc_img.append(img_mean_roc)
                list_final_precision_img.append(img_mean_precision)
                list_final_roc_fusion.append(fusion_mean_roc)
                list_final_precision_fusion.append(fusion_mean_precision)
                list_final_mean_doc.append(mean_doc)
                list_final_max_seq_len.append(max_seq_len)
                list_final_mean_doc_COCO.append(mean_doc_COCO)
                list_final_max_seq_len_COCO.append(max_seq_len_COCO)
                
            name_window_past=name_window_pres
            list_past_txt=list_pres_txt            
            list_past_txt_COCO=list_pres_txt_COCO        
            list_past_img=list_pres_img
            y_past=y_pres
            
            list_pres_txt=[]
            list_pres_txt_COCO=[]
            list_pres_img=[]
            y_pres=[]
            
            count_window=count_window+1           
            data_now=row.name_window
        
        name_window_pres=row.name_window
        list_pres_txt.append(row.text_normal)
        list_pres_txt_COCO.append(df_all_COCO.iloc[index]['text_normal'])
        list_pres_img.append(row.author + '_' + str(row.id) )
        y_pres.append(row.tag_manual)
        
        
        
    df_algorithm_txt= pd.DataFrame(rows_total_txt, columns=['id','author','date_time','date', 'name_window','text_normal','tag_manual','tag_predic','audience','roc','precision','probNovel','probNotNovel'])
    df_algorithm_txt_COCO= pd.DataFrame(rows_total_txt_COCO, columns=['id','author','date_time','date', 'name_window','text_normal','tag_manual','tag_predic','audience','roc','precision','probNovel','probNotNovel'])
    df_algorithm_img= pd.DataFrame(rows_total_img, columns=['id','author','date_time','date', 'name_window','text_normal','tag_manual','tag_predic','audience','roc','precision','probNovel','probNotNovel'])
    df_algorithm_fusion= pd.DataFrame(rows_total_fusion, columns=['id','author','date_time','date', 'name_window','text_normal','tag_manual','tag_predic','audience','roc','precision','probNovel','probNotNovel','type_fusion'])
    
    df_algorithm_txt.to_csv(fname_result + '/result_autoencoder_w20_txt.csv', encoding='utf-8', index=False)
    df_algorithm_txt.to_csv(fname_result + '/result_autoencoder_w20_txt_COCO.csv', encoding='utf-8', index=False)
    df_algorithm_img.to_csv(fname_result + '/result_autoencoder_w20_img.csv', encoding='utf-8', index=False)
    df_algorithm_fusion.to_csv(fname_result + '/result_autoencoder_w20_fusion.csv', encoding='utf-8', index=False)
    
    
    '''  
    df_algorithm_txt.to_csv(fname_result + '/result_knn_w40_txt.csv', encoding='utf-8', index=False)
    df_algorithm_txt.to_csv(fname_result + '/result_knn_w40_txt_COCO.csv', encoding='utf-8', index=False)
    df_algorithm_img.to_csv(fname_result + '/result_knn_w40_img.csv', encoding='utf-8', index=False)
    df_algorithm_fusion.to_csv(fname_result + '/result_knn_w40_fusion.csv', encoding='utf-8', index=False)
    
  
    df_algorithm_txt.to_csv(fname_result + '/result_lof_txt.csv', encoding='utf-8', index=False)
    df_algorithm_txt.to_csv(fname_result + '/result_lof_txt_COCO.csv', encoding='utf-8', index=False)
    df_algorithm_img.to_csv(fname_result + '/result_lof_img.csv', encoding='utf-8', index=False)
    df_algorithm_fusion.to_csv(fname_result + '/result_lof_fusion.csv', encoding='utf-8', index=False)
   
  
    df_algorithm_txt.to_csv(fname_result + '/result_iforest_txt.csv', encoding='utf-8', index=False)
    df_algorithm_txt.to_csv(fname_result + '/result_iforest_txt_COCO.csv', encoding='utf-8', index=False)
    df_algorithm_img.to_csv(fname_result + '/result_iforest_img.csv', encoding='utf-8', index=False)
    df_algorithm_fusion.to_csv(fname_result + '/result_iforest_fusion.csv', encoding='utf-8', index=False)
    
       
    df_algorithm_txt.to_csv(fname_result + '/result_abod_txt.csv', encoding='utf-8', index=False)
    df_algorithm_txt.to_csv(fname_result + '/result_abod_txt_COCO.csv', encoding='utf-8', index=False)
    df_algorithm_img.to_csv(fname_result + '/result_abod_img.csv', encoding='utf-8', index=False)
    df_algorithm_fusion.to_csv(fname_result + '/result_abod_fusion.csv', encoding='utf-8', index=False)
     '''
     
     
    print('Total window: ' + str(count_window))      
    print('TXT  - ROC mean: ' + str(np.mean(list_final_roc_txt)))      
    print('TXT  - Precision mean: ' + str(np.mean(list_final_precision_txt)))      
    print('COCO - ROC mean: ' + str(np.mean(list_final_roc_txt_COCO)))      
    print('COCO - Precision mean: ' + str(np.mean(list_final_precision_txt_COCO)))      
    print('IMG  - ROC mean: ' + str(np.mean(list_final_roc_img)))      
    print('IMG  - Precision mean: ' + str(np.mean(list_final_precision_img)))      
    print('FUSION - ROC mean: ' + str(np.mean(list_final_roc_fusion)))      
    print('FUSION - Precision mean: ' + str(np.mean(list_final_precision_fusion)))      
    print('Doc character mean: ' + str(np.mean(list_final_mean_doc)))      
    print('Doc max character mean: ' + str(np.mean(list_final_max_seq_len)))      
    print('Doc character mean COCO: ' + str(np.mean(list_final_mean_doc_COCO)))      
    print('Doc max character mean COCO: ' + str(np.mean(list_final_max_seq_len_COCO)))        
  
    
    #IMPLEMENTAR O QUE O PATRICK PEDIU
    
    #https://pyod.readthedocs.io/en/latest/example.html?highlight=ROC
    
    
    