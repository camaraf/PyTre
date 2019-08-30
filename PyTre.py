import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import scipy.spatial
import matplotlib.cm as cm
import scipy.stats
import tensorflow as tf
import keras
import math
from collections import OrderedDict 


from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import BatchNormalization, Input, concatenate
import tempfile
import keras.models
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.callbacks import ModelCheckpoint
import keras.backend as K

from sklearn.linear_model  import LogisticRegressionCV, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from adjustText import adjust_text


import pylogit as pl                   # For MNL model estimation and
                        

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)



def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

class EmbeddingsModel:
    ''' Class variables '''


    
    '''Init functions'''
    
    def __init__(self, EMBEDDING_SIZE=2, EPOCHS=100, verbose=False):
        self.EMBEDDING_SIZE=EMBEDDING_SIZE
        self.EPOCHS=EPOCHS
        self.embeddings=None
        self.index2alfa_from=None
        self.alfa2index=None
        self.EXOG_DIM=None
        self.INPUT_DIM=None


    
    '''Utilities functions'''
    def create_index(self, alfabet):
        index2alfa={}
        alfa2index={}
        for i in range(len(alfabet)):
            index2alfa[i]=alfabet[i]
            alfa2index[alfabet[i]]=i
        return index2alfa, alfa2index
    
    #returns mapping dicionarity alfabet -> one-hot encoding index
    def one_hot_enc(self, x, size):
        vec = np.zeros(size)
        vec[x] = 1
        return vec

    
    def visualize_embeddings(self, embeddings=[], labels={}, fromlabel="Source", tolabel="target", adjust=True):
        if labels=={}:
            labels=self.index2alfa_from

        if embeddings==[]:
            embeddings=self.embeddings

        mds = MDS(n_components=2)
        mds_result = mds.fit_transform(embeddings)
        maxwidth=max(mds_result[:,0])-min(mds_result[:,0])
        maxheight=max(mds_result[:,1])-min(mds_result[:,1])
        
        plt.scatter(mds_result[:,0], mds_result[:,1], s=100, c=range(len(embeddings)))
        if adjust:
            texts = [plt.text(mds_result[i,0], mds_result[i,1], labels[i]) for i in range(len(embeddings))]
            adjust_text(texts)
        else:
            for i in range(len(embeddings)):
                plt.annotate(labels[i], (mds_result[i,0]+0.01*maxwidth, mds_result[i,1]+0.02*maxheight))
        
        plt.title(fromlabel+" to "+tolabel+" embeddings");
        
    def visualize_estimation_performance(self):
        plt.figure(figsize=(16,6))
        plt.plot(self.model.history.history['loss'], label="Loss")
        plt.plot(self.model.history.history['val_loss'],label="Val loss")
        plt.show()
        
    def replace_with_embeddings(self, df, fromkey, dropfromkey=True, verbose=False):
        if type(fromkey)!=list:
            fromkey=[fromkey]

        newdf=df[:]
        for fk in fromkey:
            encoded=self.encode(fk, newdf[fk], verbose)

            for e in range(encoded.shape[1]):
                newdf[fk+str(e)]=encoded[:,e]
            if dropfromkey:
                del newdf[fk]
        return newdf

    def pca_embedding(self, mat, target_mat=None, varexp=0.95):
        pca=PCA()
        if type(target_mat)==pd.core.frame.DataFrame:
            pca.fit(mat)#[:,:EMB_SIZE]
            mat_trans=pca.transform(target_mat)
        else:
            mat_trans=pca.fit_transform(mat)#[:,:EMB_SIZE]
        if type(varexp)==int:
            EMB_SIZE=varexp
        else:
            cumexp=[sum(pca.explained_variance_ratio_[:i+1]) for i in range(len(pca.explained_variance_ratio_))]  
            EMB_SIZE=[i for i in range(len(cumexp)) if cumexp[i]>varexp][0] 
        if EMB_SIZE==0:
            EMB_SIZE=1
        return mat_trans[:,:EMB_SIZE], EMB_SIZE, pca


    def encode(self, key, X, verbose=False):
        encoded=[]
        for x in X:
            ''' 
            print("mapping ", x, "to ", "...")
            print(self.embeddings_dic['alfa2index_from'][x])
            print(self.embeddings_dic['embeddings'][self.embeddings_dic['alfa2index_from'][x]])
            '''
            
            #IF KEY NOT IN DIC, FILL WITH NANS, print warning
            if not x in self.embeddings_dic[key]['alfa2index_from']:
                encoded.append(np.full(self.embeddings_dic[key]['embeddings'].shape[1], 0.0))
                if verbose:
                    print("WARNING: Could not find embeddings for ",x, " in variable ", key)
            else:
                encoded.append(self.embeddings_dic[key]['embeddings'][self.embeddings_dic[key]['alfa2index_from'][x]])
        return np.array(encoded) 

    
    def plot_distance_histogram(self, embs, bins=None):
        if embs in self.embeddings_dic.keys():
            embs=self.embeddings_dic[embs]['embeddings']
            
        distances=[]
        
        for e1 in embs:
            for e2 in embs:
                if (e1==e2).all():
                    continue
                distances.append(np.linalg.norm(e1-e2))
                
        if bins!=None:
            plt.hist(distances, bins)
        else:
            plt.hist(distances)
        plt.show()
  
    def save_embeddings(self, fromkey, tokey):
        f=open(fromkey+"_to_"+tokey+".embeddings", "w")
        for w in range(len(self.index2alfa_from)):
            st=str(self.index2alfa_from[w])+' ,'
            for val in self.embeddings[w][:-1]:
                st+=str(val)+", "
            st+=str(self.embeddings[w][-1])
            f.write(st+"\n")
        f.close()
   

    def save_model(self, fromkey, tokey):
        
        filename=fromkey+"_to_"+tokey+".pickle"
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        self.model.save(fromkey+"_"+tokey+".h5")

    def load_model_pickle(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 

        
    def load_model_h5(self, filename):
        self.model=load_model(filename) 

    
    '''Training functions'''
    def fit(self,x, y, exogenous, xlabels=[], EMB_SIZE='auto', varexp=0.9, CRC=True, verbose=1, EPOCHS=None, EMB_BIAS=True, LOSS='categorical_crossentropy', VALIDATION_SPLIT=0.3):
        '''
        if EPOCHS==None:
            EPOCHS=self.EPOCHS
        '''
        if type(x)==list or type(x)==pd.core.frame.DataFrame:
            x=np.array(x)
        if type(y)==list:
            y=np.array(y)
        if type(y)==pd.core.frame.DataFrame:
            y=y.values
        y=y.ravel()
        
        self.N_embeddings=x.shape[1]
        if EMB_SIZE=='auto':
            EMB_S=[]
            for i in range(self.N_embeddings):
                mat=pd.DataFrame(np.array(x).T[i])
                mat=pd.get_dummies(mat, columns=mat.columns)
                _,EMB_S_,_=self.pca_embedding(mat, varexp)
                print("Automatic determination of embedding size (PCA, varexp=%f) %s -> %s (reduction from %f)"%(varexp, xlabels[i], EMB_S_, len(np.unique(np.array(x).T[i]))))
                #print(mat)
                EMB_S.append(EMB_S_)
            EMB_SIZE=EMB_S
            
        elif type(EMB_SIZE)==float:
            EMB_SIZE=[int(math.ceil(len(np.unique(x_var))*EMB_SIZE)) for x_var in x.T]
            if verbose:
                print([x_var for x_var in x.T])

        self.EMB_SIZE=EMB_SIZE
        
        if len(xlabels)==0:
            xlabels=['emb_'+str(i) for i in range(self.N_embeddings)]
            
        
        self.embeddings_dic={}
        for i in range(self.N_embeddings):
            emb_alf={}
            emb_alf['name']=xlabels[i]
            emb_alf['dim']=len(np.unique(np.array(x).T[i]))
            emb_alf['index']=i
            emb_alf['index2alfa_from'], emb_alf['alfa2index_from']=self.create_index(np.unique(np.array(x).T[i]))
            self.embeddings_dic[xlabels[i]]=emb_alf
        
        exogenous=np.array(exogenous)    

        y_alf=np.unique(y)
        index2alfa_to, alfa2index_to=self.create_index(y_alf)

        #INPUT_DIMS=[self.embeddings_dic[xlab]['dim'] for xlab in xlabels]
        self.CLASSES=len(alfa2index_to)
        self.EXOG_DIM=exogenous.shape[1]
        
        output_space=[]
        intermediate_space=[]
        input_space=[]
        exogenous_space=[]

        for i in range(len(x.T)):
            input_i=[]
            intermediate_i=[]
            for xi in np.array(x).T[i]:
                input_i.append(self.embeddings_dic[xlabels[i]]['alfa2index_from'][xi])
                intermediate_i.append(np.array(self.one_hot_enc(self.embeddings_dic[xlabels[i]]['alfa2index_from'][xi], self.embeddings_dic[xlabels[i]]['dim'])))
                
            input_space.append(np.array(input_i))
            intermediate_space.append(np.array(intermediate_i))
    
        for yi, exi in zip(y, exogenous):
            output_space.append(np.array(self.one_hot_enc(alfa2index_to[yi], self.CLASSES)))
            exogenous_space.append(exi)
            
        #input_space = np.array(input_space)
        output_space = np.array(output_space)
        #intermediate_space = np.array(intermediate_space)
        exogenous_space=np.array(exogenous_space)

        #input_act = Input(shape=(self.INPUT_DIM,))
        hidden_flat_dropouts=[]
        aux_outputs=[]
        input_acts=[]
        for i in range(self.N_embeddings):
            input_act = Input(shape=(1,))
            hidden = Embedding(output_dim=EMB_SIZE[i], name="embeddings_"+xlabels[i], embeddings_regularizer=regularizers.l2(0.01), input_dim=self.embeddings_dic[xlabels[i]]['dim'])(input_act)
            hidden_flat = Flatten()(hidden)
            hidden_flat_dropout=Dropout(0.2)(hidden_flat)
            #embedding_layer = Dense(self.EMBEDDING_SIZE, activation='linear', 1)(input_act)
            
            if CRC:
                aux_output=Dense(self.embeddings_dic[xlabels[i]]['dim'], activation='softmax', name="crc_embeddings_"+xlabels[i], use_bias=True, kernel_regularizer=regularizers.l2(0.05))(hidden_flat_dropout)
                aux_outputs.append(aux_output)
                
            hidden_flat_dropouts.append(hidden_flat_dropout)
            input_acts.append(input_act)
        
        exog_input = Input(shape=(self.EXOG_DIM,))
        intermediate=concatenate(hidden_flat_dropouts+[exog_input])
        output_act = Dense(self.CLASSES, activation='softmax', use_bias=True, name="output_layer", kernel_regularizer=regularizers.l2(0.01))(intermediate)

        if CRC:
            self.model = Model(input_acts+[exog_input], [output_act]+ aux_outputs)
            OUT_SPACE=[output_space]+intermediate_space,
        else:
            self.model = Model(input_acts+[exog_input], [output_act])
            OUT_SPACE=[output_space]

        self.model.compile(optimizer='adam', loss=LOSS)
        
        myCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 20)
        self.model.fit(input_space+[exogenous_space], [output_space]+intermediate_space, 
                batch_size=128,
                epochs=EPOCHS,
                callbacks=[myCallback],
                validation_split=VALIDATION_SPLIT,
                verbose=verbose)
        #model.save_weights(fromkey+"_to_"+tokey+"_embeddings.pickle")
        for i in range(self.N_embeddings):
            self.embeddings_dic[xlabels[i]]['embeddings']=self.model.layers[self.N_embeddings+i].get_weights()[0]
        return self.model, self.embeddings_dic
        #embeddings=model.layers[1].get_weights()[0]

      
