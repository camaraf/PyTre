#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import scipy.spatial
import matplotlib.cm as cm
import scipy.stats
from pprint import pprint
import tensorflow as tf
import keras

import math
from collections import OrderedDict 


from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import BatchNormalization, Input, concatenate
import types
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


import pylogit as pl                   # For MNL model estimation and
                        
import PyTre


# In[2]:


SPLIT_EMBEDDINGS_DCM=.80
TRAINSET_WITHOUT_DEV=.75


# In[3]:


sw_df_full=pd.read_csv('swissmetro_rand.dat')#,sep='\t')


# In[4]:


sw_df,_=train_test_split(sw_df_full, train_size=SPLIT_EMBEDDINGS_DCM, shuffle=False)


# In[5]:


tickets={0: 'None', 1: '2 way w 1/2 price', 2: '1 way w 1/2 price', 3: '2 way normal price', 4: '1 way normal price', 5: 'Half day', 6: 'Annual ticket', 7: 'Annual ticket Junior or Senior', 8: 'Free travel after 7pm', 9: 'Group ticket', 10: 'Other'}
ages={1: 'age≤24', 2: '24<age≤39', 3: '39<age≤54', 4: '54<age≤ 65', 5: '65 <age', 6: 'not known'}
incomes={0:'under 50', 1: 'under 50', 2: 'between 50 and 100', 3: 'over 100', 4: 'unknown'}
whos={0: 'unknown', 1: 'self', 2: 'employer', 3: 'half-half'}
cantons={1:'ZH', 2: 'BE', 3: 'LU', 4:'UR', 5:'SZ', 6:'OW', 7:'NW', 8:'GL', 9:'ZG', 10: 'FR', 11: 'SO', 12:'BS', 13:'BL', 
         14:'SH', 15: 'AR', 16:'AI', 17:'SG', 18:'GR', 19:'AG', 20: 'TH', 21: 'TI', 22:'VD', 23:'VS', 24: 'NE', 25:'GE', 26:'JU'}


# In[6]:


def update_table_codes(dic, df, column):
    for t in dic:
        df[column]=df[column].replace(t, dic[t])
    return df
    


# In[7]:


sw_df=update_table_codes(tickets, sw_df, 'TICKET')
sw_df=update_table_codes(ages, sw_df, 'AGE')
sw_df=update_table_codes(incomes, sw_df, 'INCOME')
se_df=update_table_codes(whos, sw_df, 'WHO')


# In[ ]:





# In[8]:


def pair_up(df, pair_name, varlist, enforce_int=False): #ONLY for categorical!!
    new_df=df[:]
    new_var=[]
    for index, row in new_df.iterrows():
        nv=""
        for v in varlist:
            val=row[v]
            if enforce_int:
                val=int(val)
            #nv+=str(val)+"_"
            nv+=cantons[val]+"_"
        nv=nv[:-1]
        new_var.append(nv)
    new_df[pair_name]=new_var
    
    return new_df


# In[9]:


sw_df=pair_up(sw_df, "OD", ['ORIGIN', 'DEST'])


# In[10]:


emb_vars=['OD', 'TICKET', 'WHO', 'AGE', 'INCOME']
EMB_SIZES=[10, 3, 1,  2, 2]
EMB_SIZES=[55,5,2,3,3]


# In[11]:


ex_vars=['CAR_TT', 'TRAIN_TT', 'SM_TT', 'FIRST', 'SURVEY']


# In[12]:


y_var=['CHOICE']




# ------------------------------------------


long_swiss_metro_full=pd.read_csv("long_swissmetro_rand.dat")


# In[20]:


long_swiss_metro=pair_up(long_swiss_metro_full, "OD", ['ORIGIN', 'DEST'], True)


# In[21]:


long_swiss_metro=update_table_codes(tickets, long_swiss_metro, 'TICKET')
long_swiss_metro=update_table_codes(ages, long_swiss_metro, 'AGE')
long_swiss_metro=update_table_codes(incomes, long_swiss_metro, 'INCOME')
long_swiss_metro=update_table_codes(whos, long_swiss_metro, 'WHO')


# In[22]:


# Create the list of individual specific variables
ind_variables = sw_df.columns.tolist()[:15]

# Specify the variables that vary across individuals and some or all alternatives
# The keys are the column names that will be used in the long format dataframe.
# The values are dictionaries whose key-value pairs are the alternative id and
# the column name of the corresponding column that encodes that variable for
# the given alternative. Examples below.
alt_varying_variables = {u'travel_time': dict([(1, 'TRAIN_TT'),
                                               (2, 'SM_TT'),
                                               (3, 'CAR_TT')]),
                          u'travel_cost': dict([(1, 'TRAIN_CO'),
                                                (2, 'SM_CO'),
                                                (3, 'CAR_CO')]),
                          u'headway': dict([(1, 'TRAIN_HE'),
                                            (2, 'SM_HE')]),
                          u'seat_configuration': dict([(2, "SM_SEATS")])}

# Specify the availability variables
# Note that the keys of the dictionary are the alternative id's.
# The values are the columns denoting the availability for the
# given mode in the dataset.
availability_variables = {1: 'TRAIN_AV',
                          2: 'SM_AV', 
                          3: 'CAR_AV'}

##########
# Determine the columns for: alternative ids, the observation ids and the choice
##########
# The 'custom_alt_id' is the name of a column to be created in the long-format data
# It will identify the alternative associated with each row.
custom_alt_id = "mode_id"

# Create a custom id column that ignores the fact that this is a 
# panel/repeated-observations dataset. Note the +1 ensures the id's start at one.
obs_id_column = "custom_id"
sw_df[obs_id_column] = np.arange(sw_df.shape[0],
                                            dtype=int) + 1


# Create a variable recording the choice column
choice_column = "CHOICE"


# In[23]:


##########
# Create scaled variables so the estimated coefficients are of similar magnitudes
##########
# Scale the travel time column by 60 to convert raw units (minutes) to hours
long_swiss_metro["travel_time_hrs"] = long_swiss_metro["travel_time"] / 60.0

# Scale the headway column by 60 to convert raw units (minutes) to hours
long_swiss_metro["headway_hrs"] = long_swiss_metro["headway"] / 60.0

# Figure out who doesn't incur a marginal cost for the ticket
# This can be because he/she owns an annual season pass (GA == 1) 
# or because his/her employer pays for the ticket (WHO == 2).
# Note that all the other complexity in figuring out ticket costs
# have been accounted for except the GA pass (the annual season
# ticket). Make sure this dummy variable is only equal to 1 for
# the rows with the Train or Swissmetro
long_swiss_metro["free_ticket"] = (((long_swiss_metro["GA"] == 1) |
                                    (long_swiss_metro["WHO"] == 2)) &
                                   long_swiss_metro[custom_alt_id].isin([1,2])).astype(int)
# Scale the travel cost by 100 so estimated coefficients are of similar magnitude
# and acccount for ownership of a season pass
long_swiss_metro["travel_cost_hundreth"] = (long_swiss_metro["travel_cost"] *
                                            (long_swiss_metro["free_ticket"] == 0) /
                                            100.0)

##########
# Create various dummy variables to describe the choice context of a given
# invidual for each choice task.
##########
# Create a dummy variable for whether a person has a single piece of luggage
long_swiss_metro["single_luggage_piece"] = (long_swiss_metro["LUGGAGE"] == 1).astype(int)

# Create a dummy variable for whether a person has multiple pieces of luggage
long_swiss_metro["multiple_luggage_pieces"] = (long_swiss_metro["LUGGAGE"] == 3).astype(int)

# Create a dummy variable indicating that a person is NOT first class
long_swiss_metro["regular_class"] = 1 - long_swiss_metro["FIRST"]

# Create a dummy variable indicating that the survey was taken aboard a train
# Note that such passengers are a-priori imagined to be somewhat partial to train modes
long_swiss_metro["train_survey"] = 1 - long_swiss_metro["SURVEY"]


# In[24]:


def create_spec():
    # NOTE: - Specification and variable names must be ordered dictionaries.
    #       - Keys should be variables within the long format dataframe.
    #         The sole exception to this is the "intercept" key.
    #       - For the specification dictionary, the values should be lists
    #         of integers or lists of lists of integers. Within a list, 
    #         or within the inner-most list, the integers should be the 
    #         alternative ID's of the alternative whose utility specification 
    #         the explanatory variable is entering. Lists of lists denote 
    #         alternatives that will share a common coefficient for the variable
    #         in question.

    basic_specification = OrderedDict()
    basic_names = OrderedDict()

    basic_specification["intercept"] = [1, 2]
    basic_names["intercept"] = ['ASC Train',
                                'ASC Swissmetro']

    basic_specification["travel_time_hrs"] = [[1, 2,], 3]
    basic_names["travel_time_hrs"] = ['Travel Time, units:hrs (Train and Swissmetro)',
                                      'Travel Time, units:hrs (Car)']

    basic_specification["travel_cost_hundreth"] = [1, 2, 3]
    basic_names["travel_cost_hundreth"] = ['Travel Cost * (Annual Pass == 0), units: 0.01 CHF (Train)',
                                           'Travel Cost * (Annual Pass == 0), units: 0.01 CHF (Swissmetro)',
                                           'Travel Cost, units: 0.01 CHF (Car)']

    basic_specification["headway_hrs"] = [1, 2]
    basic_names["headway_hrs"] = ["Headway, units:hrs, (Train)",
                                  "Headway, units:hrs, (Swissmetro)"]

    basic_specification["seat_configuration"] = [2]
    basic_names["seat_configuration"] = ['Airline Seat Configuration, base=No (Swissmetro)']

    basic_specification["train_survey"] = [[1, 2]]
    basic_names["train_survey"] = ["Surveyed on a Train, base=No, (Train and Swissmetro)"]

    basic_specification["regular_class"] = [1]
    basic_names["regular_class"] = ["First Class == False, (Swissmetro)"]

    basic_specification["single_luggage_piece"] = [3]
    basic_names["single_luggage_piece"] = ["Number of Luggage Pieces == 1, (Car)"]

    basic_specification["multiple_luggage_pieces"] = [3]
    basic_names["multiple_luggage_pieces"] = ["Number of Luggage Pieces > 1, (Car)"]
    
    return basic_specification, basic_names


# In[25]:


basic_specification, basic_names=create_spec()


# In[26]:


all_ODs=list(long_swiss_metro['OD'].unique())


# In[35]:

pt=PyTre.EmbeddingsModel()


    # In[14]:


m, embs=pt.fit(sw_df[emb_vars], sw_df[y_var], sw_df[ex_vars], xlabels=emb_vars, EPOCHS=80, VALIDATION_SPLIT=0.0, verbose=0)#, EMB_SIZE=EMB_SIZES)



remove_ODs=[od for od in all_ODs if od not in list(pt.embeddings_dic['OD']['index2alfa_from'].values())]
for r in remove_ODs:
    long_swiss_metro=long_swiss_metro[long_swiss_metro.OD!=r]



# In[36]:


long_swiss_metro_train,long_swiss_metro_test=train_test_split(long_swiss_metro, train_size=SPLIT_EMBEDDINGS_DCM, shuffle=False)




# In[39]:


testsetsize=len(long_swiss_metro_test)/3
trainsetsize=len(long_swiss_metro_train)/3
print(testsetsize, trainsetsize)


# In[40]:


# Estimate the multinomial logit model (MNL)
swissmetro_mnl = pl.create_choice_model(data=long_swiss_metro_train,
                                        alt_id_col=custom_alt_id,
                                        obs_id_col=obs_id_column,
                                        choice_col=choice_column,
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)


deg_freedom=sum([len(b) for b in basic_specification.values()])
# Specify the initial values and method for the optimization.
swissmetro_mnl.fit_mle(np.zeros(deg_freedom))

# Look at the estimation results
swissmetro_mnl.get_statsmodels_summary()


# In[41]:


long_probs=swissmetro_mnl.predict(long_swiss_metro_test)
SCORE=sum([np.log(x) for x in np.multiply(long_probs, long_swiss_metro_test['CHOICE']) if x!=0.0])

print(SCORE)

obs_ll=swissmetro_mnl.null_log_likelihood/swissmetro_mnl.nobs
rsq=1-(SCORE/(obs_ll*testsetsize))
adjrsq=rsq-(1-rsq)*deg_freedom/(testsetsize-deg_freedom-1)
AIC=2*deg_freedom-2*SCORE
BIC=math.log(testsetsize)*deg_freedom-2*SCORE
print("model results for original model (%d observations)"%testsetsize)
print("R^2 test=%f & adjusted R^2 test=%f"%(rsq, adjrsq))
print("AIC test=%f & BIC test=%f"%(AIC, BIC))    




best_ll=-99999999
best_r2=0
best_AIC=0
best_BIC=0
best_ll_dev=-99999999
best_ll_t=0

embruns_stat={}
for runs in range(300):

    print("Now with embeddings...")
    pt=PyTre.Discrete()


    # In[14]:

    sw_df_train,_=train_test_split(sw_df, train_size=TRAINSET_WITHOUT_DEV, shuffle=False)


    m, embs=pt.fit(sw_df_train[emb_vars], sw_df_train[y_var], sw_df_train[ex_vars], xlabels=emb_vars, EPOCHS=80, VALIDATION_SPLIT=0.0, verbose=0)#, EMB_SIZE=EMB_SIZES)


    # In[17]:


    OUT_EMB_SIZES=[]

    for e in emb_vars:
        print("Original DIM:", embs[e]['dim'])
        #print("EMB SIZE:", embs[e][''])
        print(embs[e]['embeddings'].shape)
        OUT_EMB_SIZES.append(embs[e]['embeddings'].shape[1])



    new_long_sw_df=pt.replace_with_embeddings(long_swiss_metro, emb_vars)


    # In[ ]:





    # In[80]:


    emb_vars_long=[item for sublist in [[v for v in new_long_sw_df.columns if v.startswith(x)] for x in emb_vars] for item in sublist]


    # In[81]:


    exclude_ODs=[d for d in emb_vars_long if not d.startswith("OD")]


    # In[82]:


    basic_specification, basic_names=create_spec()
    for emb in emb_vars_long:
        if emb.startswith("TICKET"):
            basic_specification[emb] = [1]
            basic_names[emb] = [emb+"_Train"]
        else:
            basic_specification[emb] = [1,2]
            basic_names[emb] = [emb+"_Train", emb+"_SM"]


    # In[83]:


    from sklearn.preprocessing import StandardScaler
    new_long_sw_df[emb_vars_long]=StandardScaler().fit_transform(new_long_sw_df[emb_vars_long])


    # In[84]:


    new_long_sw_df_train,new_long_sw_df_test=train_test_split(new_long_sw_df, train_size=SPLIT_EMBEDDINGS_DCM, shuffle=False)



    new_long_sw_df_train_without_dev,new_long_sw_df_dev=train_test_split(new_long_sw_df_train, train_size=TRAINSET_WITHOUT_DEV, shuffle=False)

    # In[86]:

    embruns_stat[runs]={}
    # Estimate the multinomial logit model (MNL)
    swissmetro_mnl = pl.create_choice_model(data=new_long_sw_df_train_without_dev,
                                            alt_id_col=custom_alt_id,
                                            obs_id_col=obs_id_column,
                                            choice_col=choice_column,
                                            specification=basic_specification,
                                            model_type="MNL",
                                            names=basic_names)

    deg_freedom=sum([len(b) for b in basic_specification.values()])
    # Specify the initial values and method for the optimization.
    swissmetro_mnl.fit_mle(np.zeros(deg_freedom))

    # Look at the estimation results
    swissmetro_mnl.get_statsmodels_summary() 
    embruns_stat[runs]['train_log_likelihood']=swissmetro_mnl.llf
    embruns_stat[runs]['train_r2']=swissmetro_mnl.rho_squared
    embruns_stat[runs]['train_adjr2']=swissmetro_mnl.rho_bar_squared
    embruns_stat[runs]['train_AIC']=swissmetro_mnl.aic
    embruns_stat[runs]['train_BIC']=swissmetro_mnl.bic


    # In[87]:

    long_probs_dev=swissmetro_mnl.predict(new_long_sw_df_dev)
    SCORE_DEV=sum([np.log(x) for x in np.multiply(long_probs_dev, new_long_sw_df_dev['CHOICE']) if x!=0.0])


    long_probs=swissmetro_mnl.predict(new_long_sw_df_test)
    SCORE=sum([np.log(x) for x in np.multiply(long_probs, new_long_sw_df_test['CHOICE']) if x!=0.0])
    

    print(SCORE)


    obs_ll=swissmetro_mnl.null_log_likelihood/swissmetro_mnl.nobs
    rsq=1-(SCORE/(obs_ll*testsetsize))
    adjrsq=rsq-(1-rsq)*deg_freedom/(testsetsize-deg_freedom-1)
    AIC=2*deg_freedom-2*SCORE
    BIC=math.log(testsetsize)*deg_freedom-2*SCORE
    print("model results for global embeddings model")
    print("R^2 test=%f & adjusted R^2 test=%f"%(rsq, adjrsq))
    print("AIC test=%f & BIC test=%f"%(AIC, BIC))    
    embruns_stat[runs]['test_log_likelihood']=SCORE
    embruns_stat[runs]['test_r2']=rsq
    embruns_stat[runs]['test_adjr2']=adjrsq
    embruns_stat[runs]['test_AIC']=AIC
    embruns_stat[runs]['test_BIC']=BIC


    
    if SCORE_DEV>best_ll_dev:
        best_ll=SCORE
        best_r2=rsq
        best_ll_dev=SCORE_DEV
        best_ll_t=swissmetro_mnl.llf
        best_adjr2=adjrsq
        best_AIC=AIC
        best_BIC=BIC

        print(swissmetro_mnl.get_statsmodels_summary().as_latex())
        print("variable name\tcoeff\tstderr\tz-score\tp-value")

        count=0
        countsig=0
        for var in ['OD', 'WHO', 'TICKET', 'INCOME', 'AGE']:#emb_vars:#[d for d in emb_vars if d!='OD']:

            alfs=pt.embeddings_dic[var]['index2alfa_from']
            embs=pt.embeddings_dic[var]['embeddings']

            alternatives=['Train', 'SM']
            if var in ["TICKET"]:#, "OD"]:
                alternatives=['Train']
            for m in alternatives:
                for i in range(embs.shape[0]):
                    mean=0
                    stde=0
                    mean_default=0
                    stde_default=0
                    for j in range(embs.shape[1]):
                        mean+=embs[i,j]*swissmetro_mnl.coefs[var+str(j)+"_"+m]
                        stde+=embs[i,j]**2*swissmetro_mnl.standard_errors[var+str(j)+"_"+m]**2
                    if i==0:
                        stde_default=stde
                        mean_default=mean
                        continue
                    stde=math.sqrt(stde+stde_default)
                    z_score=(mean-mean_default)/stde        
                    p_value = scipy.stats.norm.sf(abs(z_score))*2 #twosided
                    signif=""
                    if p_value<0.05:
                        signif="**"
                        countsig+=1
                    elif p_value<0.1:
                        signif="*"
                        countsig+=1
                    #print(var+str(i)+"_"+m+"\t%.3f\t%.3f\t%.3f\t%.3f%s"% (mean, stde, z_score, p_value, signif))
                    #print(alfs[i]+"_"+m+"\t%.3f\t%.3f\t%.3f\t%.3f%s"% (mean, stde, z_score, p_value, signif))
                    if signif!="" or var!="OD":
                        print(var+"_"+alfs[i]+"_"+m+"&%.3f&%.3f&%.3f&%.3f%s\\\\"% (mean, stde, z_score, p_value, signif))
                    count+=1



        
            



        
            

    with open("multiple_runs.csv", "a") as f:
        for key in embruns_stat[runs].keys():
            f.write("%s, %s, "%(key, embruns_stat[runs][key]))
        f.write("\n")   


print("Results of best train model=")
print("LL (train)=%f"%(best_ll_t))
print("LL=%f, R^2=%f, adjr^2=%f, AIC=%f, BIC=%f"%(best_ll, best_r2, best_adjr2, best_AIC, best_BIC))
