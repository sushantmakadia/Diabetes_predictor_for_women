# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:16:33 2020

@author: sushant
"""

import pandas as pd
import numpy as np
import pickle as pkl

label_name=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
label_name_new=['insulin','bmi','pedigree','age','label']
dataset=pd.read_csv('pima-indians-diabetes.data',header=None,names=label_name)
dataset_new=dataset[['insulin','bmi','pedigree','age','label']]
x=dataset.drop('label',axis=1)
y=dataset['label']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
#with open('model2.pkl', 'wb') as file:
 #   pkl.dump(reg,file)
#make a prediction
#new=[[0,100,70,29,0,30,0.500,34]]
#y_pred_new=reg.predict(new)


