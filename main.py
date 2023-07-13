import classifier as classifier
import pandas as pd
import seaborn
import math
import random
import sns as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn.cluster import KMeans
df = pd.read_csv("C:/Users/PCS/Desktop/iris.csv")
print(df.head())
print(df.columns)
print(df['Species'].value_counts())
print(df.info())
print(df.sample(5))

train_set,test_set=train_test_split(df,test_size=0.3)
print(test_set.shape)
print(train_set.shape)
x_test=test_set.iloc[:,1:4]
y_test=test_set.iloc[:,5]
x_train= train_set.iloc[:,1:4]
y_train= train_set.iloc[:,5]


