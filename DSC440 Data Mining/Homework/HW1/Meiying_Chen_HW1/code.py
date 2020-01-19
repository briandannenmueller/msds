#!/usr/bin/python3 
# -*- coding: utf-8 -*-

'''
Data Mining Assignment 1, Part 2

@Vesion  :  1.0
@Time    :  09/10/2019, 17:40
@Author  :  Meiying(Melissa) Chen
@Contact :  meiying.chen@rochester.edu

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from scipy import stats
sns.set(style = "white")


# load data
data = pd.read_csv("data.csv")
# drop useless columns in dataset
data = data.drop(['Unnamed: 32','id'],axis = 1)
# check data size, features and display the first 5 lines
print('data size is: ', data.shape)
print('features: ', data.columns)

# measuring the central tendency
print("mean: ",data.texture_mean.mean())
print("median: ",data.texture_mean.median())
print("mode: ",stats.mode(data.texture_mean))


# plot distribution graph
sns.distplot(data.texture_mean, rug=True)


# box plot
melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = ['perimeter_mean', 'texture_mean'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)

# scatter plot
plt.figure(figsize = (15,10))
sns.jointplot(data.perimeter_mean,data.concavity_mean ,kind="regg")

# plot relationship between more than 2 features
df = data.loc[:,["radius_mean","area_mean","fractal_dimension_se"]]
g = sns.PairGrid(df,diag_sharey = False)
g.map_lower(sns.kdeplot,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,lw =3)

# heatmap
f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()













