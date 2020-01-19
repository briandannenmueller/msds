#!/usr/bin/python3 
# -*- coding: utf-8 -*-

'''
@Vesion  :  1.0
@Time    :  09/24/2019, 15:40
@Author  :  Meiying(Melissa) Chen
@Contact :  meiying.chen@rochester.edu

'''

import numpy as np
import matplotlib.pyplot as plt
import math

################# 3.3 #################
age = [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70]
bins = []
b = []
print(len(age))
for i in range(len(age)):
    if i % 3 == 2:
        b.append(age[i])
        bins.append(b)
        b = []
    else:
        b.append(age[i])
print(bins)

bins_mean = []
for b in bins:
    mean = np.mean(b);
    bins_mean.append([mean, mean, mean])
print(bins_mean)

bins_meian = []
for b in bins:
    median = np.median(b);
    bins_meian.append([median, median, median])
print(bins_meian)

bins_boundary = []
for b in bins:
    temp = []
    l = len(b)
    lower = b[0]
    upper = b[l-1]
    for i in range(l):
        if l-i < l/2:
            temp.append(upper)
        else:
           temp.append(lower) 
    bins_boundary.append(temp)
print(bins_boundary)


################# 3.7 ################# 
age_minmax = (age-np.min(age))/(np.max(age)-np.min(age))
print(age_minmax[17])
age_zscore = (age - np.mean(age)) / 19.24
print(age_zscore[17])
j = round(np.log10(np.max(age)))
age_decimal = age / 10**j
print(age_decimal[17])


################# 3.11 ################# 
plt.hist(age, bins = [10, 20, 30, 40, 50, 60 ,70])
plt.xlabel('age')
plt.ylabel('frequency')
plt.show()


################# 3.13 ################# 
# The automatic generation of a concept hierarchy for nominal data 
# based on the number of distinct values of attributes
def concept_hierarchy_distinct_value(attributes:dict):
    d = {}
    res = []
    for i in attributes.keys():
        d.update({len(set(attributes.get(i))):i})
    sort = np.sort(list(d.keys()))
    for s in sort:
        res.append([d.get(s),s])
    return res

# test distinct values
countries = ['China','China','China','Soviet Union', 'Yugoslavia']
provinces = ['Andong','Anhui', 'Zhili','Liaobei','Fengtian','Nanjiang','Qahar']
cities = ['Hong Kong', 'Macau','Beijing','Chongqin','Shanghai','Tianjin','Anqing','Bengbu','Bozhou','Chaohu']
attributes = {'counties':countries,'citites':cities,'provinces':provinces}
print(concept_hierarchy_distinct_value(attributes)) 

# based on the equal-width partitioning rule
def concept_hierarchy_equal_width(attribute:list, width:int):
    res = []
    sort = np.sort(attribute)
    b = []
    for i in range(len(sort)):
        if sort[i] != sort[0] and sort[i] - b[0] >= width:
            res.append(b)
            b = []
        b.append(sort[i])
        if i == len(sort)-1:
            res.append(b)
    return res

# test equal-width partitioning
print(concept_hierarchy_equal_width(age, 10))


# based on the equal-frequency partitioning rule
def concept_hierarchy_equal_freq(attribute:list, freq:int):
    res = []
    b = []
    count = 0
    sort = np.sort(attribute)
    for i in range(len(sort)):
        if count == freq or i == len(sort)-1:
            res.append(b)
            b = []
            count = 0
        b.append(i)
        count += 1
    return res

# test equal-width partitioning
print(concept_hierarchy_equal_freq(age, 5))