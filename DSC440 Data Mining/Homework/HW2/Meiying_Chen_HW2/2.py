#!/usr/bin/python3 
# -*- coding: utf-8 -*-

'''
@Vesion  :  1.0
@Time    :  09/22/2019, 00:10
@Author  :  Meiying(Melissa) Chen
@Contact :  meiying.chen@rochester.edu

'''

from scipy.spatial import distance
from sklearn import preprocessing
import numpy as np

# problem 2.6
x1 = [22, 1, 42, 10]
x2 = [20, 0, 36, 8]
euclidean = distance.euclidean(x1, x2)
manhattan = distance.cityblock(x1, x2)
minkowski = distance.minkowski(x1, x2,p = 3)
supremum  = distance.chebyshev(x1, x2)


# problem 2.8
x_exist = [[1.5, 1.7],
            [2, 1.9],
           [1.6, 1.8],
           [1.2, 1.5],
           [1.5, 1.0]]
x = [1.4, 1.6]

sim = np.zeros((5,4))
for i in range(0,5):
    sim[i, 0] = distance.euclidean(x_exist[i], x) 
    sim[i, 1] = distance.cityblock(x_exist[i], x) 
    sim[i, 2] = distance.chebyshev(x_exist[i], x) 
    sim[i, 3] = 1 - distance.cosine(x_exist[i], x) # cosine similarity

print(sim)

x = [[1.4, 1.6]]
x_norm = preprocessing.normalize(x, norm='l2', axis = 1) 
x_exist_norm = preprocessing.normalize(x_exist, norm='l2', axis = 1)
print(x_norm)
print(x_exist_norm)

sim_norm = np.zeros((5,1))
for i in range(0,5):
    sim_norm[i, 0] = distance.euclidean(x_exist_norm[i], x_norm) 
print(sim_norm)