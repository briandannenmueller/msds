#!/usr/bin/python3
#-*- coding: utf-8 -*-

'''
Apriori and FP-Growth algorithm using external library MLxtend
This file is for testing purpose only,
my original implement please see apriori.py and fp_growth.py

@Version: MLxtend
@Author: Meiying Chen
@Date: 2019-10-16 20:33:18
@LastEditTime: 2019-10-17 10:40:22
'''

# With much love~

import pandas as pd
from mlxtend import preprocessing, frequent_patterns

def apriori(trans:list, min_supp:float, min_conf:float, rules=False):
    
    te = preprocessing.TransactionEncoder()
    te_ary = te.fit(trans).transform(trans)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = frequent_patterns.apriori(df, min_support=min_supp, use_colnames=True)
    if rules:
        rules = frequent_patterns.association_rules(frequent_itemsets, min_threshold=min_conf)
    else: 
        rules = ['rules not required']
        
    return frequent_itemsets, rules


def fp_growth(trans:list, min_supp:float, max_len=None):
    
    te = preprocessing.TransactionEncoder()
    te_ary = te.fit(trans).transform(trans)
    df = pd.DataFrame(te_ary, columns=te.columns_) 
    frequent_itemsets = frequent_patterns.fpgrowth(df, min_support=min_supp, use_colnames=True, max_len=max_len)

    return frequent_itemsets