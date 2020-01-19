#!/usr/bin/python3
#-*- coding: utf-8 -*-

'''
Main entrance

@Version: vanilla
@Author: Meiying Chen
@Date: 2019-10-16 20:11:16
@LastEditTime: 2019-10-17 23:55:16
'''

# With much love~

import apriori, fp_growth, myMLxtend
from dataloader import Dataloader
import time

if __name__ == "__main__":
    # load data
    trans = Dataloader().apriori_toy_example()
    trans = Dataloader().fp_toy_example()
    trans = Dataloader(file_path='./dataset/adult.data').UCI_adult()
    print('#transactions =',len(trans))

    t1 = time.time()

    # set threshold
    min_supp = 0.3
    min_conf = 0.7
    
    # aprioi, vallina implement
    L, supps = apriori.apriori(trans, min_supp)
    rules = apriori.gen_rules(L, supps, min_conf)
    print(L)
    print(rules)

    # aprioi, improvement using transaction reduction
    L, supps = apriori.apriori_tr(trans, min_supp)
    rules = apriori.gen_rules(L, supps, min_conf)
    print(L)
    print(rules) 

    # fp-growth
    frequent_set = fp_growth.fpgrowth(trans, min_supp=0.5)
    print(frequent_set)

    
    # testing my apriori result with MLxtend.apriori()
    L, rules = myMLxtend.apriori(trans, min_supp, min_conf, rules=False)
    print(L)
    print(rules)

  

    # testing my fp-growth result with MLxtend.fpgrowth() 
    L = myMLxtend.fp_growth(trans, min_supp, max_len=None)
    print(L)

  
    
    