#!/usr/bin/python3
#-*- coding: utf-8 -*-

'''
Apriori algorithm 
for frequent set detection and pattern generation

@Version: vanilla & improvement with transaction reduction  
@Author: Meiying Chen
@Date: 2019-10-16 00:47:32
@LastEditTime: 2019-10-17 17:25:16
'''

# With much love~

########################################################################################
# Overview:
# Step 1: Generate items set C1
# Step 2: Generate frequenct set L1 from C1
# Step 3: Generate new possible frequent set Ck(k > 1) from L(k-1)
# Step 4: Iteratively generate Ck and Lk (k > 1) util no further combination can be made
# Step 5: Generate association rules from Lk
# Extra step: Improvement with transaction reduction 
########################################################################################


# generate item set c1
def gen_c1(trans:list):
    c1 = set()

    for tran in trans:
        for item in tran:
            c1.add(frozenset([item]))

    return c1


# generate frequent set l
def gen_l(trans:list, c, min_supp:float):
    l = set()
    counts = {}
    supps = {}
    ntrans = float(len(trans))

    # counting occurrence
    for tran in trans:
        for item in c:            
            # if transaction contains current item set
            if item.issubset(tran): 
               if item in counts: 
                counts[item] += 1
               else:
                counts[item] = 1
    
    # calcualte supports
    for item in counts: # ignore zero count item sets
        supp = counts[item] / ntrans
        if supp - min_supp >= 0.0:
            l.add(item)
            supps[item] = supp # ignore non-frequent item sets

    return l, supps


# generate possible frequent set ck from l(k-1), where k > 1
def gen_ck(l_prev, k:int):
    ck = set()
    length = len(l_prev)
    l_list = [list(x) for x in l_prev] #convinient to iterate)

    for i1 in range(0,length-1):
        for i2 in range(i1+1,length):
            for j in l_list[i2]:
                new_item = l_list[i1] + [j]
                new_set = frozenset(new_item)
                if len(new_set) == k:
                    ck.add(new_set)
    return ck



# iteratively generate ck and lk, where k > 1
def apriori(trans:list, min_supp:float, max_len = float('inf')):
    l = []
    supps = {}
    k = 2

    c1 = gen_c1(trans)
    l1, supp1 = gen_l(trans, c1, min_supp)
    l.append(l1)
    supps.update(supp1)

    
    while k - max_len <= 0:
        ck = gen_ck(l[k-2], k)
        lk, suppk = gen_l(trans, ck, min_supp)

        if lk:
            l.append(lk)
            supps.update(suppk)
            k += 1
        else:
            break

    return l, supps


# generate association rules
def gen_rules(l, supps, min_conf:float):
    rules = []
    length = len(l)

    for k in range(length - 1):
        for freq_set in l[k]:
            for sub_set in l[k + 1]:
                if freq_set.issubset(sub_set):
                    conf = supps[sub_set] / supps[freq_set]
                    rule = [freq_set, sub_set - freq_set, conf]
                    if conf >= min_conf:
                        rules.append(rule)

    return rules



#########################################  Transaction Reduction  #########################################
# reducing the number of transactions scanned in future itera- tions)
# A transaction that does not contain any frequent k-itemsets cannot contain any frequent (k + 1)-itemsets. 
# Therefore, such a transaction can be marked or removed from further consideration.

# apriori algorithm with transaction reduction
def apriori_tr(trans:list, min_supp:float, max_len = float('inf')):
    l = []
    supps = {}
    k = 2
    # keep a list to record if there a transaction contains any frequent set
    # if not, the transaction location is coded to 0, and do not be evaluated any more
    has_freq = [1] * len(trans) 

    c1 = gen_c1(trans)
    l1, supp1, has_freq = gen_l_tr(trans, has_freq, c1, min_supp)
    l.append(l1)
    supps.update(supp1)

    
    while k - max_len <= 0:
        ck = gen_ck(l[k-2], k)
        lk, suppk, has_freq = gen_l_tr(trans,has_freq, ck, min_supp)

        if lk:
            l.append(lk)
            supps.update(suppk)
            k += 1
        else:
            break

    return l, supps



# generate frequent set l with Transaction reduction technique
def gen_l_tr(trans:list, has_freq:list, c, min_supp:float):
    l = set()
    counts = {}
    supps = {}
    ntrans = len(trans)
    # maintain a list to record if a transaction contains frequent set
    has_freq_curr = [0] * ntrans 
    
    # counting occurrence
    for i in range(ntrans):
        if has_freq[i]: # ingore transactions contains no frequent set
            for item in c:            
                if item.issubset(trans[i]): 
                    has_freq_curr[i] = 1
                    if item in counts: 
                        counts[item] += 1
                    else:
                        counts[item] = 1
    
    # calcualte supports
    for item in counts: # ignore zero count item sets
        supp = counts[item] / float(ntrans)
        if supp - min_supp >= 0.0:
            l.add(item)
            supps[item] = supp # ignore non-frequent item sets

    return l, supps, has_freq_curr


