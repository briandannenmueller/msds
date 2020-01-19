#!/usr/bin/python3
#-*- coding: utf-8 -*-

'''
FP-Growth Algorithm
for frequent set generation

@Version: vanilla
@Author: Meiying Chen
@Date: 2019-10-16 20:10:36
@LastEditTime: 2019-10-17 21:48:50
'''

# With much love~

####################################################################
# Overview:
#
# Part A: generate fp tree from raw data
#   step 1: scan all transactions
#       - calculate support for every item
#       - delete items whose support is under the threshold
#       - init header table
#   step 2: scan all transactions
#       - grow fp tree with every transaction
#       - update header table while growing
#
# Part B: generate frequent set from fp tree
#   step 1: get conditional pattern base(perfix path)
#   step 2: generate conditional fp-tree for every item
#   step 3: filter with threshold 
#
####################################################################


####################### Part A #####################################
from fptree import Node # see fptree.py
import math

# scan all transactions for the first time
# calculate support for every item
# and delete items whose support is under the threshold
# init header table
def init_itemset(trans:list, min_supp:float):
    itemset = {} # frequent item set, {itemvalue: occurence count}
    header = {} # {itemvalue: [support value, head pointer]}
    counts = {} # {itemvalue: support value}
    min_count = math.ceil(min_supp * len(trans))

    # count occurence for all items
    for tran in trans:
        for item in tran:
            if item not in counts:
                counts[item] = 1
            else:
                counts[item] += 1 

     # calcualte supports and generate the frequent itemset
    for item in counts:
        supp = counts[item]
        if supp >= min_count:
            itemset[item] = supp
            # init header table
            header[item] = [supp, None]

    return itemset, header


# sort transactions according to item support value
def sort_trans(trans:list, itemset:dict):
    trans_sorted = []
    
    for tran in trans:
        count = {}
        for i in tran:
            # if transaction contains non-frequent value
            if i not in itemset: break
            count[i] = itemset[i]
        sort = sorted(count.items(), key=lambda x: x[1], reverse = True)
        trans_sorted.append([x[0] for x in sort])
    
    return trans_sorted
        

# update tree
def update_tree(fptree, header, sorted_tran):
    curr_node = fptree
    i = 0

    while i <  len(sorted_tran):
        item = sorted_tran[i]
        if item in curr_node.children:
            curr_node.children[item].passby()
        else:
            curr_node.children[item] = Node(item, 1, curr_node)
        curr_node = curr_node.children[item]
        i += 1

        # link nodes with same values
        if header[item][1]:
            if curr_node.occurence == 1:
                curr_node.next_node = header[item][1]
        header[item][1] = curr_node 
        
    return fptree, header

    

# scan all transactions for the second time
# grow fp tree and update header
def grow_fptree(trans:list, min_supp:float):
    fptree = Node('φ', 1, None)

    itemset, header = init_itemset(trans, min_supp)
    trans_sorted = sort_trans(trans, itemset)
    for sorted_tran in trans_sorted:
       fptree, header = update_tree(fptree, header, sorted_tran) 

    return fptree, itemset, header



####################### Part B #####################################

#  get conditional pattern base(perfix path)
def gen_path(header_key, header):
    curr_item = header[header_key][1]
    path = []

    while curr_item is not None:
        p = []
        occu = []
        curr_node = curr_item
        # to the root node
        while curr_node.father is not None:
            p = p + [curr_node.value]
            occu = occu +[curr_node.occurence] 
            curr_node = curr_node.father
        if len(p[1:]) > 0:
            path.append((p[1:],min(occu)))
    
        curr_item = curr_item.next_node
        
    return path


#   generate conditional fp-tree for every item
#   filter with threshold
#   fp-growth main entrance
def fpgrowth(trans:list, min_supp:float):
    fptree, itemset, header = grow_fptree(trans, min_supp)
    paths = {}
    for item in header.keys():
        res = (gen_path(item, header))
        if len(res) > 0:
            paths[item] = res
    
    while len(paths) > 1:
        trans = []

        for i in paths:
            for tran in paths[i]:
                for j in range(tran[1]):
                    trans.append(tran[0])

        fptree, itemset, header = grow_fptree(trans, min_supp)
        paths = {}
        for item in header.keys():
            res = (gen_path(item, header))
            if len(res) > 0:
                paths[item] = res

    return paths



if __name__ == "__main__":
    trans = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['z'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

    frequent_set = fpgrowth(trans, min_supp=0.5)
    print(frequent_set)
    
    

    
    
