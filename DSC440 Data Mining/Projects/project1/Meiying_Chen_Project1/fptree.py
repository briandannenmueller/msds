#!/usr/bin/python3
#-*- coding: utf-8 -*-

'''
FP-tree nodes structure for fp-growth algorithm

@Version: vanilla
@Author: Meiying Chen
@Date: 2019-10-17 17:39:10
@LastEditTime: 2019-10-17 19:15:38
'''

# With much love~

class Node:
    def __init__(self, value, occurence, father, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value
        self.occurence = occurence
        self.father = father
        self.children = {}
        self.next_node = None
    
    def passby(self):
        self.occurence += 1
    
