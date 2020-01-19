#!/usr/bin/python3
#-*- coding: utf-8 -*-

'''
Load data for testing apriori and fp-growth

@Version: vanilla
@Author: Meiying Chen
@Date: 2019-10-16 20:09:33
@LastEditTime: 2019-10-17 17:28:14
'''

# With much love~


import csv

class Dataloader:
    def __init__(self, file_path = '', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
    
    
    def apriori_toy_example(self):
        trans = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
        return trans


    def fp_toy_example(self):
        trans = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
        return trans


    def UCI_adult(self):
        try:
            with open(file = self.file_path) as f:
                f_csv = csv.reader(f)
                trans = []
                for row in f_csv:
                    if row:
                        trans.append(row)
        except IOError:
            print('Check your file path')
        return trans
    
    
