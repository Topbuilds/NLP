# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:32:56 2017

"""
from __future__ import division
from collections import defaultdict
import sys
class NaiveBayes_Classifi:
    def __init__(self):
        self.data = []
        self.all_cl = []
        self.cl = None
        self.tokens = []
        self.V = None
        self.size_v = 0
        self.cl_par_pairs = None
        self.pri = None
        self.python2 = sys.version_info < (3,)

    def parse(self, fileName):     
        # training data
        with open(fileName, 'r') as text:
            tokens = text.readlines()
        
        range_wrap = xrange if self.python2 else range
        for i in range_wrap(0, len(tokens)):
            line_str = tokens[i].split()
            tp = (line_str[-1], line_str[:-1])             
            self.data.append(tp)  
            
        # get all classifications from training data
        for tp in self.data:
            self.all_cl.append(tp[0])
            
        # count num of every classification
        self.cl = dict.fromkeys(self.all_cl, 0)
        for i in self.all_cl:
            self.cl[i] += 1
   
        # tokens N 
        for tp in self.data:
            for s in tp[1]:
                self.tokens.append(s)
        
        
        # vocabulary V
        self.V = dict.fromkeys(self.tokens, 0)
        self.size_v = len(self.V)
        
        # parameters of every classification
        self.cl_par_pairs = {c: defaultdict(int) for c in self.cl}
        
        # init
        for c in self.cl_par_pairs:
            for w in self.V:
                self.cl_par_pairs[c][w] = 0
        
        # count                    
        for tp in self.data:
            words = dict.fromkeys(tp[1])
            for s in words:
                self.cl_par_pairs[tp[0]][s] += 1
                                 
        return
                        
    def train_module(self):
        # prior predict value of add-one smooth
        self.pri = dict.fromkeys(self.all_cl, 0)
        for i in self.cl:
            self.pri[i] = (self.cl[i] + 1)/ (len(self.all_cl) + len(self.cl))        
        
        # calculate parameters' likelihood by add-one smooth
        # in this case, add one more dataset including all vocabulary for each class
        for c in self.cl_par_pairs:
            for word in self.cl_par_pairs[c]:            
                self.cl_par_pairs[c][word] = (self.cl_par_pairs[c][word] + 1) / (self.cl[c] + 1)  
                 
    
    
        # file-output.py
        f = open('movie-review.NB','w')        
        for c in self.cl_par_pairs:
            f.write('\n' + str(c) + ' class parameters:\n')
            s = 'Prior of this class: ' + str(self.pri[c]) + '\n' + 'word parameters list: \n'
            f.write(s)
            for word in self.cl_par_pairs[c]: 
                s = word + ': ' + str(self.cl_par_pairs[c][word]) + '\n'
                f.write(s)
                
        f.close()
        
        return
    
    def predict(self, fileName):
        # test data
        with open(fileName, 'r') as text:
            test_tokens = text.readlines()
        
        # store the test data in list
        test_data = []
        range_wrap = xrange if self.python2 else range
        for i in range_wrap(0, len(test_tokens)):
            test_data.append(test_tokens[i].split())           
        
        # calculate the probability of each class for every line of test data
        for l in test_data:
            # get the probability value of each class
            pred_t = dict.fromkeys(self.cl, 0)
            for c in pred_t:
                pred_t[c] = self.pri[c]
                # calculate the probability by multiple prior and parameters from model
                for w in l:
                    pred_t[c] = pred_t[c]*self.cl_par_pairs[c][w]  
            
            # print the probability of each class for current test data line
            for c in pred_t:
                print('probability of ' + str(c) + ': ' + str(pred_t[c]))   
                
            # predict the class has the maximum probability from all class 
            values = (pred_t.itervalues() if self.python2 else
                          pred_t.values())
            max_value = max(values)  # maximum value
            
            items = pred_t.iteritems() if self.python2 else pred_t.items()
            max_c = [c for c, v in items if v == max_value] # getting all keys containing the `maximum`                    
            print('predict class is: ' + str(max_c))   
            
        return
    
def main():
    ###################
    # parse file
    ###################
     # create a NaiveBayesian Classification object
    model = NaiveBayes_Classifi()
    
    # parse the training data
    model.parse('train_data.txt')
            
    ###################
    # Train model
    ###################
    model.train_module()
    
    ###################
    # Test prediction
    ###################                
    model.predict('test_data.txt')
    
    return

if __name__ == '__main__':
    main()
