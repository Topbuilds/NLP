#!/usr/bin/env python
from __future__ import division
import argparse
import string
from math import log10, isnan
from collections import Counter, defaultdict


"""
Unitgram, Bigrams, and add one smoothing  perplexity, 
"""


class NLP_UniAndBigrams:
    def __init__(self, opts):
        self.threshold = 1  # freq of words considered unknown
        self.start_token = "<s> "
        self.end_token = " </s> "
        self.unk_token = "<unk>"
        self.unigrams = None
        self.bigrams = None  # Keep separate for potential backoff
        self.total_words = None  # Don't calc everytime, store and use later
        self.types = 0
        self.train_len = None
        self.alpha = opts.ad
        self.n = opts.n
        self.training_set = opts.training_set
        self.test_set = opts.test_set

    def init_lm_pro(self, n, typ):
        unk = 1
        tokens = self.parse_file(typ)
        
        if typ == 0:   # train model
            self.bigrams = None
            self.bi_ocm = None
        
            if n == 1:
                word_freq_pairs, total_words = self.uni_count_pairs(tokens, n, unk)
            elif n == 2:
                word_freq_pairs, total_words = self.bi_count_pairs(tokens, n, unk)
    
            self.types = len(total_words)
            self.total_words = total_words
            self.train_len = len(tokens)
            
            return string, word_freq_pairs, total_words
        else:
            
            return tokens

    def parse_file(self, typ):
        filename = self.test_set if typ % 2 else self.training_set

        with open(filename, 'r') as text:
            tokens = text.readlines()
            
        # add <s> and </s> in every sentence    
        tokens_str = ''
        
        for i in range(0, len(tokens)):
            tokens[i] = self.start_token + tokens[i] + self.end_token
            tokens[i] = tokens[i].lower()
            tokens_str += tokens[i]   # make all tokens in a string
        
        tokens = tokens_str.split()      
        return tokens

    def top_lvl_unk_tokenize(self, total_words, tokens):
        unk_words = set()
        itms = total_words.items()
        for word, count in itms:
            if count <= self.threshold:
                unk_words.add(word)
                total_words[self.unk_token] += count

        unk_words.discard(self.start_token)
        unk_words.discard(self.end_token)
        unk_words.discard(self.unk_token)

        if unk_words:
            for i in range(0, len(tokens)):
                if tokens[i] in unk_words:
                    tokens[i] = self.unk_token
                #else:
                 #   tokens = [word]
           # tokens = [self.unk_token if word in unk_words else word
            #          for word in tokens]
        return tokens, unk_words, total_words

    def bottom_unk_tokenize(self, word_freq_pairs, n):
        list_wrap = list 
        tmp_pairs = word_freq_pairs
        stack = [(tmp_pairs, n)]

        while stack:
            tmp_pairs, n = stack.pop()
            if n == 2:
                values = tmp_pairs.values()
                for nxt_lvl_dict in values:
                    for word, count in list_wrap(nxt_lvl_dict.items()):
                        if (count <= self.threshold and
                                word != self.unk_token):
                            del nxt_lvl_dict[word]
                            nxt_lvl_dict[self.unk_token] += count
            else:
                n -= 1
                for word in tmp_pairs:
                    stack.append((tmp_pairs[word], n))

        return word_freq_pairs

    """
    Get total counts, and word frequency dictionaries.
    """
    def uni_count_pairs(self, tokens, n, unk):
        total_words = Counter(tokens)
        #self.total_words[self.end_token] -= 1  # Last token won't have a bigram
        total_words[self.unk_token] = 0
        
        word_freq_pairs = dict.fromkeys(tokens, 0)
        word_freq_pairs[self.unk_token] = 0

        for token in tokens:
            word_freq_pairs[token] += 1

        if unk:
            unk_words = set()
            items = word_freq_pairs.items()
            for word, count in items:
                if count <= self.threshold:
                    unk_words.add(word)
                    word_freq_pairs[self.unk_token] += count

            unk_words.discard(self.start_token)
            unk_words.discard(self.end_token)
            unk_words.discard(self.unk_token)

            for word in unk_words:
                del word_freq_pairs[word]

        return word_freq_pairs, total_words
    
    def bi_count_pairs(self, tokens, n, unk):
        list_wrap = list
        start_token = self.start_token
        end_token = self.end_token
        unk_token = self.unk_token
        thresh = self.threshold
            
        total_words = Counter(tokens)
        
        #self.total_words[self.end_token] -= 1  # Last token won't have a bigram
        total_words[self.unk_token] = 0

        if unk:
            tokens, unks, total_words = self.top_lvl_unk_tokenize(total_words, tokens)
            for word in unks:
                del total_words[word]

        word_freq_pairs = {word: defaultdict(int) for word in total_words}
        for i in range(0, len(tokens)-1):
            word_freq_pairs[tokens[i]][tokens[i+1]] += 1

        return word_freq_pairs, total_words

    """
    Computes MLE probability distributions.
    """
    def unsmoothed_unigrams(self, word_freq_pairs):
        prob_dict = word_freq_pairs
        items = prob_dict.items()
        for word, count in items:
            prob_dict[word] = count / self.train_len

        self.unigrams = prob_dict

    def unsmoothed_bigrams(self, word_freq_pairs):
        prob_dict = word_freq_pairs
        items = prob_dict.items()
        for word, nxt_lvl_dict in items:
            nxt_lvl_items = nxt_lvl_dict.items()
            for word_infront, cnt in nxt_lvl_items:
                nxt_lvl_dict[word_infront] = cnt / self.total_words[word]

        self.bigrams = prob_dict

 
    """Computes additive smoothed probability distributions.
    """
    def additive_unigrams(self, word_freq_pairs):
        prob_dict = word_freq_pairs
        items = prob_dict.items()
        N = sum(self.total_words.values())
        V = self.types
        for word, count in items:
            prob_dict[word] = (count+self.alpha) / (N+V)

        self.unigrams = prob_dict

    def additive_bigrams(self, word_freq_pairs, n):
        alpha = self.alpha
        V = self.types
        
        prob_dict = word_freq_pairs
        items = prob_dict.items()
                    
        for top_word, nxt_lvl_dict in items:
            nxt_lvl_items = nxt_lvl_dict.items()
                    
            add_one_demonitor = V + self.total_words[top_word]
                    
            for bot_word, cnt in nxt_lvl_items:
                nxt_lvl_dict[bot_word] = ((cnt+alpha) /
                                                 add_one_demonitor)
 
        self.bigrams = prob_dict

    """
    Generates sentences based on probability distributions.
    """
    def uni_perplex(self, tokens):
   
        unigrams = self.unigrams
            
        entropy = 0.0

        V = self.types
        alpha = self.alpha

        #if not found, treat as unknown token
        uk_p = unigrams.get(self.unk_token)
        unk_tokens_test = 0
        unk_V = set()
        total_words = Counter(tokens)
        
        for token in tokens:
            if unigrams.get(token) is None:
             
                entropy -= log10(uk_p) # unseen data, then unk_token probability
                unk_tokens_test += 1
                unk_V.add(token)
            else:
                entropy -= log10(unigrams.get(token)) 
                
        print("unitgram percentage of unknow tokens in test:" + str(unk_tokens_test/len(tokens)))
        print("unitgram percentage of unknow word types in test:" + str(len(unk_V)/len(total_words)))
        
        return 10**(entropy / len(tokens))

    def bi_perplex(self, tokens, ad):

        tw = self.total_words
        bigrams = self.bigrams

        ut = self.unk_token
        entropy = 0.0
        prev_t = tokens.pop(0)

        V = self.types
        alpha = self.alpha
        #uk_p = bigrams.get(self.unk_token)
        
        uk_p = bigrams.get(self.unk_token)
        unk_tokens_test = 0
        unk_V = set()
        total_words = Counter(tokens)
        
        for token in tokens:
            if token == '<s>':
                # don't need to calculate <s> symbol probability
                prev_t = token
                continue
            
            if prev_t in bigrams:
                
                if bigrams[prev_t].get(token) is None:
                    #if  bigrams[prev_t].get(ut, 0) == 0:
                    if(ad == 1):
                        if bigrams[prev_t].get(ut) is None:                            
                            if bigrams[ut].get(ut) is None:
                                # probability nearly 0
                                entropy -= log10(1/9999999999)
                            else:
                                # probability of unknow unknow
                                entropy -= log10(bigrams[ut].get(ut))
                        else:
                            entropy -= log10(bigrams[prev_t].get(ut))
                    else:
                        #unseen bigrams so perplexity is undefined
                        entropy = None
                        
                    unk_tokens_test += 1
                    unk_V.add(token)
                else:
                    if (entropy is not None):
                        entropy -= log10(bigrams[prev_t].get(token))
            else:
                
                if bigrams[ut].get(token) is None:
                    if(ad == 1):
                        if bigrams[ut].get(ut) is None:
                            # probability nearly 0
                            entropy -= log10(1/9999999999)
                        else:
                            # probability of unknow unknow
                            entropy -= log10(bigrams[ut].get(ut))
                    else:
                        #unseen bigrams so perplexity is undefined
                        entropy = None                     
                    
                    unk_tokens_test += 1
                    unk_V.add(token)
                else:
                    if (entropy is not None):
                        entropy -= log10(bigrams[ut].get(token))

            prev_t = token    

        print("bigrams percentage of unknow tokens in test:" + str(unk_tokens_test/len(tokens)))
        print("bigrams percentage of unknow word types in test:" + str(len(unk_V)/len(total_words)))
        
        if (entropy is not None):
            return 10**(entropy / len(tokens))
        else:
            return "Undefined"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=2,
                        help="Size of the probability tuples. N should be < ")

    parser.add_argument("-ad", action="store", nargs='?',
                                 type=float, default=0, metavar="\u03B1",
                                 const=1, help="Use additive (Additive) "
                                               "smoothing; plus-one is the "
                                               "default (\u03B1 = 1).")

    parser.add_argument("training_set", action="store",
                        help="Must be at end of command.")
    parser.add_argument("test_set", action="store", nargs='?',
                        help="Must be at end of command.")

    parser.usage = ("NLP_UniAndBigrams.py [-n 1 or 2]\n "
                    "[-ad 1] \n"
                    "<training_set test_set>")

    error_str = ""
    opts = parser.parse_args()
     
    if opts.n < 1:
        if error_str:
            error_str += "                  "
        error_str += "argument -n: invalid int value: N must be >= 1.\n"
    
    
    if isnan(opts.ad) or (opts.ad == float('inf')):
        if error_str:
            error_str += "                  "
        error_str += ("argument -sa: invalid int value: \u03B1 cannot "
                      "be NaN and cannot be inf (\u221E).\n")

    if error_str:
        parser.error(error_str[:-1])
    return opts


def finish_model(model, n, word_freq_pairs, ad):

    if n == 1:
        if ad:
            model.additive_unigrams(word_freq_pairs)
        else:     
            model.unsmoothed_unigrams(word_freq_pairs)
    else:
        if ad:
            model.additive_bigrams(word_freq_pairs, n)
        else:
            model.unsmoothed_bigrams(word_freq_pairs)


def main():
    opts = parse_args()
    n = opts.n

    train_str = test_str = None
    
    model = NLP_UniAndBigrams(opts)


    train_str, word_freq_pairs, total_words = model.init_lm_pro(n, 0)
    finish_model(model, n, word_freq_pairs, opts.ad)
        
    test_t = model.init_lm_pro(n, 1)
    if n == 1:
        perplexity = model.uni_perplex(test_t)
    elif n == 2:
        perplexity = model.bi_perplex(test_t, opts.ad)
  
    if opts.ad is 0:
        print("Perplexity: " + str(perplexity))    
    else:
        print("Perplexity(add-one smoothing)" + str(perplexity)) 

if __name__ == '__main__':
    main()
