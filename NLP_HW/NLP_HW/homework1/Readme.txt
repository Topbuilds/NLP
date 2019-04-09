1. The homework assignment is developed under Anaconda3

2. Using Anaconda3 or ohter IDEs  to open NLP_UniAndBigrams.py, then run the file to get the results

3. The output of perplexity with different LMs and percents of unknow tokens/words in test files are on the console. 


For the running command parameters:
===============================================================================================================
brown-test.txt
./NLP_UniAndBigrams.py -n 1 brown-train.txt brown-test.txt                            #unigram module
./NLP_UniAndBigrams.py -n 2 brown-train.txt brown-test.txt                            #bigram module
./NLP_UniAndBigrams.py -ad 1 -n 2 brown-train.txt brown-test.txt                      #add-one smooth bigram module

---------------------------------------------------------------------------------------------------------------
learner-test.txt
./NLP_UniAndBigrams.py -n 1 brown-train.txt learner-test.txt                            #unigram module
./NLP_UniAndBigrams.py -n 2 brown-train.txt learner-test.txt                            #bigram module
./NLP_UniAndBigrams.py -ad 1 -n 2 brown-train.txt learner-test.txt                      #add-one smooth bigram module