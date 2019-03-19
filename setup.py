import os
import nltk as nltk
import Utils.DataUtils
import training.train
from training.vocab import save_dictionary, build_dictionary

"""
Author: William Blackie

Script to create a new corpus and dictionary; then train new model from new vectors.
"""

current_corpus_dir = r'D:\Projects\skip-thoughts\text\corpus\corpus42.txt'

print "Checking for corpus...",
if len(os.listdir('text/corpus')) == 0:  # Create new corpus from WikiExtractor data
    print " Creating corpus...",
    dataUtils = Utils.DataUtils.DataUtils()
    dataUtils.create_corpus('text/new')
print "Done"

print "Checking for dictionary...",
if len(os.listdir(r'D:\Projects\skip-thoughts\models\corpus')) == 1:  # Create new dictionary from cleaned data
    dictionary, word_frequency = build_dictionary(r'D:\Projects\skip-thoughts\text\corpus')
    save_dictionary(dictionary, word_frequency, 'models/corpus/new_dict.pk1')
    print "created...",
print "Done"


print "Loading sentences...",
with open(current_corpus_dir, 'r') as temp: 
	# Training created vectors on selected corpus
    x = []
    x = nltk.tokenize.sent_tokenize(temp.read().decode("ascii", errors='ignore'), 'english')
    print len(x)
    print "done"
training.train.trainer(x)
