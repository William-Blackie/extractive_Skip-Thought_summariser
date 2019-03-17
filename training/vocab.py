"""
Constructing and loading dictionaries

Modified: William Blackie
"""
import cPickle as pkl
import collections

import numpy
from collections import OrderedDict
import Utils.DataUtils
import pandas as pd


def read_file(path, block_size=1024):
    """
    reads file into more manageable chunks
    :param path: current path to file
    :param block_size: size to chunk file
    :return: chunked file
    """
    with open(path, 'rb') as f:
        while True:
            piece = f.read(block_size)
            if piece:
                yield piece
            else:
                return


def build_dictionary(directory):
    """
    creates dictionary of 20,000 most common words input
    :param directory: directory of pre-sanitised corpus
    :return: dictionary of 20,000 most common words
    """

    wordcount = OrderedDict()

    data_utils = Utils.DataUtils.DataUtils()
    directories, root = data_utils.get_data_directory(directory)

    for dir in directories:
        print(dir)
        for chunk in read_file(dir):
            chunk = str(chunk)
            words = chunk.split()
            for w in words:
                if w not in wordcount:
                    wordcount[w] = 0
                wordcount[w] += 1

    highest_freq_dict = dict(collections.Counter(wordcount).most_common(20000))
    wordcount.clear()

    words = highest_freq_dict.keys()
    freqs = highest_freq_dict.values()
    wordcount.clear()
    sorted_idx = numpy.argsort(freqs)[::-1]
    freqs = []

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx + 2  # 0: <eos>, 1: <unk>

    return worddict, wordcount


def load_dictionary(loc):
    """
    Load a dictionary
    """
    with open(loc, 'r') as f:
        worddict = pkl.load(f)
    return worddict


def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, 'w+') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)
