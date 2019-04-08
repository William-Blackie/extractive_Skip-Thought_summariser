"""
A selection of functions for extracting vectors
Encoder + vocab expansion

Modified: William Blackie

Reason: PEP8 formatting, refactoring name of load_googlenews for clarity, refactoring path and name of vectors to FastText.
Reason: Replace dict with dict literal

TODO need to modify for Fast-text + some naming conventions for clarity...
"""
import gensim
import theano
import theano.tensor as tensor
from gensim.models import KeyedVectors

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import warnings
import cPickle as Pkl
import numpy
import nltk

from collections import OrderedDict, defaultdict
from nltk.tokenize import word_tokenize
from scipy.linalg import norm

warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='gensim')  # Remove genism warnings for windows as they are irelevent
from sklearn.linear_model import LinearRegression

from utils import load_params, init_tparams
from model import init_params, build_encoder, build_encoder_w2v


# -----------------------------------------------------------------------------#
# Specify model and dictionary locations here
# -----------------------------------------------------------------------------#
path_to_model = r'D:\Projects\skip-thoughts\models\my_bi_skip.npz'
path_to_dictionary = r'D:\Projects\skip-thoughts\models\corpus\saved_dict.pk1'
path_to_fasttext = r'D:\Projects\skip-thoughts\models\vectors\wiki.en.vec'


# -----------------------------------------------------------------------------#
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def load_model(embed_map=None):
    """
    Load all model components + apply vocab expansion

    Modified: to load FastText vectors
    """

    # Load the worddict
    print 'Loading dictionary...'
    with open(path_to_dictionary, 'r') as f:
        worddict = Pkl.load(f)

    # Create inverted dictionary
    print 'Creating inverted dictionary...'
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # Load model options
    print 'Loading model options...'
    with open('%s.pkl' % path_to_model, 'rb') as f:
        options = Pkl.load(f)

    # Load parameters
    print 'Loading model parameters...'
    params = init_params(options)
    params = load_params(path_to_model, params)
    tparams = init_tparams(params)

    # Extractor functions
    print 'Compiling encoder...'
    trng = RandomStreams(1234)
    trng, x, x_mask, ctx, emb = build_encoder(tparams, options)
    f_enc = theano.function([x, x_mask], ctx, name='f_enc')
    f_emb = theano.function([x], emb, name='f_emb')
    trng, embedding, x_mask, ctxw2v = build_encoder_w2v(tparams, options)
    f_w2v = theano.function([embedding, x_mask], ctxw2v, name='f_w2v')

    # Load word2vec, if applicable
    if embed_map == None:
        print 'Loading FastText embeddings...'
        embed_map = load_fasttext_vectors(path_to_fasttext)

    # Lookup table using vocab expansion trick
    print 'Creating word lookup tables...'
    table = lookup_table(options, embed_map, worddict, word_idict, f_emb)

    # Store everything we need in a dictionary
    print 'Packing up...'
    model = {'options': options, 'table': table, 'f_w2v': f_w2v}

    return model


def encode(model, X, use_norm=True, verbose=True, batch_size=128, use_eos=False):
    """
    Encode sentences in the list X. Each entry will return a vector
    """
    # first, do preprocessing
    X = preprocess(X)

    # word dictionary and init
    d = defaultdict(lambda: 0)
    for w in model['table'].keys():
        d[w] = 1
    features = numpy.zeros((len(X), model['options']['dim']), dtype='float32')

    # length dictionary
    ds = defaultdict(list)
    captions = [s.split() for s in X]
    for i, s in enumerate(captions):
        ds[len(s)].append(i)

    # Get features. This encodes by length, in order to avoid wasting computation
    for k in ds.keys():
        if verbose:
            print k
        numbatches = len(ds[k]) / batch_size + 1
        for minibatch in range(numbatches):
            caps = ds[k][minibatch::numbatches]

            if use_eos:
                embedding = numpy.zeros((k + 1, len(caps), model['options']['dim_word']), dtype='float32')
            else:
                embedding = numpy.zeros((k, len(caps), model['options']['dim_word']), dtype='float32')
            for ind, c in enumerate(caps):
                caption = captions[c]
                for j in range(len(caption)):
                    if d[caption[j]] > 0:
                        embedding[j, ind] = model['table'][caption[j]]
                    else:
                        embedding[j, ind] = model['table']['UNK']
                if use_eos:
                    embedding[-1, ind] = model['table']['<eos>']
            if use_eos:
                ff = model['f_w2v'](embedding, numpy.ones((len(caption) + 1, len(caps)), dtype='float32'))
            else:
                ff = model['f_w2v'](embedding, numpy.ones((len(caption), len(caps)), dtype='float32'))
            if use_norm:
                for j in range(len(ff)):
                    ff[j] /= norm(ff[j])
            for ind, c in enumerate(caps):
                features[c] = ff[ind]

    return features


def preprocess(text):
    """
    Preprocess text for encoder
    """
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for t in text:
        sents = sent_detector.tokenize(t)
        result = ''
        for s in sents:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    return X


def load_fasttext_vectors():
    """
    load the word2vec fasttext vectors
    """

   # fast_text_model = KeyedVectors.load_word2vec_format(path_to_word2vec, binary=False)
    embed_map = KeyedVectors.load_word2vec_format(path_to_fasttext, binary=False)
    return embed_map


def lookup_table(options, embed_map, worddict, word_idict, f_emb, use_norm=False):
    """
    Create a lookup table from linear mapping of word2vec into RNN word space
    """
    wordvecs = get_embeddings(options, word_idict, f_emb)
    clf = train_regressor(options, embed_map, wordvecs, worddict)
    table = apply_regressor(clf, embed_map, use_norm=use_norm)

    for i in range(options['n_words']):
        w = word_idict[i]
        table[w] = wordvecs[w]
        if use_norm:
            table[w] /= norm(table[w])
    return table


def get_embeddings(options, word_idict, f_emb, use_norm=False):
    """
    Extract the RNN embeddings from the model
    """
    d = OrderedDict()
    for i in range(options['n_words']):
        caption = [i]
        ff = f_emb(numpy.array(caption).reshape(1, 1)).flatten()
        if use_norm:
            ff /= norm(ff)
        d[word_idict[i]] = ff
    return d


def train_regressor(options, embed_map, wordvecs, worddict):
    """
    Return regressor to map word2vec to RNN word space

    Modified: Changed to accept FastText object
    """
    # Gather all words from word2vec that appear in wordvecs
    d = defaultdict(lambda: 0)
    for w in embed_map.vocab.keys():
        d[w] = 1
    shared = OrderedDict()
    count = 0
    for w in worddict.keys()[:options['n_words'] - 2]:
        if d[w] > 0:
            shared[w] = count
            count += 1

    # Get the vectors for all words in 'shared'
    w2v = numpy.zeros((len(shared), 300), dtype='float32')
    sg = numpy.zeros((len(shared), options['dim_word']), dtype='float32')
    for w in shared.keys():
        w2v[shared[w]] = embed_map[w]
        sg[shared[w]] = wordvecs[w]

    clf = LinearRegression()
    clf.fit(w2v, sg)
    return clf


def apply_regressor(clf, embed_map, use_norm=False):
    """
    Map words from word2vec into RNN word space
    """
    wordvecs = OrderedDict()
    embed_map = numpy.array(embed_map.vocab.keys()).reshape(1, -1) # to try after thi
    for i, w in enumerate(embed_map):
        if '_' not in w:
            wordvecs[w] = clf.predict(embed_map[w]).astype('float32')
            if use_norm:
                wordvecs[w] /= norm(wordvecs[w])
    return wordvecs
