from training import tools
import cPickle as pkl

"""
Author: William Blackie

Script for loading FastText vectors into trained vectors and saving the expanded vectors.

"""
print 'load model...',
embed_map = tools.load_fasttext_vectors()
model = tools.load_model(embed_map)
print 'Vectors expanded'

print 'Saving model...',
pkl.dump(model, open('%s.pkl' % r'D:\Projects\skip-thoughts\models\new_uni_skip.npz', 'wb'))
print 'Model saved'