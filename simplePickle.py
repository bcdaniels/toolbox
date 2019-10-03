# simplePickle.py
#
# Bryan Daniels
# 1.24.2012
#

import io
import sys

if sys.version_info.major > 2:
    # Python 3
    import pickle
else:
    # Python 2
    import cPickle as pickle

def save(obj,filename):
    fout = io.open(filename,'wb')
    # Note: we currently save using backward-compatible protocol 2
    pickle.dump(obj,fout,2)
    fout.close()

def load(filename):
    fin = io.open(filename,'rb')
    obj = pickle.load(fin)
    fin.close()
    return obj
