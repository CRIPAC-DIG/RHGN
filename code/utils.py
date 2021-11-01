import gzip
import hickle
import _pickle as cPickle

def load_obj(filename, serializer=cPickle):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin,encoding='iso-8859-1')  #,encoding='iso-8859-1'
    return obj

def dump_obj(obj, filename, protocol=-1, serializer=cPickle):
    if serializer == hickle:
        serializer.dump(obj, filename, mode='w', compression='gzip')
    else:
        with gzip.open(filename, 'wb') as fout:
            serializer.dump(obj, fout, protocol)