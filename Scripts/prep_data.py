
import sys
import cPickle
import gzip

if __name__ == '__main__':

#    f = gzip.open('/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/wav/1421675810481/1421675810481.pickle', 'rb')
    f_train = gzip.open('/home/piero/Documents/Speech_databases/test/pickle_data/train.pickle', 'rb')        
    f_valid = gzip.open('/home/piero/Documents/Speech_databases/test/pickle_data/valid.pickle', 'rb')        
    train_set = cPickle.load(f_train)
    valid_set = cPickle.load(f_valid)
#    test_set = cPickle.load(f)
    cPickle.dump(train_set, gzip.open('/home/piero/Documents/Speech_databases/test/train.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(valid_set, gzip.open('/home/piero/Documents/Speech_databases/test/valid.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)
#    cPickle.dump(test_set, gzip.open('test.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)

