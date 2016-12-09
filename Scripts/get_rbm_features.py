import sys

sys.path.append("/home/piero/Documents/Softwares/pdnn")

import numpy as np
from io_func.model_io import _file2nnet, log
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from models.dnn import DNN
from models.srbm import SRBM
from utils.learn_rates import _lrate2file, _file2lrate
from utils.network_config import NetworkConfig
from utils.rbm_config import RBMConfig
from utils.utils import parse_arguments
import cPickle
import os
from sklearn.neighbors import NearestNeighbors as KNN
import math

# check the arguments
arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
arguments = parse_arguments(arg_elements)
required_arguments = ['train_data', 'wdir']
for arg in required_arguments:
    if arguments.has_key(arg) == False:
        print "Error: the argument %s has to be specified" % (arg); exit(1)

train_data_spec = arguments['train_data']
wdir = arguments['wdir']

path = "/home/piero/Documents/Experiments/Real_Test/Spectral Coef/Noise vs BG_voice+Conversation vs Shout+Scream/fft coef/"
os.chdir(path)
filename = "rbm.cfg"
train_data = "train.pickle.gz"
test_data = "test.pickle.gz"
batch_size = 128

log('> ... setting up the model and loading parameters')
numpy_rng = np.random.RandomState(89677)
theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
cfg_dnn = cPickle.load(open(filename,'r'))
cfg_dnn.init_activation()
model = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg_dnn)

# load model parameters
_file2nnet(model.layers, filename = wdir + '/rbm.param')

# initialize data reading
cfg_dnn.init_data_reading_test(train_data_spec)

# get the function for feature extraction
log('> ... getting the feat-extraction function')
extract_func = model.build_extract_feat_function(-1)

output_mat = None  # store the features for all the data in memory
log('> ... generating features from the specified layer')
while (not cfg_dnn.test_sets.is_finish()):  # loop over the data
    cfg_dnn.test_sets.load_next_partition(cfg_dnn.test_xy)
    batch_num = int(math.ceil(cfg_dnn.test_sets.cur_frame_num / batch_size))

for batch_index in xrange(batch_num):  # loop over mini-batches
    start_index = batch_index * batch_size
    end_index = min((batch_index+1) * batch_size, cfg_dnn.test_sets.cur_frame_num)  # the residue may be smaller than a mini-batch
    output = extract_func(cfg_dnn.test_x.get_value()[start_index:end_index])
    if output_mat is None:
        output_mat = output
    else:
        output_mat = np.concatenate((output_mat, output)) # this is not efficient

log('> ... fitting a KNN cluster')
knn = KNN(n_neighbors=3)
knn.fit(output_mat)
log('> ... computing the graph of class')
results = knn.kneighbors_graph(output_mat)
print(results)
# results.toarray()
# print(results.toarray())
