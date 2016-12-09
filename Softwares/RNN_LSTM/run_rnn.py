from RNN import RNN
import numpy as np
import sys, os, gzip
import cPickle
from util_func import parse_arguments, log
from liblatex import *


def parse_classes(class_str,N):
    class_list = []
    start = 1
    i=0
    for j in range(N):
        while class_str[i] not in [',', '}']:
            i += 1
        end = i
        class_list.append(class_str[start:end])
        start = i+1
        i += 1
    return class_list

# ############################### Read Data ###############################
def read(data):
    if data.endswith('.gz'):
        fopen = gzip.open(data, 'rb')
    else:
        fopen = open(data, 'rb')
    [input, label, mask] = cPickle.load(fopen)
    return input, label, mask

# ########################### SET SOME VARIABLES #############################
arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
arguments = parse_arguments(arg_elements)
Nlayers = int(arguments['nlayers'])
Ndirs = int(arguments['ndir'])
Nx = int(arguments['nx'])
Nh = int(arguments['nh'])
Ah = arguments['ah']
Ay = arguments['ay']
predictPer = arguments['predict']
loss = arguments['loss']
L1reg = float(arguments['l1'])
L2reg = float(arguments['l2'])
momentum = float(arguments['momentum'])
frontEnd = arguments['frontEnd']
if frontEnd == 'None':
    frontEnd = None
filename = arguments['filename']
if filename == 'None':
    filename = None
initParams = arguments['initparams']
if initParams == 'None':
    initParams = None
filesave = arguments['filesave']
fileTex = arguments['fileTex']
classes= arguments['classes']
n_epoch = int(arguments['epoch'])
lrate = float(arguments['lambda'])
train_data_spec = arguments['train_data']
valid_data_spec = arguments['valid_data']
test_data_spec = arguments['test_data']
input, label, mask = read(train_data_spec)
test, label_test, mask_test = read(test_data_spec)
valid, label_valid, mask_valid = read(valid_data_spec)
n_streams = np.shape(label)[1]
len_batch = np.shape(label)[2]
n_classes = np.shape(label)[3]
n_batches = len(input)
n_test = len(test)
n_valid = len(valid)
error = np.zeros((n_test, n_streams, len_batch, n_classes))

# ############################ Create instance of class RNN #########################
var = np.random.RandomState()
seed = var.randint(90000)
if os.path.exists(filesave):
    filename = filesave
    log('...found previous configuration...')
rnn = RNN(Nlayers, Ndirs, Nx, Nh, n_classes, Ah, Ay, predictPer, loss, L1reg, L2reg, momentum,
          seed, frontEnd, filename, initParams)

################################ TRAIN THE RNN #############################
train_cost = []
delta_train = 5.0
delta_valid = 10.0
old_training_error = 0.0
old_valid_error = 0.0
result = []  # list for saving all predictions made by the network
# file = 'training_pred.pickle.gz'
for k in range(n_epoch):
    correct_number_train = 0.0
    correct_number_valid = 0.0
    class_occurrence_train = np.zeros(n_classes)
    class_occurrence_valid = np.zeros(n_classes)
    confusion_matrix_train = np.zeros((n_classes, n_classes))
    confusion_matrix_valid = np.zeros((n_classes, n_classes))
    n_data = 0
    for i in range(n_batches):
        cost, output, gradient, hidden_act, t, x_1, wout = rnn.train(input[i], mask[i], label[i], lrate)
        rnn.save(filesave)
        # if not isinstance(cost, float):
        #     print('----> hidden activation: ', hidden_act)
        #     print('----> Cost: ', cost)
        #     print('----> Predictions from network: ', output)
        #     print('----> Labels: ', t)
        #     print('----> Inputs: ', input[i])
        #     print('----> Gradient: ', gradient)
        #     print('\n\n\n\n')
        train_cost.append(cost)
        prob = rnn.predict(input[i], mask[i])
        for jj in range(prob.shape[1]):
            for kk in range(prob.shape[0]):
                prediction = (-prob[kk, jj, :]).argsort()
                label_sorted = (-label[i][kk, jj, :]).argsort()
                if any(label[i][kk, jj, :]):
                    n_data += 1
                    if prediction[0] == label_sorted[0]:
                        correct_number_train += 1
                    confusion_matrix_train[prediction[0], label_sorted[0]] += 1
                    class_occurrence_train[label_sorted[0]] += 1
    log('> epoch ' + str(k) + ', training cost: ' + str(100 * np.mean(train_cost)))
    confusion_matrix_train = 100 * confusion_matrix_train / class_occurrence_train
    train_error_rate = 100 * (1.0 - correct_number_train / n_data)
    log('Error rate on training set is ' + str(train_error_rate) + ' (%)')
    log('Confusion Matrix on training set is \n\n ' + str(confusion_matrix_train) + ' (%)\n')
    n_data = 0
    for i in range(n_valid):
        prob = rnn.predict(valid[i], mask_valid[i])
        for jj in range(prob.shape[1]):
            for kk in range(prob.shape[0]):
                prediction = (-prob[kk, jj, :]).argsort()
                # result.append([])
                # result[-1] = prediction
                label_sorted = (-label_valid[i][kk, jj, :]).argsort()
                if any(label_valid[i][kk, jj, :]):
                    n_data += 1
                    if prediction[0] == label_sorted[0]:
                        correct_number_valid += 1
                    confusion_matrix_valid[prediction[0], label_sorted[0]] += 1
                    class_occurrence_valid[label_sorted[0]] += 1
    confusion_matrix_valid = 100 * confusion_matrix_valid / class_occurrence_valid
    valid_error_rate = 100 * (1.0 - correct_number_valid / n_data)
    log('Error rate is on validation set is ' + str(valid_error_rate) + ' (%)')
    log('Confusion Matrix on validation set is \n\n ' + str(confusion_matrix_valid) + ' (%)\n')
    if k != 0:
        if train_error_rate > (old_training_error + delta_train):
            print('--->>> Network is diverging!')
            print('previous training error: ', old_training_error)
            print('new training error: ', train_error_rate)
            break
        if valid_error_rate > (old_valid_error + delta_valid):
            print('Network is over-fitting!')
            print('previous valid error: ', old_valid_error)
            print('new valid error: ', valid_error_rate)
            break

### IMPLEMENT STOPPING CRITERION ########

    old_training_error = train_error_rate
    old_valid_error = valid_error_rate
# cPickle.dump(result, gzip.open(file, 'wb'))

########################## TEST ON THE TEST DATA ###########################
correct_number = 0.0
n_data = 0
confusion_matrix = np.zeros((n_classes, n_classes))
class_occurrence = np.zeros(n_classes)
log('number of test batches:' + str(n_test))
for j in range(n_test):
    output = rnn.predict(test[j], mask_test[j])
    for jj in range(output.shape[1]):
        for kk in range(output.shape[0]):
            p = output[kk, jj, :]
            l = label_test[j][kk, jj, :]
            p_sorted = (-p).argsort()
            l_sorted = (-l).argsort()
            if any(l):
                n_data += 1
                if p_sorted[0] == l_sorted[0]:
                    correct_number += 1
                confusion_matrix[p_sorted[0], l_sorted[0]] += 1
                class_occurrence[l_sorted[0]] += 1

# ############################## TEST RESULTS #############################
confusion_matrix = 100 * confusion_matrix / class_occurrence
error_rate = 100 * (1.0 - correct_number / n_data)
log('Error rate is ' + str(error_rate) + ' (%)')
log('Confusion Matrix is \n\n ' + str(confusion_matrix) + ' (%)\n')
rnn.save(filesave)
N = len(confusion_matrix) # number of classes =pred_mat.shape[1] ?
class_list = parse_classes(classes,N)
Docpath = fileTex
TabPath = fileTex
Docname = "Confusion_matrix"
TabName = "table"
texfile = DocTabTex(confusion_matrix,class_list,"Tab01","legende",Docpath,TabPath,Docname,TabName,option='t',commentaire="%test commentaires")
texfile.creatTex();
