import numpy
import sys
import os
import cPickle, gzip
import numpy as np
from text_to_latex import *

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--","").replace("-", "_");
        args[key] = arg_elements[2*i+1]
    return args

def parse_classes(class_str,N):
    class_list = []
    start = 0
    i=0
    for j in range(N):
        while class_str[i] not in [',', '}']:
            i += 1
        end = i
        class_list.append(class_str[start:end])
        start = i+1
        i += 1
    return class_list

arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
arguments = parse_arguments(arg_elements)
pred_file = arguments['pred_file']
class_str = arguments['classes']
filepath = arguments['filepath']
os.chdir(filepath)
if '.gz' in pred_file:
    pred_mat = cPickle.load(gzip.open(pred_file, 'rb'))
else:
    pred_mat = cPickle.load(open(pred_file, 'rb'))

# load the testing set to get the labels
test_file = pred_file[0:-19]+'.pickle.gz'
test_data, test_labels = cPickle.load(gzip.open(test_file, 'rb'))
test_labels = test_labels.astype(numpy.int32)

confusion_matrix = np.zeros((pred_mat.shape[1],pred_mat.shape[1])) # rows represent predicted classes and columns represent true classes
class_occurrence = np.zeros(pred_mat.shape[1])
correct_number = 0.0
for i in xrange(pred_mat.shape[0]):
    p = pred_mat[i,:]
    p_sorted = (-p).argsort()
    if p_sorted[0] == test_labels[i]:
        correct_number += 1
        confusion_matrix[test_labels[i],test_labels[i]] += 1
    else:
        confusion_matrix[p_sorted[0],test_labels[i]] += 1

    class_occurrence[test_labels[i]]+=1

confusion_matrix = 100*confusion_matrix/class_occurrence
error_rate = 100 * (1.0 - correct_number / pred_mat.shape[0])
print 'Error rate is ' + str(error_rate) + ' (%)'
print 'Confusion Matrix is \n\n ' + str(confusion_matrix) + ' (%)\n'
N = len(confusion_matrix) # number of classes
class_list = parse_classes(class_str,N)
text_to_latex(confusion_matrix,class_list,filepath)
