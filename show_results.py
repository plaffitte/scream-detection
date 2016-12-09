import numpy
import sys
import os
import cPickle, gzip
import numpy as np
import util_func as utils

arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
arguments = utils.parse_arguments(arg_elements)
pred_file = arguments['pred_file']
class_str = arguments['classes']
filepath = arguments['filepath']
#pred_file = 'dnn.classify.pickle.gz'
if '.gz' in pred_file:
    pred_mat = cPickle.load(gzip.open(pred_file, 'rb'))
else:
    pred_mat = cPickle.load(open(pred_file, 'rb'))

# load the testing set to get the labels
test_data, test_labels = cPickle.load(gzip.open('data/test.pickle.gz', 'rb'))
test_labels = test_labels.astype(numpy.int32)

confusion_matrix = np.zeros((pred_mat.shape[1], pred_mat.shape[1])) # rows represent predicted classes and columns
                                                                                                                    # represent true classes
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
if '+' in class_str:
    mult = class_str.split('+')
    class_list =[]
    for j in range(len(mult[0].split(','))):
        if j==0:
            string_1 = mult[0].split(',')[j][2:]
        elif j ==(len(mult[0].split(',')) - 1):
            string_1 = mult[0].split(',')[j][:-1]
        else:
            string_1 = mult[0].split(',')[j]
        for k in range(len(mult[1].split(','))):
            if k==0:
                class_list.append( string_1 + '_'  + mult[1].split(',')[k][1:])
            elif k==(len(mult[1].split(',')) - 1):
                class_list.append(string_1 + '_ '+ mult[1].split(',')[k][:-2])
            else:
                class_list.append(string_1 + '_' + mult[1].split(',')[k])
else:
    class_list = utils.parse_classes(class_str)
for i in range(len(class_list)):
    if '_' in class_list[i]:
        class_list[i] = class_list[i].replace('_', ' ')
print(class_list)
utils.text_to_latex(confusion_matrix,class_list,filepath)
