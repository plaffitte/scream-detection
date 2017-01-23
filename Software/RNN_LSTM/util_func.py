import sys
from datetime import datetime
import numpy

def parse_classes(class_str):
    class_list = []
    start = 1
    i=0
    while not class_str[i+1] is '}':
        while class_str[i+1] not in {',', '}'}:
            i += 1
        end = i+1
        class_list.append(class_str[start:end])
        if not class_str[i+1] is '}':
            i += 1
        start = i + 1
    return class_list

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--", "").replace("-", "_")
        args[key] = arg_elements[2*i+1]
    return args

def log(string):
    sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')
    print('[' + str(datetime.now()) + '] ' + str(string) + '\n')

def format_results(pred, labels, multi_label, n_outs):
    correct_number = 0.0
    confusion_matrix = numpy.zeros((n_outs, n_outs))
    class_occurrence = numpy.zeros((1, n_outs))
    recall_matrix = numpy.zeros(n_outs)
    false_pred_matrix = numpy.zeros(n_outs)
    precision_matrix = numpy.zeros(n_outs)
    N_pred_true = numpy.zeros(n_outs)
    N_pred_false = numpy.zeros(n_outs)
    [a, b, c] = numpy.shape(pred)
    N_samples = 0
    for i in range(a):
        for j in range(b):
            out=numpy.array(pred[i][j],dtype=int)
            lab=numpy.array(labels[i][j],dtype=int)
            res=~(~(out==1)&lab) | (out&~(lab==1))
            correct_number += len(numpy.where(res))
            for k in range(n_outs):
                if lab[k]:
                    class_occurrence[0, k] += 1
                    if out[k]==lab[k]:
                        N_pred_true[k] += 1
                else:
                    if out[k]!=lab[k]:
                        N_pred_false[k] += 1
            N_samples += 1
    log("Number of true pred:"+str(N_pred_true))
    log("Number of false pred:"+str(N_pred_false))
    log("Class occurence:" + str(class_occurrence))
    error = (numpy.sum(numpy.sum(class_occurrence)) - correct_number) / numpy.sum(numpy.sum(class_occurrence))
    recall_matrix = 100 * N_pred_true / class_occurrence.T[:, 0]
    precision_matrix = 100 * N_pred_true / (N_pred_false + N_pred_true)
    false_pred_matrix = 100 * N_pred_false / (N_samples - class_occurrence.T[:, 0])
    log('Accuracy (Recall) Matrix: \n\n ' + str(numpy.around(recall_matrix, 2)) + ' (%)\n')
    log('Precision Matrix: \n\n ' + str(numpy.around(precision_matrix, 2)) + ' (%)\n')
    log('False Predictions Matrix: \n\n ' + str(numpy.around(false_pred_matrix, 2)) + ' (%)\n')
    return error
