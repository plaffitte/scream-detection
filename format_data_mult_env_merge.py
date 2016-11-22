# This script is made for multi-environment audio classification problems, where several label files exist for the same
# audio file. It reads all the label files and merges them into one unique label file with n*m classes, where n and m are
# the original number of classes in each label file.

import os as os
import scikits.audiolab as audlab
import cPickle, gzip, sys
import numpy as np
import math, shutil
from spectral import get_mfcc
from util_func import parse_arguments, parse_classes

arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
arguments = parse_arguments(arg_elements)
name_var = arguments['data_type']
classes_1 = parse_classes(arguments['classes_1'])
classes_2 = parse_classes(arguments['classes_2'])
window_step = float(arguments['window_step'])
window_size = float(arguments['window_size'])
highfreq = int(arguments['highfreq'])
lowfreq = int(arguments['lowfreq'])
size = int(arguments['size'])
N = int(arguments['N'])
slide = int(arguments['slide'])
exp_path = arguments['exp_path']
threshold = int(arguments['threshold'])
compute_delta = arguments['deltas']
shutil.copyfile('/home/piero/Documents/Scripts/format_pickle_data.py',
                os.path.join(exp_path,'format_pickle_data.py'))
N_classes_1 = len(classes_1)
N_classes_2 = len(classes_2)
label_dic_1 = {}
label_dic_2 = {}
label_dic_comp = {}
for k in range(N_classes_1):
    label_dic_1[classes_1[k]] = k
for k in range(N_classes_2):
    label_dic_2[classes_2[k]] = k
n = 0
for i, j in zip(classes_1, classes_2):
    label_dic_comp[n] = str(int(label_dic_1[i]) + int(label_dic_2[j]))
    n += 1
initial_path = '/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/'+name_var+'_labels' # label files
target_path = os.path.join(exp_path,'data')
os.chdir(initial_path)
cur_dir = os.getcwd()
file_list = os.listdir(cur_dir)
wav_dir = os.path.join(os.path.split(initial_path)[0], 'wav')
label_vector = np.zeros(1, dtype=np.float32)
if compute_delta == "True":
    size = 2 * size
data_vector = np.zeros((1, size * N), dtype=np.float32)
# time_per_occurrence_class = [[] for i in range(N_classes)]
logfile = os.path.join(target_path, 'data_log_'+name_var+'.log')
log = open(logfile, 'w')
time = 0
start = 0.0
stop = 0.0
end_file = False

### Find all lab files from similar audio file
multi_lab = {}
for i in range(len(file_list)):
    sub_list = [n for n in file_list if file_list[i] not in n]
    sim_name=[n for n in sub_list if file_list[i][:-6] in n]
    multi_lab[file_list[i]]=sim_name
multi_lab = dict((k, v) for k, v in multi_lab.iteritems() if v)
### Work on each pair of multi-env label files one by one
for i,j in multi_lab.iteritems():
    lab_name_1 =  i
    lab_name_2 = j
    index_1 = 0
    index_2 = 0
    mv_1 = True
    mv_2 = True
    print("-->> Reading file:", lab_name_1)
    if ('~' in lab_name_1) or ('~' in lab_name_2):
        continue
    with open(os.path.join(cur_dir, lab_name_1), 'r') as f1:
        with open(os.path.join(cur_dir, lab_name_2), 'r') as f2:
            lines_1 = f1.readlines()
            lines_2 = f2.readlines()
            if "WS" in lab_name_1:
                wave_name_1 = os.path.join(wav_dir, lab_name_1[:-7]+'.wav')
            else:
                wave_name_1 = os.path.join(wav_dir, lab_name_1[:-4]+'.wav')
            if "WS" in lab_name_2:
                wave_name_2 = os.path.join(wav_dir, lab_name_2[:-7]+'.wav')
            else:
                wave_name_2 = os.path.join(wav_dir, lab_name_2[:-4]+'.wav')
            wave_1 = audlab.Sndfile(wave_name_1, 'r')
            wave_2 = audlab.Sndfile(wave_name_2, 'r')
            freq_1 = wave_1.samplerate
            freq_2 = wave_2.samplerate
            while not end_file :
                try:
                    ### Check if next label needs to be retrieved for each label file
                    if mv_1:
                        cur_line_1 = lines_1[index_1].split()
                        start_1 = float(cur_line_1[0])
                        stop_1 = float(cur_line_1[1])
                        label_1 = cur_line_1[2]
                    if mv_2:
                        cur_line_2 = lines_2[index_2].split()
                        start_2 = float(cur_line_2[0])
                        stop_2 = float(cur_line_2[1])
                        label_2 = cur_line_2[2]
                    ### Find out which of the two labels has the shortest length and read audio for that length
                    if np.min(start_1, start_2) == start_1:
                        start = start_1
                    else:
                        start = start_2
                    if np.min(stop_1, stop_2) == stop_1:
                        stop = stop_1
                        index_1 += 1
                        mv_1 = True
                    else:
                        mv_1 = False
                    if np.min(stop_1, stop_2) == stop_2:
                        stop = stop_2
                        index_2 += 1
                        mv_2 = True
                    else:
                        mv_2 = False
                    if "WS" in lab_name:
                        length = stop - start
                    else:
                        length = (stop - start) / 10.0 ** 7
                    audio = f.read_frames(freq * length)
                    if (label_1 in label_dic_1) and (label_2 in label_dic_2):
                        if time < threshold:
                            signal = audio  # audio/math.sqrt(energy)
                            mfcc = get_mfcc(signal, freq, winstep=window_step, winlen=window_size, nfft=2048, lowfreq=lowfreq,
                                            highfreq=highfreq, numcep=size, nfilt=size + 2)
                            if compute_delta == "True":
                                d1_mfcc = np.zeros((mfcc.shape[0]-1,mfcc.shape[1]))
                                for k in range(mfcc.shape[0]-1):
                                    d1_mfcc[k,:] = mfcc[k+1,:] - mfcc[k,:]
                                mfcc = mfcc[1:,:]
                            N_iter = np.floor((len(mfcc) - N) / slide)
                            # apply context window
                            if (length/window_step) > N:
                                # time_per_occurrence_class[label_dic[label]].append(length)
                                # time = np.sum(time_per_occurrence_class[label_dic[label]])
                                mfcc_matrix = np.zeros((1, size * N))
                                for k in range(int(N_iter)):
                                    mfcc_vec = []
                                    for kk in range(N):
                                        mfcc_vec = np.concatenate((mfcc_vec, mfcc[k * slide + kk, :]))
                                        if compute_delta == "True":
                                            mfcc_vec = np.concatenate((mfcc_vec, d1_mfcc[k * slide + kk, :]))
                                    mfcc_matrix = np.concatenate((mfcc_matrix, mfcc_vec[np.newaxis, :]))
                                label_key = int(str(label_dic_1[label_1])+str(label_dic_2[label_2]))
                                num_label =  label_dic_comp.keys()[label_dic_comp.values().index(label_key)] * np.ones(len(mfcc_matrix) - 1)
                                label_vector = np.append(label_vector, num_label.astype(np.float32, copy=False))
                                data_vector = np.concatenate((data_vector, mfcc_matrix[1:,:].astype(np.float32, copy=False)),0)
                            else:
                                print('Input data sequence does not match minimal length requirement: ignoring')
                    else:
                        del audio
                except KeyError, e:
                    print "Wrong label name:", label, "at line", j+1
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
    print("Size of data_vector: ", data_vector.shape)
data_vector = data_vector[1:,:]
label_vector = label_vector[1:]

total_L_sec = stop / (10.0 ** 7)
total_N = total_L_sec/window_step
obj = [data_vector, label_vector]
# Now write to file, for pdnn learning
target_name = os.path.join(target_path,name_var+'.pickle.gz')
cPickle.dump(obj, gzip.open(target_name,'wb'),cPickle.HIGHEST_PROTOCOL)
