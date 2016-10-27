import os as os
import scikits.audiolab as audlab
import cPickle, gzip, sys
import numpy as np
import math, shutil
from spectral import *
from features import fbank
from util_func import parse_arguments, parse_classes

arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
arguments = parse_arguments(arg_elements)
name_var = arguments['data_type']
classes = parse_classes(arguments['classes'])
# name_var = 'test'
window_step = float(arguments['window_step'])
window_size = float(arguments['window_size'])
highfreq = int(arguments['highfreq'])
lowfreq = int(arguments['lowfreq'])
N = int(arguments['N'])
slide = int(arguments['slide'])
exp_path = arguments['exp_path']
threshold = int(arguments['threshold'])
nfilt = int(arguments['nfilt'])
compute_delta = arguments['deltas']
N_classes = len(classes)
label_dic = {}
shutil.copyfile('/home/piero/Documents/Scripts/format_data_cnn.py',
                os.path.join(exp_path,'format_data_cnn.py'))
for k in range(N_classes):
    label_dic[classes[k]] = k
initial_path = '/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/'\
                        +name_var+'_labels' # label files
target_path = os.path.join(exp_path,'data')
os.chdir(initial_path)
cur_dir = os.getcwd()
file_list = os.listdir(cur_dir)
wav_dir = os.path.join(os.path.split(initial_path)[0], 'wav')
label_vector = np.zeros(1, dtype=np.float32)
if compute_delta == "True":
    data_vector = np.zeros((1, 2 * nfilt * N), dtype=np.float32)
else:
    data_vector = np.zeros((1, nfilt * N), dtype=np.float32)
time_per_occurrence_class = [[] for i in range(N_classes)]
logfile = os.path.join(target_path, 'data_log_'+name_var+'.log')
log = open(logfile, 'w')
time = 0

for i in range(len(file_list)):
    lab_name = file_list[i] #os.path.split(os.path.join(wav_dir,file_list[i]))[1]
    print("-->> Reading file:", lab_name)
    if '~' in lab_name:
        continue
    with open(os.path.join(cur_dir, file_list[i]), 'r') as f:
        lines = f.readlines()
        if "WS" in lab_name:
            wave_name = os.path.join(wav_dir, lab_name[:-7]+'.wav')
        else:
            wave_name = os.path.join(wav_dir, lab_name[:-4]+'.wav')
        f = audlab.Sndfile(wave_name, 'r')
        freq = f.samplerate
        for j in xrange(len(lines)):
            try:
                cur_line = lines[j].split()
                start = float(cur_line[0])
                stop = float(cur_line[1])
                label = cur_line[2]
                if "WS" in lab_name:
                    length = stop - start
                else:
                    length = (stop - start) / 10.0 ** 7
                audio = f.read_frames(freq * length)
                if label in label_dic:
                    time_per_occurrence_class[label_dic[label]].append(length)
                    time = np.sum(time_per_occurrence_class[label_dic[label]])
                    if time < threshold:
                        # energy = np.sum(audio ** 2, 0) / len(audio)
                        signal = audio  # audio/math.sqrt(energy)
                        feat, energy = MFSC(signal, freq, winstep=window_step, nfft=2048,
                                            lowfreq=100, highfreq=highfreq, nfilt=nfilt)
                        # feat = np.concatenate((feat, energy[:, np.newaxis]), 1)
                        feat[:, 0] = energy
                        if compute_delta == "True":
                            d1_mfcc = np.zeros((feat.shape[0]-1, feat.shape[1]))
                            for k in range(feat.shape[0]-1):
                                d1_mfcc[k,:] = feat[k+1,:] - feat[k,:]
                            feat = feat[1:,:]
                        N_iter = len(feat) / N # math.floor(L/window_step/N)
                        # apply context window
                        if (length/window_step) > N:
                            mfcc_matrix = np.zeros((1, nfilt * N))
                            d1_matrix = np.zeros((1, nfilt * N))
                            for k in range(int(N_iter)):
                                mfcc_vec = []
                                d1_vec = []
                                for kk in range(N):
                                    mfcc_vec = np.concatenate((mfcc_vec, feat[k * N + kk, :]))
                                    if compute_delta == "True":
                                        d1_vec = np.concatenate((d1_vec, d1_mfcc[k * slide + kk, :]))
                                mfcc_matrix = np.concatenate((mfcc_matrix, mfcc_vec[np.newaxis, :]))
                                if compute_delta == "True":
                                    d1_matrix = np.concatenate((d1_matrix, d1_vec[np.newaxis, :]))
                            if compute_delta == "True":
                                merged = np.concatenate((mfcc_matrix, d1_matrix), 1)
                            num_label = label_dic[label] * np.ones(len(mfcc_matrix) - 1)
                            label_vector = np.append(label_vector,
                                                     num_label.astype(np.float32, copy=False))
                            if compute_delta == "True":
                                data_vector = np.concatenate((data_vector,
                                                              merged[1:,:].astype(np.float32, copy=False)),0)
                            else:
                                data_vector = np.concatenate((data_vector,
                                                              mfcc_matrix[1:,:].astype(np.float32, copy=False)),0)
                        else:
                            print('Input data sequence does not match \
                                  minimal length requirement: ignoring')
            except KeyError, e:
                print "Wrong label name:", label, "at line", j+1
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise
    print("Size of data_vector: ", data_vector.shape)
data_vector = data_vector[1:,:]
label_vector = label_vector[1:]
# Feature Standardization
#data_vector = preproc.scale(data_vector)
# shutil.copyfile('/home/piero/Documents/Scripts/format_pickle_data3.py','/home/piero/Documents/Speech_databases/test/DBN_audio_classification3/format_pickle_data3.py')
total_L_sec = stop / (10.0 ** 7)
total_N = total_L_sec/window_step
obj = [data_vector, label_vector]
# Now write to file, for pdnn learning
target_name = os.path.join(target_path,name_var+'.pickle.gz')
cPickle.dump(obj, gzip.open(target_name,'wb'),cPickle.HIGHEST_PROTOCOL)

for class_name, class_value in label_dic.items():
    string = 'Name of corresponding wav file:'+wave_name+'\n'
    log.write(string)
    string = 'number of data from class' + class_name + ':' + str(len(time_per_occurrence_class[class_value]))+'\n'
    log.write(string)
    string = 'length of smallest data from class:' + class_name + ':' + str(min(time_per_occurrence_class[class_value]))+'\n'
    log.write(string)
    string = 'length of longest data from class:' + class_name + ':' + str(max(time_per_occurrence_class[class_value]))+'\n'
    log.write(string)
    string = 'mean length of data from class:' + class_name + ':' + str(np.mean(time_per_occurrence_class[class_value]))+'\n'
    log.write(string)
    string = 'total length of data from class:' + class_name + ':' + str(np.sum(time_per_occurrence_class[class_value]))+'\n'
    log.write(string)
log.close()
