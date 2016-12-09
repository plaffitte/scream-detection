import os as os
import scikits.audiolab as audlab
import cPickle
import gzip
import sys
import numpy as np
import shutil
from mfcc import get_mfcc

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--", "").replace("-", "_")
        args[key] = arg_elements[2*i+1]
    return args

arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
# arguments = parse_arguments(arg_elements)
# name_var = arguments['data_type']
name_var = 'test'
initial_path = '/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/'+name_var+'_labels' # label files
target_path = "/home/piero/Documents/Speech_databases/test/test/data"
os.chdir(initial_path)
cur_dir = os.getcwd()
file_list = os.listdir(cur_dir)
wav_dir = os.path.join(os.path.split(initial_path)[0], 'wav')
label_dic = {}
N_classes = 2
label_dic['Noise'] = 0
# label_dic['BG_voice'] = 1
# label_dic['Conversation'] = 0
# label_dic['Shouting'] = 1
label_dic['Scream'] = 1
# label_dic['Other'] = 1
window_step = 0.010 # in seconds
label_vector = np.zeros(1, dtype=np.float32)
data_vector = np.zeros((1, 130), dtype=np.float32)
time_per_occurrence_class = [[] for i in range(N_classes)]
logfile = os.path.join(target_path, 'data_log_'+name_var+'.log')
log = open(logfile, 'w')
N = 10 # contextual window
total_length = 0.0
for i in range(len(file_list)):
    lab_name = file_list[i] # os.path.split(os.path.join(wav_dir,file_list[i]))[1]
    if '~' in lab_name:
        continue
    with open(os.path.join(cur_dir, file_list[i]), 'r') as f:
        lines = f.readlines()
        wave_name = os.path.join(wav_dir, lab_name[:-4]+'.wav')
        f = audlab.Sndfile(wave_name, 'r')
        freq = f.samplerate
        for j in xrange(len(lines)):
            try:
                cur_line = lines[j].split()
                start = int(cur_line[0])
                stop = int(cur_line[1])
                label = cur_line[2]
                length = stop/10.0**7-start/10.0**7
                audio = f.read_frames(freq*(length))
                total_length += length
                if label in label_dic:
                    mono_signal = audio  # audio[:,0]
                    energy = np.sum(mono_signal**2, 0)/len(mono_signal)
                    signal = mono_signal  # mono_signal/math.sqrt(energy)
                    samplerate = f.samplerate
                    mfcc = get_mfcc(signal, samplerate, winstep=window_step, nfft=2048, highfreq=8000, lowfreq=10)
                    N_iter = np.floor(len(mfcc)/N)
                    # apply context window
                    if (length/window_step) > N:
                        mfcc_matrix = np.zeros((1, 13*N))
                        for k in range(int(N_iter)):
                            mfcc_vec = []
                            for kk in range(N):
                                mfcc_vec = np.concatenate((mfcc_vec, mfcc[k*N+kk, :]))
                            mfcc_matrix = np.concatenate((mfcc_matrix, mfcc_vec[np.newaxis, :]))
                        # get the numeric label corresponding to the literal label
                        num_label = label_dic[label]*np.ones(len(mfcc_matrix)-1)
                        if j == 0:
                            data_vector = mfcc_matrix[1:, :].astype(np.float32, copy=False)
                            label_vector = num_label.astype(np.float32, copy=False)
                            for k in range(len(label_dic)):
                                if label_dic[label] == k:
                                    time_per_occurrence_class[k].append((stop-start)/(10.0**7))
                        else:
                            label_vector = np.append(label_vector, num_label.astype(np.float32, copy=False))
                            data_vector = np.concatenate((data_vector, mfcc_matrix[1:, :].astype(np.float32, copy=False)), 0)
                            for k in range(len(label_dic)):
                                if label_dic[label] == k:
                                    time_per_occurrence_class[k].append((stop-start)/(10.0**7))
                    else:
                        print('Input data sequence does not match minimal length requirement: ignoring')
                else:
                    del audio
            except KeyError, e:
                print "Wrong label name:", label, "at line", j+1
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise

data_vector = data_vector[1:, :]
label_vector = label_vector[1:]
shutil.copyfile('/home/piero/Documents/Scripts/format_pickle_data.py',
                '/home/piero/Documents/Speech_databases/test/DBN_audio_classification/format_pickle_data.py')
total_L_sec = stop/(10.0**7)
total_N = total_L_sec/window_step
obj = [data_vector, label_vector]
# Now write to file, for pdnn learning
target_name = os.path.join(target_path, name_var+'.pickle.gz')
cPickle.dump(obj, gzip.open(target_name, 'wb'), cPickle.HIGHEST_PROTOCOL)
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
string = 'total length:' + str(total_length) + '\n'
log.write(string)
string = 'length of data vector: ' + str(len(data_vector)) + '\n'
log.write(string)
log.close()
