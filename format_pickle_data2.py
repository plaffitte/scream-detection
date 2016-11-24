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
classes = parse_classes(arguments['classes'])
# name_var = 'test'
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
shutil.copyfile('/home/piero/Documents/Scripts/format_pickle_data2.py', os.path.join(exp_path,'format_pickle_data2.py'))
N_classes = len(classes)
label_dic = {}
for k in range(N_classes):
    label_dic[classes[k]] = k
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
time_per_occurrence_class = [[] for i in range(N_classes)]
logfile = os.path.join(target_path, 'data_log_'+name_var+'.log')
log = open(logfile, 'w')
time = 0

for i in range(len(file_list)):
    lab_name = file_list[i] #os.path.split(os.path.join(wav_dir,file_list[i]))[1]
    print("-->> Reading file:", lab_name)
    if '~' in lab_name:
        continue
    with open(os.path.join(cur_dir, lab_name), 'r') as lab_f:
        lines = lab_f.readlines()
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
                    if time < threshold:
                        # energy = np.sum(audio ** 2, 0) / len(audio)
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
                            time_per_occurrence_class[label_dic[label]].append(length)
                            time = np.sum(time_per_occurrence_class[label_dic[label]])
                            mfcc_matrix = np.zeros((1, size * N))
                            for k in range(int(N_iter)):
                                mfcc_vec = []
                                for kk in range(N):
                                    mfcc_vec = np.concatenate((mfcc_vec, mfcc[k * slide + kk, :]))
                                    if compute_delta == "True":
                                        mfcc_vec = np.concatenate((mfcc_vec, d1_mfcc[k * slide + kk, :]))
                                mfcc_matrix = np.concatenate((mfcc_matrix, mfcc_vec[np.newaxis, :]))
                            num_label = label_dic[label] * np.ones(len(mfcc_matrix) - 1)
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
# Feature Standardization
#data_vector = preproc.scale(data_vector)

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
