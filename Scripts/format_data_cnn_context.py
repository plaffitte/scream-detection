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
shutil.copyfile('/home/piero/Documents/Scripts/format_data_cnn.py', os.path.join(exp_path,'format_data_cnn.py'))
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
    nfilt = 2 * size
data_vector = np.zeros((1, nfilt * N), dtype=np.float32)
time_per_occurrence_class = [[] for i in range(N_classes)]
logfile = os.path.join(target_path, 'data_log_'+name_var+'.log')
log = open(logfile, 'w')
time = 0
buffer_vec = []
ind_buffer = 0

for i in range(len(file_list)):
    lab_name = file_list[i] #os.path.split(os.path.join(wav_dir,file_list[i]))[1]
    print("-->> Reading file:", lab_name)
    if '~' in lab_name:
        continue
    with open(os.path.join(cur_dir, file_list[i]), 'r') as lab_f:
        zero_pad = False
        lines = lab_f.readlines()
        j = 0
        end_file = False
        if "WS" in lab_name:
            wave_name = os.path.join(wav_dir, lab_name[:-7]+'.wav')
        else:
            wave_name = os.path.join(wav_dir, lab_name[:-4]+'.wav')
        f = audlab.Sndfile(wave_name, 'r')
        freq = f.samplerate
        while not end_file:
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
                        feat, energy = MFSC(signal, freq, winstep=window_step, winlen=window_size, nfft=2048, lowfreq=100,
                                            highfreq=highfreq, nfilt=nfilt)
                        # feat = np.concatenate((feat, energy[:, np.newaxis]), 1)
                        feat[:, 0] = energy
                        if compute_delta == "True":
                            d1_feat = np.zeros((feat.shape[0]-1, feat.shape[1]))
                            for k in range(feat.shape[0]-1):
                                d1_feat[k,:] = feat[k+1,:] - feat[k,:]
                            feat = feat[1:,:]
                        N_iter = len(feat) / N # math.floor(L/window_step/N)
                        indx = 0
                        mfcc_matrix = np.zeros((1, nfilt * N))
                        no_label = True
                        # apply context window
                        while indx < len(feat):
                            for kk in range(min(len(feat) - indx - 1, N - (len(buffer_vec) / nfilt))):
                                buffer_vec = np.concatenate((buffer_vec, feat[indx + kk, :]))
                                ind_buffer += 1
                            if compute_delta == "True":
                                buffer_vec_d = d1_feat[indx:min(len(d1_feat), indx + N), :]
                                buffer_vec = np.concatenate((buffer_vec, buffer_vec_d))
                              # Use label from sequence located in center of buffer !!
                            if ind_buffer >= (N / 2) and no_label:
                                num_label = label_dic[label]
                                no_label = False
                                # time_per_occurrence_class[label_dic[label]].append(length)
                                # time = np.sum(time_per_occurrence_class[label_dic[label]])
                            if len(buffer_vec) == nfilt * N:
                                data_vector = np.concatenate((data_vector, buffer_vec[np.newaxis, :].astype(np.float32, copy=False)), 0)
                                label_vector = np.append(label_vector, num_label)
                                buffer_vec = []
                                ind_buffer = 0
                            indx += (nfilt - slide)
                    if len(buffer_vec) != 0:
                        zero_pad = True
                else:
                    del audio
                    if zero_pad:
                        buffer_vec = np.concatenate((buffer_vec, np.zeros((nfilt * N) - len(buffer_vec))))
                        data_vector = np.concatenate((data_vector, buffer_vec[np.newaxis, :].astype(np.float32, copy=False)), 0)
                        buffer_vec = []
                        zero_pad = False
                j += 1
                if j == len(lines):
                    end_file = True;
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
    log.write("\n")
log.close()
