import os as os
import scikits.audiolab as audlab
import cPickle, gzip, sys
import numpy as np
import math, shutil
from spectral import MFSC
from util_func import parse_arguments, parse_classes

typeFeature = "MFSC"
name_cur_file = os.path.basename(__file__)
name_cur_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.getcwd()

##### Parse Arguments #####
arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
arguments = parse_arguments(arg_elements)
name_var = arguments['data_type']
target_path = arguments['rep_test']
param_str = arguments['param']
classes = parse_classes(arguments['classes'])
N_classes = len(classes)
param_list = parse_classes(param_str)
window_step=float(param_list[0] )# in seconds, hop size between two successive mfcc windows
window_size=float(param_list[1] )# in seconds, size of MFCC window
highfreq=float(param_list[2]) # maximal analysis frequency for mfcc
lowfreq=float(param_list[3]) # minimal analysis frequency for mfcc
nfilt=int(param_list[4]) # number of mfcc coef
N=int(param_list[5]) #contextual window
slide=int(param_list[6])
threshold = int(param_list[7])
compute_delta = param_list[8]

##### Initialize label dic #####
label_dic = {}
for i in range(len(classes)):
    label_dic[classes[i]] = i
    print classes[i] + " : " + str(i)

##### Copy label files and current script where necessary to trace experiment #####
shutil.copyfile(os.path.join(name_cur_dir, name_cur_file), os.path.join(target_path, name_cur_file))
label_path = source_dir + '/Data_Base/' + name_var + '_labels' # Path to label files
shutil.rmtree(os.path.join(target_path, name_var+'_labels'), ignore_errors=True)
shutil.copytree(label_path, os.path.join(target_path, name_var+'_labels'))

##### Memory allocation #####
label_vector = np.zeros(1, dtype=np.float32)
if compute_delta == "True":
    nfilt = 2 * nfilt
data_vector = np.zeros((1, nfilt * N), dtype=np.float32)
time_per_occurrence_class = [[] for i in range(N_classes)]

##### Couple log writings #####
logfile = os.path.join(target_path, 'data_log_'+name_var+'.log')
log = open(logfile, 'w')
string = '===== Parametre Features:\n'; log.write(string)
string = ' typeFeature : ' + typeFeature + '\n'; log.write(string)
string = ' window_step : ' + param_list[0] + '\n'; log.write(string)
string = ' window_size : ' + param_list[1] + '\n'; log.write(string)
string = ' highfreq : ' + param_list[2] + '\n'; log.write(string)
string = ' lowfreq : ' + param_list[3] + '\n'; log.write(string)
string = ' nfilt : ' + param_list[4] + '\n'; log.write(string)
string = ' N contextual window : ' + param_list[5] + '\n'; log.write(string)
string = ' Slide : ' + param_list[6] + '\n\n'; log.write(string)
string = '===== Name of corresponding wav file:\n'; log.write(string)

##### Main Loop #####
file_list = os.listdir(label_path)
file_list = [file for file in file_list if os.path.isfile(os.path.join(label_path, file))]
wav_dir = os.path.join(os.path.split(label_path)[0], 'wav')
time = 0
for i in range(len(file_list)):
    lab_name = file_list[i] #os.path.split(os.path.join(wav_dir,file_list[i]))[1]
    print("-->> Reading file:", lab_name)
    if '~' in lab_name:
        continue
    with open(os.path.join(label_path, file_list[i]), 'r') as lab_f:
        lines = lab_f.readlines()
        if "WS" in lab_name:
            wave_name = os.path.join(wav_dir, lab_name[:-7]+'.wav')
        else:
            wave_name = os.path.join(wav_dir, lab_name[:-4]+'.wav')
        f = audlab.Sndfile(wave_name, 'r')
        freq = f.samplerate
        print("length of current label file:", len(lines))
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
                            time_per_occurrence_class[label_dic[label]].append(length)
                            time = np.sum(time_per_occurrence_class[label_dic[label]])
                            mfcc_matrix = np.zeros((1, nfilt * N))
                            for k in range(int(N_iter)):
                                mfcc_vec = []
                                for kk in range(N):
                                    mfcc_vec = np.concatenate((mfcc_vec, feat[k * N + kk, :]))
                                    if compute_delta == "True":
                                        mfcc_vec = np.concatenate((mfcc_vec, d1_mfcc[k * slide + kk, :]))
                                mfcc_matrix = np.concatenate((mfcc_matrix, mfcc_vec[np.newaxis, :]))
                            num_label = label_dic[label] * np.ones(len(mfcc_matrix) - 1)
                            label_vector = np.append(label_vector, num_label.astype(np.float32, copy=False))
                            data_vector = np.concatenate((data_vector, mfcc_matrix[1:,:].astype(np.float32, copy=False)),0)
                        else:
                            pass
                            # print('Input data sequence does not match minimal length requirement: ignoring')
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
