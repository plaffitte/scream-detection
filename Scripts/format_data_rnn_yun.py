import os as os
import scikits.audiolab as audlab
import cPickle, gzip, sys
import numpy as np
import shutil
from spectral import get_mfcc
import kaldi_format_data
import time
import matplotlib.pyplot as plt
from util_func import parse_classes, parse_arguments

typeFeature = "MFCC"
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
param_list = parse_classes(param_str)
window_step=float(param_list[0] )# in seconds, hop size between two successive mfcc windows
window_size=float(param_list[1] )# in seconds, size of MFCC window
highfreq=float(param_list[2]) # maximal analysis frequency for mfcc
lowfreq=float(param_list[3]) # minimal analysis frequency for mfcc
size=int(param_list[4]) # number of mfcc coef
occurrence_threshold = int(param_list[5])
compute_delta = param_list[6]
max_seq_length = int(arguments['max_seq_len'])
n_stream = int(arguments['n_stream'])
max_batch_len = int(arguments['max_batch_len'])

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
time_per_occurrence_class = [[] for i in range(N_classes)]
data_struct = []
label_struct = []
mask_struct = []

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
trim = False
zero_pad = False
n_batch_tot = 0
re_use = False
restart = True
plot = False
plot_detail = False

########## Callback processing function ##########
def create_data(sig, label):
    mfcc = get_mfcc(sig, freq, winstep=window_step, winlen=window_size, nfft=2048, lowfreq=lowfreq,
                    highfreq=highfreq, numcep=size, nfilt=size+2)
    # One-hot encoding
    num_label = np.zeros((len(mfcc), N_classes))
    num_label[:, label_dic[label]] = 1
    # Direct encoding
    # num_label = label_dic[label] * np.ones(len(mfcc))
    return mfcc, num_label

########### Zero padding function ############
def zero_pad():
    global stream_full, data_vector, label_vector, mask_vector, line_index, max_batch_len, size
    stream_full = True
    line_index -= 1
    padding_len = max_batch_len + 1 - len(data_vector)
    if padding_len > 0:
        data_vector = np.concatenate((data_vector, np.zeros((padding_len, size))))
        label_vector = np.concatenate((label_vector, np.zeros((padding_len, N_classes))))
        # label_vector = np.concatenate((label_vector, np.zeros(padding_len)))
        mask_vector = np.concatenate((mask_vector, -1 * np.ones(padding_len)))

####### Main Loop ##########
for i in xrange(len(file_list)):
    end_file = False
    stream_full = False
    print('---------------------------->>>>>>>>>>> reading new lab file')
    #raw_input("Press Enter to continue...")
    lab_name = file_list[i] # os.path.split(os.path.join(wav_dir,file_list[i]))[1]
    print(lab_name)
    line_index = 0
    if '~' in lab_name:
        print('wrong lab file')
        continue
    with open(os.path.join(label_path, file_list[i]), 'r') as lab_f:
        lines = lab_f.readlines()
        if 'WS' in lab_name:
            wave_name = os.path.join(wav_dir, lab_name[:-7]+'.wav')
        else:
           wave_name = os.path.join(wav_dir, lab_name[:-4]+'.wav')
        # wave_name = wav_dir+"/Fic_TEST_P0_L.wav"
        f = audlab.Sndfile(wave_name, 'r')
        freq = f.samplerate
        ind_start = 0
        ind_end = 0
        while end_file != True:#            print('-------------------> Creating new batch')
            n_batch_tot += 1
            label_tensor = np.zeros((n_stream, max_batch_len, N_classes))
            # label_tensor = np.zeros((n_stream, max_batch_len))
            data_tensor = np.zeros((n_stream, max_batch_len, size))
            mask_matrix = -1 * np.zeros((n_stream, max_batch_len))
            for kk in range(n_stream):
                stream_cnt = kk
                print('--------> Creating new stream')
                stream_full = False
                data_vector = np.zeros((1, size)).astype(np.float32)
                label_vector = np.zeros((1, N_classes)).astype(np.int32)
                mask_vector = np.zeros(1).astype(np.int32)
                while stream_full != True:
                    try:
                        print('\n')
                        cur_line = lines[line_index].split()
                        start = float(cur_line[0])
                        stop = float(cur_line[1])
                        label = cur_line[2]
                        if 'WS' in lab_name:
                            length = stop - start
                        else:
                            length = (stop - start) / 10.0 ** 7
                        if re_use:
                            pass
                        else:
                            audio = f.read_frames(np.floor(freq * length))
                            time_per_occurrence_class[label_dic[label]].append(length)
                        if label in label_dic:
                            signal = audio  # audio/math.sqrt(energy)
                            n_occurrence = np.sum(time_per_occurrence_class[label_dic[label]])
                            if n_occurrence < occurrence_threshold:
                                data, labels = create_data(signal, label)
                                if plot_detail:
                                    im, ax = plt.subplots()
                                    ax.imshow(data, aspect='auto')
                                length = len(labels)
                                print('>length of new data:', length)
                                # If current data longer than maximum stream lentgh
                                if len(labels) > max_batch_len:
                                    # If data_vector is empty then proceed in chopping current data and placing it into data vector
                                    print('length data vector:', len(data_vector))
                                    if len(data_vector) == (max_batch_len+1) or len(data_vector) == 1:
                                        print('>very long sequence, will trim')
                                        if restart:
                                            ind_start = 0
                                            ind_end = max_batch_len - 1
                                            restart = False
                                        data = data[ind_start: ind_end, :]
                                        labels = labels[ind_start:ind_end, :]
                                        remaining_data = min(max_batch_len , length - ind_end)
                                        if remaining_data > 2:
                                            restart = False
                                            line_index -= 1
                                            ind_end += remaining_data
                                            ind_start += max_batch_len
                                        else:
                                            restart  =True
                                        label_vector = np.concatenate((label_vector, labels.astype(np.float32)))
                                        data_vector = np.concatenate((data_vector, data.astype(np.float32)))
                                        interm = np.zeros(len(labels))
                                        interm[0] = 1
                                        interm[-1] = 2
                                        mask_vector = np.concatenate((mask_vector, interm.astype(np.int32)))
                                        if restart:
                                            re_use = False
                                        else:
                                            re_use = True
                                    else:
                                        # Otherwise zero pad current stream and put data in next stream
                                        print('>padding with zeros to match length of maximum stream in batch')
                                        zero_pad()
                                        stream_full = True
                                        re_use = True
                                else:
                                    # Data shorter than max_stream_len -> check if fits in current stream
                                    if (len(data_vector) + len(labels)) > (max_batch_len+1):
                                        # Data doesn't fit in current stream, zero pad current stream and put data in next stream
                                        print('>padding with zeros to match length of maximum stream in batch')
                                        zero_pad()
                                        re_use = True
                                    else:
                                        # Data fits in current stream
                                        if len(labels) < max_seq_length:
                                            # Data shorter than max sequence length
                                            label_vector = np.concatenate((label_vector, labels.astype(np.float32)))
                                            data_vector = np.concatenate((data_vector, data.astype(np.float32)))
                                            interm = np.zeros(len(labels))
                                            interm[0] = 1
                                            interm[-1] = 2
                                            mask_vector = np.concatenate((mask_vector, interm.astype(np.int32)))
                                        # if data bigger than max_sequence_len, chop up in chuncks of lentgh max_sequence_len
                                        else:
                                            new_index = 0
                                            num_it = int(np.floor(len(labels) / max_seq_length))
                                            for j in range(num_it):
                                                L = min(max_seq_length, len(labels) - j * max_seq_length)
                                                if  L > 0:
                                                    data_sub = data[j * max_seq_length : j * max_seq_length + L, :]
                                                    label_sub = labels[j * max_seq_length : j * max_seq_length + L]
                                                    data_vector = np.concatenate((data_vector, data_sub.astype(np.float32)))
                                                    label_vector = np.concatenate((label_vector, label_sub.astype(np.float32)))
                                                    interm = np.zeros(L)
                                                    interm[0] = 1
                                                    interm[-1] = 2
                                                    mask_vector = np.concatenate((mask_vector, interm.astype(np.int32)))
                                                else:
                                                    pass
                                        re_use = False
                        else:
                            print('label not in label dic')
                        line_index += 1
                        print('>line index is:', line_index)
                        if line_index == len(lines) - 1:
                            end_file = True
                            stream_full = True
                            print('end of file:', end_file)
                    except KeyError, e:
                        print 'Wrong label name:', label, 'at line', line_index
                        line_index += 1
                        if line_index == len(lines) - 1:
                            end_file = True
                            stream_full = True
                    except:
                        print 'Unexpected error:', sys.exc_info()[0]
                        raise
                if end_file and len(data_vector) < max_batch_len + 1:
                    zero_pad()
                data_tensor[kk, :, :] = data_vector[1:, :]
                label_tensor[kk, :, :] = label_vector[1:, :]
                # label_tensor[kk, :] = label_vector[1:]
                mask_matrix[kk, :] = mask_vector[1:]
                if end_file == True:
                    break
            if stream_cnt == n_stream-1:
                data_struct.append(data_tensor.astype(np.float32, copy=False))
                label_struct.append(label_tensor.astype(np.float32, copy=False))
                mask_struct.append(mask_matrix.astype(np.int32, copy=False))
        print('\n')
total_L_sec = stop/(10.0 ** 7)
total_N = total_L_sec/window_step
print('\n')
print('total number of mini-batches:', n_batch_tot)
obj = [data_struct, label_struct, mask_struct]
target_name = os.path.join(target_path, name_var + '.pickle.gz')
####### Write Pickle file
#print('writing pickle file:', np.shape(obj))
cPickle.dump(obj, gzip.open(target_name,'wb'),cPickle.HIGHEST_PROTOCOL)

string = '\n======= data description:\n'
log.write(string)
for class_name, class_value in label_dic.items():
    try:
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
    except:
        continue
log.close()
