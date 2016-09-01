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
exp_path = arguments['exp_path']
occurrence_threshold = int(arguments['threshold'])
max_seq_length = int(arguments['max_seq_len'])
n_stream = int(arguments['n_stream'])
max_batch_len = int(arguments['max_batch_len'])
N_classes = len(classes)
label_dic = {}
for k in range(N_classes):
    label_dic[classes[k]] = k
print('Classes: ', label_dic)
shutil.copyfile('/home/piero/Documents/Scripts/format_data_rnn_yun.py',
                os.path.join(exp_path,'format_data_rnn_yun_copy.py'))

initial_path = '/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/'+name_var+'_labels' # label files
target_path = os.path.join(exp_path,'data')
os.chdir(initial_path)
cur_dir = os.getcwd()
file_list = os.listdir(cur_dir)
wav_dir = os.path.join(os.path.split(initial_path)[0], 'wav')
data_struct = []
label_struct = []
mask_struct = []
time_per_occurrence_class = [[] for i in range(N_classes)]
logfile = os.path.join(target_path, 'data_log_'+name_var+'.log')
log = open(logfile, 'w')
trim = False
zero_pad = False
n_batch_tot = 0
re_use = False
restart = True
listen = False
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
    lab_name = file_list[i] # os.path.split(os.path.join(wav_dir,file_list[i]))[1]
    print(lab_name)
    line_index = 0
    if '~' in lab_name:
        print('wrong lab file')
        continue
    with open(os.path.join(cur_dir, lab_name), 'r') as f:
        lines = f.readlines()
        if 'WS' in lab_name:
            wave_name = os.path.join(wav_dir, lab_name[:-7]+'.wav')
        else:
            wave_name = os.path.join(wav_dir, lab_name[:-4]+'.wav')
        f = audlab.Sndfile(wave_name, 'r')
        freq = f.samplerate
        ind_start = 0
        ind_end = 0
        while end_file != True:
            print('-------------------> Creating new batch')
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
                # label_vector = np.zeros(1).astype(np.int32)
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
                            # print('filling up with rest of previous data')
                        else:
                            # print('reading new data')
                            audio = f.read_frames(np.floor(freq * length))
                            time_per_occurrence_class[label_dic[label]].append(length)
                        if label in label_dic:
                            # print('This is a:', label)
                            if listen:
                                if not re_use:
                                    audlab.play(audio, freq)
                                    time.sleep(1)
                            signal = audio  # audio/math.sqrt(energy)
                            n_occurrence = np.sum(time_per_occurrence_class[label_dic[label]])
                            if n_occurrence < occurrence_threshold:
                                # print('-->extracting features')
                                data, labels = create_data(signal, label)
                                if plot_detail:
                                    im, ax = plt.subplots()
                                    ax.imshow(data, aspect='auto')
                                length = len(labels)
                                # print('>length of new data:', length)
                                # If current data longer than maximum stream lentgh
                                if len(labels) > max_batch_len:
                                    # If data_vector is empty then proceed in chopping current data and placing it into data vector
                                    # print('length data vector:', len(data_vector))
                                    if len(data_vector) == 501 or len(data_vector) == 1:
                                        # print('>very long sequence, will trim')
                                        if restart:
                                            ind_start = 0
                                            ind_end = max_batch_len - 1
                                            restart = False
                                        data = data[ind_start: ind_end, :]
                                        labels = labels[ind_start:ind_end, :]
                                        # labels = labels[ind_start:ind_end]
                                        remaining_data = min(max_batch_len - 1, length - ind_end)
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
                                        # print('>padding with zeros to match length of maximum stream in batch')
                                        zero_pad()
                                        stream_full = True
                                        re_use = True
                                else:
                                    # Data shorter than max_stream_len -> check if fits in current stream
                                    if (len(data_vector) + len(labels)) > max_batch_len:
                                        # Data doesn't fit in current stream, zero pad current stream and put data in next stream
                                        # print('>padding with zeros to match length of maximum stream in batch')
                                        zero_pad()
                                        re_use = True
                                    else:
                                        # Data fits in current stream
                                        # print('>data fits current stream')
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
                                            # print('current data exceeds max sequence length, chopping')
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
                                # print('-->length of data_vector at the end of the loop:', data_vector.shape)
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
                if plot:
                    print('This is data_vector at the end')
                    plt.imshow(data_vector[1:, :].T, aspect='auto')
                    plt.plot(label_vector[1:, 2])
                    plt.show()
                data_tensor[kk, :, :] = data_vector[1:, :]
                label_tensor[kk, :, :] = label_vector[1:, :]
                # label_tensor[kk, :] = label_vector[1:]
                mask_matrix[kk, :] = mask_vector[1:]
                if end_file == True:
                    break
            if stream_cnt == n_stream:
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
print('writing pickle file:', np.shape(obj))
cPickle.dump(obj, gzip.open(target_name,'wb'),cPickle.HIGHEST_PROTOCOL)
# n = 0
# for k in range(n_batch_tot):
#     for kk in range(n_stream):
#         obj = [data_struct[k][kk, :, :], label_struct[k][kk, :], mask_struct[k][kk, :]]
#         target_name = os.path.join(target_path, str(n) + name_var + '.pickle.gz')
#         cPickle.dump(obj, gzip.open(target_name, 'wb'),cPickle.HIGHEST_PROTOCOL)
#         n += 1

for class_name, class_value in label_dic.items():
    string = 'Name of corresponding wav file:'+wave_name+'\n'
    string += 'number of data from class' + class_name + ':' + str(len(time_per_occurrence_class[class_value]))+'\n'
    string += 'length of smallest data from class:' + class_name + ':' + str(min(time_per_occurrence_class[class_value]))+'\n'
    string += 'length of longest data from class:' + class_name + ':' + str(max(time_per_occurrence_class[class_value]))+'\n'
    string += 'mean length of data from class:' + class_name + ':' + str(np.mean(time_per_occurrence_class[class_value]))+'\n'
    string += 'total length of data from class:' + class_name + ':' + str(np.sum(time_per_occurrence_class[class_value]))+'\n'
    print(string)
    log.write(string)
log.close()
