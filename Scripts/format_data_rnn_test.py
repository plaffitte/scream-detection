import os as os
import scikits.audiolab as audlab
import cPickle, gzip, sys
import numpy as np
import shutil
from spectral import get_mfcc
import kaldi_format_data

######################## SET  PARAMS ##############################
window_step = 0.010 # in seconds, hop size between two successive mfcc windows
window_size = 0.025 # in seconds, size of MFCC window
highfreq = 16000 # maximal analysis frequency for mfcc
lowfreq = 50# minimal analysis frequency for mfcc
size = 30 # number of mfcc coef
exp_path = "/home/piero/Documents/Speech_databases/test/RNN_audio_classification" # path which experiment is run from
threshold = 2000
max_seq_length = 10

######################## SET CLASS PARAMS ###########################
N_classes = 3
label_dic = {}
label_dic['Noise'] = 0
# label_dic['BG_voice'] = 1
label_dic['Conversation'] = 1
label_dic['Shouting'] = 2
# label_dic['Scream'] = 2
# label_dic['Other'] = 1

shutil.copyfile('/home/piero/Documents/Scripts/format_data_rnn_test.py',
                os.path.join(exp_path,'format_data_rnn_test.py'))

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--", "").replace("-", "_")
        args[key] = arg_elements[2*i+1]
    return args

######### Some global variables ###########
arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
arguments = parse_arguments(arg_elements)
name_var = arguments['data_type']
# name_var = 'test'
initial_path = '/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/'+name_var+'_labels' # label files
target_path = os.path.join(exp_path,"data")
os.chdir(initial_path)
cur_dir = os.getcwd()
file_list = os.listdir(cur_dir)
wav_dir = os.path.join(os.path.split(initial_path)[0], 'wav')
label_vector = []
data_vector = []
time_per_occurrence_class = [[] for i in range(N_classes)]
logfile = os.path.join(target_path, 'data_log_'+name_var+'.log')
log = open(logfile, 'w')

########## Callback processing function ##########""
def create_data(sig, label):
    mfcc = get_mfcc(sig, freq, winstep=window_step, winlen=window_size, nfft=2048, lowfreq=lowfreq,
                    highfreq=highfreq, numcep=size, nfilt=size+2)
    num_label = label_dic[label]*np.ones(len(mfcc))
    time_per_occurrence_class[label_dic[label]].append((stop - start) / (10.0 ** 7))
    return mfcc, num_label

####### Main Loop ##########
for i in xrange(len(file_list)):
    lab_name = file_list[i] # os.path.split(os.path.join(wav_dir,file_list[i]))[1]
    if '~' in lab_name:
        continue
    with open(os.path.join(cur_dir, file_list[i]), 'r') as f:
        lines = f.readlines()
        wave_name = os.path.join(wav_dir, lab_name[:-4]+'.wav')
        f = audlab.Sndfile(wave_name, 'r')
        freq = f.samplerate
        nframes = f.nframes
        frames_recovered = 0
        for j in xrange(len(lines)):
            try:
                cur_line = lines[j].split()
                start = int(cur_line[0])
                stop = int(cur_line[1])
                label = cur_line[2]
                length = stop / 10.0 ** 7 - start / 10.0 ** 7
                audio = f.read_frames(np.floor(freq * length))
                if label in label_dic:
                    signal = audio  # audio/math.sqrt(energy)
                    time = np.sum(time_per_occurrence_class[label_dic[label]])
                    if time < threshold:
                        one, two = create_data(signal, label)
                        if len(one) < max_seq_length:
                            label_vector.append(two.astype(np.int32, copy=False))
                            data_vector .append(one.astype(np.float32, copy=False))
                        else:
                            for j in range(int(np.floor(len(one) / max_seq_length))):
                                L = min(max_seq_length, len(one) - j * max_seq_length)
                                data_vector .append(one[j * max_seq_length : j * max_seq_length + L, :].astype(np.float32, copy=False))
                                label_vector.append(two[j * max_seq_length : j * max_seq_length + max_seq_length])#.astype(np.int32, copy=False))
            except KeyError, e:
                print "Wrong label name:", label, "at line", j+1
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise

total_L_sec = stop/(10.0 ** 7)
total_N = total_L_sec/window_step
# Now write to file, for pdnn learning
target_name = os.path.join(target_path, name_var + '.pfile')
####### Write Kaldi file
kaldi_format_data.writePfile(target_name, data_vector, label_vector)

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
