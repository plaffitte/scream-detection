import numpy as np
import os, glob, sys
import cPickle,gzip
import matplotlib.pyplot as plt
import scikits.audiolab as audlab
from mfcc import get_mfcc

result_dir = sys.argv[1]
keyword = sys.argv[2]
# keyword = 'control4'
label_dir = '/home/piero/Documents/test_folder/control_test_data/'+keyword
# result_dir = '/home/piero/Documents/Experiments/Real_Test/MFCC/2_classes/Scream+Shout_vs_Everything_else/3*512/26-09-2015'
wav_dir = os.path.join(label_dir, 'wav')
test_data, test_labels = cPickle.load(gzip.open(os.path.join(result_dir, keyword+'_test.pickle.gz'), 'rb'))
pred_file = os.path.join(result_dir, keyword+'_test_classify.pickle.gz')
pred_data = cPickle.load(gzip.open(pred_file, 'rb'))
prediction = (-pred_data).argsort()
predictionfinal = prediction[:, 0]
os.chdir(label_dir)
files = glob.glob('*.lab')[0]#'control_test.lab'
label_dic = {}
label_dic['Noise'] = 0
label_dic['Noise1'] = 0
label_dic['Noise2'] = 0
# label_dic['BG_voice'] = 0
# label_dic['Conversation'] = 0
label_dic['Shouting'] = 1
# label_dic['Scream'] = 1
#label_dic['Other'] = 1
label_data = []
error_count = 0.0
error_vector = []
time_vector = []
final_vector = []
window_step = 0.010
window_len = 0.025
N = 10
slide = 5
total_length = 0.0
dropped_audio = 0.0
count = 0
index = 0
save_data = 0.0
compensate_data_too_small = 0.0
f = open(os.path.join(label_dir, files), 'r')
lines = f.readlines()
length = len(lines)
os.chdir(wav_dir)
wave_name = os.path.join(wav_dir,glob.glob('*.wav')[0])
os.chdir(label_dir)
logfile = os.path.join(result_dir,keyword+'_error_visualize.log')
f1 = audlab.Sndfile(wave_name, 'r')
freq = f1.samplerate
for j in xrange(length):
    cur_line = lines[j].split()
    label = cur_line[2]
    start = int(cur_line[0])
    stop = int(cur_line[1])
    L = (stop - start) / 10.0**7
    total_length += L
    audio = f1.read_frames(freq * (L))
    if L > window_len:
       if label in label_dic:
            mono_signal = audio  # audio[:,0]
            # energy = np.sum(mono_signal**2, 0) / len(mono_signal)
            signal = mono_signal  # mono_signal/math.sqrt(energy)
            samplerate = f1.samplerate
            mfcc = get_mfcc(signal, samplerate, winstep=window_step, nfft=2048, highfreq=8000, lowfreq=10)

            start = start / (10.0**7)
            stop = stop / (10.0**7)
            time_stamp = np.arange(start + (window_len / 2 + (N - 1) * window_step) / 2, stop - slide * window_step, slide * window_step)
            data_number = len(time_stamp)
            # data_number = ((stop / 10.0**7) - (start / 10.0**7 + window_len / 2)) / window_step
            error = test_labels[index:index + data_number] - predictionfinal[index:index + data_number]
            index += data_number
            error_vector = np.append(error_vector, error)
            time_vector = np.append(time_vector, time_stamp)
    else:
        print("length of audio not long enough to get MFCC")
final_vector = np.concatenate((error_vector[:,np.newaxis], time_vector[:,np.newaxis]), 1)
error_indicator = np.any(error_vector[:,np.newaxis]!=0,axis=1)
total_error = np.sum(error_indicator)/float(len(test_labels))
f.close()
f1.close()
f2 = os.path.join(result_dir, keyword+'_label_data.txt')
f3 = os.path.join(result_dir, keyword+'_error_data.txt')
f4 = os.path.join(result_dir, keyword+'_error_indicator.txt')
f5 = os.path.join(result_dir, keyword+'_model_prediction.txt')
np.savetxt(f2, test_labels)
np.savetxt(f3, final_vector)
np.savetxt(f4, error_indicator)
np.savetxt(f5, predictionfinal)
log = open(logfile,'w')
string='final index is:'+str(index)+'\n'
string+='total error rate:'+str(total_error)+'\n'
string+='length test vector:'+str(len(test_labels))+'\n'
string+='length prediction vector:'+str(len(predictionfinal))+'\n'
string+='length error vector:'+str(len(final_vector))+'\n'
string+='total length of audio:'+str(total_length)+'\n'
string+='dropped:'+str(dropped_audio)+'\n'
log.write(string)
