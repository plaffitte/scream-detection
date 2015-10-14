import numpy as np
import os
import cPickle
import gzip
import matplotlib.pyplot as plt
import scikits.audiolab as audlab
from mfcc import get_mfcc

label_dir = '/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/test_labels'
result_dir = '/home/piero/Documents/Experiments/Real_Test/MFCC/Scream vs Noise/3*512'
wav_dir = os.path.join(os.path.split(label_dir)[0], 'wav')
test_data, test_labels = cPickle.load(gzip.open(os.path.join(result_dir, 'data/test.pickle.gz'), 'rb'))
pred_file = os.path.join(result_dir, 'dnn.classify.pickle.gz')
pred_data = cPickle.load(gzip.open(pred_file, 'rb'))
prediction = (-pred_data).argsort()
predictionfinal = prediction[:, 0]
files = os.listdir(label_dir)
label_dic = {}
label_dic['Noise'] = 0
#label_dic['BG_voice'] = 1
#label_dic['Conversation'] = 1
label_dic['Shouting'] = 1
#label_dic['Scream'] = 1
#label_dic['Other'] = 1
shouting_data = []
noise_data = []
label_data = []
error_count = 0.0
noise_error = 0
shouting_error = 0
error_vector = []
window_step = 0.010
N = 10

for i in range(len(files)):
    total_length = 0.0
    total_data = 0
    count = 0
    index = 0
    save_data = 0.0
    f = open(os.path.join(label_dir, files[i]), 'r')
    lines = f.readlines()
    length = len(lines)
    lab_name = files[i]
    wave_name = os.path.join(wav_dir, lab_name[:-4] + '.wav')
    f1 = audlab.Sndfile(wave_name, 'r')
    freq = f1.samplerate
    for j in xrange(len(lines)):
        cur_line = lines[j].split()
        label = cur_line[2]
        start = int(cur_line[0])
        stop = int(cur_line[1])
        L = (stop - start) / 10.0**7
        total_length += L
        audio = f1.read_frames(freq * (L))
        data_spacing = np.floor(L*10)
        save_data += L*10 - data_spacing
        if save_data > 1:
            data_spacing += np.floor(save_data)
            save_data =  save_data % 1
        total_data += L*10
        test_length = len(label_data)
        if label in label_dic:
            mono_signal = audio  # audio[:,0]
            energy = np.sum(mono_signal**2, 0) / len(mono_signal)
            signal = mono_signal  # mono_signal/math.sqrt(energy)
            samplerate = f1.samplerate
            mfcc = get_mfcc(signal, samplerate, winstep=window_step, nfft=2048, highfreq=8000, lowfreq=10)
            if (L / window_step) < N:
                shouting_data = np.concatenate((shouting_data, np.zeros(data_spacing, dtype=int)))
                noise_data = np.concatenate((noise_data, np.zeros(data_spacing, dtype=int)))
                label_data = np.concatenate((label_data, np.zeros(data_spacing, dtype=int)))
                error_vector = np.concatenate((error_vector, np.zeros(data_spacing, dtype=int)))
            else:
                zeros_to_add = 0
                if label == "Noise":
                    error_vector = np.concatenate((error_vector, (test_labels[index:index + data_spacing] -
                                                                  predictionfinal[index:index + data_spacing])))
                    shouting_data = np.concatenate((shouting_data, np.zeros(data_spacing, dtype=int)))
                    noise_data = np.concatenate((noise_data, np.ones(data_spacing, dtype=int)))
                    label_data = np.concatenate((label_data, np.ones(data_spacing, dtype=int)))
                    index += data_spacing - 1
                elif label == "Shouting":
                    error_vector = np.concatenate((error_vector, (test_labels[index:index + data_spacing] -
                                                                  predictionfinal[index:index + data_spacing])))
                    shouting_data = np.concatenate((shouting_data, np.ones(data_spacing, dtype=int)))
                    noise_data = np.concatenate((noise_data, np.zeros(data_spacing, dtype=int)))
                    label_data = np.concatenate((label_data, 2 * np.ones(data_spacing, dtype=int)))
                    index += data_spacing - 1
                # count += (L - len(mfcc) * window_step)
                # count += (len(mfcc) - (np.floor(len(mfcc) / N) * N)) * window_step
                # if count > 0.10:
                #     zeros_to_add += np.floor(count/0.10)
                #     count = count % 0.10
                #     shouting_data = np.concatenate((shouting_data, np.zeros(zeros_to_add, dtype=int)))
                #     noise_data = np.concatenate((noise_data, np.zeros(zeros_to_add, dtype=int)))
                #     label_data = np.concatenate((label_data, np.zeros(zeros_to_add, dtype=int)))
                #     error_vector = np.concatenate((error_vector, np.zeros(zeros_to_add, dtype=int)))
        else:
            shouting_data = np.concatenate((shouting_data, np.zeros(data_spacing, dtype=int)))
            noise_data = np.concatenate((noise_data, np.zeros(data_spacing, dtype=int)))
            label_data = np.concatenate((label_data, np.zeros(data_spacing, dtype=int)))
            error_vector = np.concatenate((error_vector, np.zeros(data_spacing, dtype=int)))
f.close()
f1.close()
f1 = os.path.join(result_dir, 'prediction_data.txt')
f2 = os.path.join(result_dir, 'error_data.txt')
np.savetxt(f1, label_data)
np.savetxt(f2, error_vector)
print('length label vector:', len(label_data))
print('length error vector:', len(error_vector))
print("total length of audio:", total_length)
print('total data:', total_data)
# plt.plot(error_vector)
# plt.show()
