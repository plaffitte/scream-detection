import os
import numpy as np
import scikits.audiolab as audlab

wav_path = "/home/piero/Documents/Dev/Data_Base/wav"
basename = "1421664904712_L_WS"
if "_WS" in basename:
    basename_2 = basename[:-3]
wav_list_dir = os.path.join(wav_path, basename_2 + ".wav.split")
wav_list = os.listdir(wav_list_dir)
wav_list = [file for file in wav_list if ".wav" in file]
wav_list = sorted(wav_list)
print wav_list
split_lab_file = basename_2 + "-split.lab"
real_lab_file = open(os.path.join(os.path.dirname(wav_path), "lab/" + basename + ".lab"), 'r')
lines_2 = real_lab_file.readlines()
file = open(os.path.join(wav_list_dir, split_lab_file))
lines = file.readlines()
k = 0
index = 0.0
for i in range(len(wav_list)):
    if ".wav" in wav_list[i]:
        new_wav_name = basename_2 + str(i) + ".wav"
        with open(os.path.join(os.path.dirname(wav_path), "labels/" + new_wav_name[:-4] +"_WS.lab"), 'w') as new_lab_file:
            start = float(lines[i].split()[0])
            stop = float(lines[i].split()[1])
            while index < stop and k < len(lines_2) - 1:
                try:
                    index = float(lines_2[k].split()[0])
                    k += 1
                    new_line =[]
                    for kk in range(2):
                        new_line.append(str(float(lines_2[k].split()[kk]) - start))
                        new_line.append(" ")
                    new_line.append(lines_2[k].split()[2])
                    new_line.append(" \n")
                    new_lab_file.writelines(new_line)
                except:
                    real_lab_file.close()
            new_lab_file.close()
        os.rename(os.path.join(wav_list_dir, wav_list[i]), os.path.join(wav_list_dir, new_wav_name))
real_lab_file.close()
