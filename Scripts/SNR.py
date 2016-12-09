# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:55:13 2015

@author: piero
"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scikits.audiolab as audlab

f = audlab.Sndfile('/home/piero/Documents/Speech_databases/DeGIV/29-30-Jan/wav/1421664904712_L.wav','r')
fs = f.samplerate
shout_in_noise = []
shout_in_standby = []
noise = []
standby = []
L_shoutNoise = 0
L_shoutStandby = 0
L_noise = 0 
L_standby = 0
f.read_frames(58.229*fs)
L_standby = 59.194*fs-58.229*fs
standby = np.concatenate((standby,f.read_frames(L_standby)))
f.read_frames((18*60+25)*fs-59.194*fs)
L_standby = (18*60+30)*fs-(18*60+25)*fs
standby = np.concatenate((standby,f.read_frames(L_standby)))
f.read_frames((19*60+12)*fs-(18*60+30)*fs)
L_noise = (19*60+20)*fs-(19*60+12)*fs
noise = np.concatenate((noise,f.read_frames(L_noise)))
f.read_frames((24*60+33.594)*fs-(19*60+20)*fs)
L_shoutStandby = (24*60+34.612)*fs-(24*60+33.594)*fs
shout_in_standby = np.concatenate((shout_in_standby,f.read_frames(L_shoutStandby)))
f.read_frames((24*60+34.776)*fs-(24*60+34.612)*fs)
L_shoutStandby = (24*60+35.885)*fs-(24*60+34.776)*fs
shout_in_standby = np.concatenate((shout_in_standby,f.read_frames(L_shoutStandby)))
f.read_frames((24*60+36.391)*fs-(24*60+35.885)*fs)
L_shoutStandby = (24*60+37.237)*fs-(24*60+36.391)*fs
shout_in_standby = np.concatenate((shout_in_standby,f.read_frames(L_shoutStandby)))
f.read_frames((24*60+37.391)*fs-(24*60+37.237)*fs)
L_shoutStandby = (24*60+37.712)*fs-(24*60+37.391)*fs
shout_in_standby = np.concatenate((shout_in_standby,f.read_frames(L_shoutStandby)))
f.read_frames((37*60+26.856)*fs-(24*60+37.712)*fs)
L_shoutNoise = (37*60+32.330)*fs-(37*60+26.856)*fs
shout_in_noise = np.concatenate((shout_in_noise,f.read_frames(L_shoutNoise)))
f.read_frames((40*60+38.324)*fs-(37*60+32.330)*fs)
L_shoutNoise = (40*60+40.500)*fs-(40*60+38.324)*fs
shout_in_noise = np.concatenate((shout_in_noise,f.read_frames(L_shoutNoise)))
f.read_frames((43*60+11.557)*fs-(40*60+40.500)*fs)
L_shoutStandby = (43*60+14.299)*fs-(43*60+11.557)*fs
shout_in_standby = np.concatenate((shout_in_standby,f.read_frames(L_shoutStandby)))
print('L_shoutNoise:',len(shout_in_noise))
print('L_shoutStandby:',len(shout_in_standby))
print('L_noise:',len(noise))
print('L_standby:',len(standby))

plt.figure(1)
ax1 = plt.subplot(4,1,1)
plt.title('Noise in standby')
plt.plot(standby)
plt.subplot(412,sharey=ax1)
plt.title('Noise in motion')
plt.plot(noise)
plt.subplot(413,sharey=ax1)
plt.title('Scream in standby')
plt.plot(shout_in_standby)
plt.subplot(414,sharey=ax1)
plt.title('Scream in motion')
plt.plot(shout_in_noise)
plt.show()

#power = np.sum(standby**2)
#power_standby = power/len(standby)
#power = np.sum(noise**2)
#power_noise = power/len(noise)
#power = np.sum(shout_in_standby**2)
#power_shout_in_standby = power/len(shout_in_standby)
#power = np.sum(shout_in_noise**2)
#power_shout_in_noise = power/len(shout_in_noise)

N = np.floor(len(standby)/(0.025*fs))
power_standby = np.zeros(N)
for i in range(int(N)):
    frame = standby[i*0.01*fs:i*0.01*fs+0.025*fs]
    power = np.sum(frame**2)
    power_standby[i] = power/len(frame)
#power_standby = np.mean(power_standby)

N = np.floor(len(noise)/(0.025*fs))
power_noise = np.zeros(N)
for i in range(int(N)):
    frame = noise[i*0.01*fs:i*0.01*fs+0.025*fs]
    power = np.sum(frame**2)
    power_noise[i] = power/len(frame)
#power_noise = np.mean(power_noise)

N = np.floor(len(shout_in_standby)/(0.025*fs))
power_shout_in_standby = np.zeros(N)
for i in range(int(N)):
    frame = shout_in_standby[i*0.01*fs:i*0.01*fs+0.025*fs]
    power = np.sum(frame**2)
    power_shout_in_standby[i] = power/len(frame)
#power_shout_in_standby = np.mean(power_shout_in_standby)

snr1 = []
for i in range(len(power_noise)):
    interm = 10*np.log(power_noise[i]/power_standby)
    snr1 = np.concatenate((snr1,interm))
    
snr2 = []
for i in range(len(power_shout_in_standby)):
    interm = 10*np.log(power_shout_in_standby[i]/power_standby)
    snr2 = np.concatenate((snr2,interm))
    
snr3 = []
for i in range(len(power_shout_in_standby)):
    interm = 10*np.log(power_shout_in_standby[i]/power_noise)
    snr3 = np.concatenate((snr3,interm))

pl.subplot(3,1,1)
pl.hist(snr1,color='y',bins=100)
pl.xlabel('Power Noise over Power Standby in dB')
pl.subplot(3,1,2)
pl.hist(snr2,color='b',bins=100)
pl.xlabel('Power Shouts in standby / Power Standby in dB')
pl.subplot(3,1,3)
pl.hist(snr3,color='g',bins=100)
pl.xlabel('Power Shouts in standby / Power Noise in dB')
pl.show()
#snr1 = 10*np.log(power_noise/power_standby)
#snr2 = 10*np.log((power_shout_in_standby)/power_standby)
#snr3 = 10*np.log(power_shout_in_standby/power_noise)

#print(snr1,snr2,snr3)