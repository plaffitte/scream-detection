import cPickle
import gzip
import matplotlib.pyplot as plt

data_train, label_train = cPickle.load(gzip.open('/home/piero/Documents/Experiments/Real Test/Noise vs Shout/Split database/3*512_NoNormalization/train.pickle','rb'))
data_test, label_test = cPickle.load(gzip.open('/home/piero/Documents/Experiments/Real Test/Noise vs Shout/Split database/3*512_NoNormalization/test.pickle','rb'))
data_train_noise = data_train[label_train==0,:]
#data_train_conversation = data_train[label_train==1,:]
data_train_shout = data_train[label_train==1,:]
data_test_noise = data_test[label_test==0,:]
#data_test_conversation = data_test[label_test==1,:]
data_test_shout = data_test[label_test==1,:]
axe1 = 2
axe2 = 3

plt.figure(1)
ax1 = plt.subplot(211)
plt.plot(data_train_noise[:,axe1],data_train_noise[:,axe2],'.b')
plt.plot(data_test_noise[:,axe1],data_test_noise[:,axe2],'.r')
plt.title('Noise: Train (blue) vs Test(red)')
plt.xlabel('1st MFCC Coef')
plt.ylabel('2nd MFCC Coef')
plt.setp(ax1.get_xticklabels(), fontsize=6)

ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
plt.plot(data_train_shout[:,axe1],data_train_shout[:,axe2],'.b')
plt.plot(data_test_shout[:,axe1],data_test_shout[:,axe2],'.r')
plt.title('Shout: Train (blue) vs Test(red)')
plt.xlabel('1st MFCC Coef')
plt.ylabel('2nd MFCC Coef')
plt.show()

plt.figure(2)
plt.plot(data_train_noise[:,axe1],data_train_noise[:,axe2],'.y')
plt.plot(data_train_shout[:,axe1],data_train_shout[:,axe2],'.g')
plt.title('train database: Noise(yellow) vs Shout(green)')
plt.xlabel('1st MFCC Coef')
plt.ylabel('2nd MFCC Coef')
plt.show()
