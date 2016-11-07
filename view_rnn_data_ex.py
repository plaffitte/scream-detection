import cPickle, gzip
import matplotlib.pyplot as plt
toto=cPickle.load(gzip.open('/home/piero/Documents/Experiments/Real_Test/RNN/LSTM/10 Classes/1*256/data/train.pickle.gz','rb'))
train_labels=toto[1]
train_mask=toto[2]
train_data=toto[0]

plt.figure()
plt.legend("Mask series for first stream of first batch")
plt.plot(train_mask[0][0],marker='o')
plt.show()
