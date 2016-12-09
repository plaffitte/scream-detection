###### Test PyBrain RNN #######

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SequentialDataSet

######### Global Variables #############
dim_input = 16
hidden_layers = {512, 512}
dim_output = 3

###### Building Dataset #########
ds = SequentialDataSet(dim_input, dim_output)

for i in range(len(dataset)):
    ds.addSample(data, target)

###### Building Network #########
net = buildNetwork(dim_input, 512, 512, dim_output, hiddenclass=TanhLayer, outclass=SoftmaxLayer)


