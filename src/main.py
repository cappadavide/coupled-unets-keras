from models import getCUNet
from parameters import getModelParameters,getTrainingParams
from training import train

modelParams = getModelParameters()
trainParams = getTrainingParams()

net = getCUNet(modelParams["shape"],modelParams["m"],modelParams["n"],modelParams["nUNet"],
               modelParams["nBlocks"],modelParams["interSupervisions"],initializeLayers=True)

net.summary()

train(net,modelParams,trainParams)

