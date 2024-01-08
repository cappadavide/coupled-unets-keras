def getModelParameters():
    modelParams = {}
    modelParams["m"] = 64
    modelParams["n"] = 16
    modelParams["nUNet"] = 4
    modelParams["nBlocks"] = 4
    modelParams["interSupervisions"] = [1,2]
    modelParams["shape"] = (128,128,3)

    return modelParams

def getTrainingParams():
    trainParams = {}
    trainParams["batch_size"] = 16
    trainParams["epochs"] = 201
    trainParams["lr"] = 6.7e-3
    trainParams["opt"] = "adam"
    trainParams["loss"] = "weighted"

