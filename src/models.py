from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from customLayers import *
import tensorflow as tf


def getUNet(input, m, n, indexUNet, nUNet, nBlocks,nets,interSupervisions):
    layerPrec = input
    listSkipConn = []


    if indexUNet != 0:
        layerPrec = nets[f"unet{indexUNet - 1}"]
        layerPrec = getUnit1(layerPrec, m)

    # down
    for i in range(nBlocks):
        if nets["layers"][f"down{i}"]:
            layerPrec = Concatenate()([layerPrec] + nets["layers"][f"down{i}"])
        layerPrec, skipConn = getDownBlock(layerPrec, m, n, i,nets)
        listSkipConn.append(skipConn)

    # bottle neck
    if nets["layers"][f"bn"]:
        layerPrec = Concatenate()([layerPrec] + nets["layers"][f"bn"])

    l = getUnit1(layerPrec, 4 * n)
    l = getUnit2(l, n)
    nets["layers"]["bn"].append(l)
    concat = Concatenate()([layerPrec, l])

    # up
    layerPrec = concat
    for i in range(nBlocks):
        layerPrec = getUpBlock(layerPrec, listSkipConn[-(i + 1)], m, n, i,nets, upLayers=nets["layers"][f"up{i}"])

    if indexUNet != nUNet - 1:
        l = Concatenate()([input, layerPrec])
    else:
        l = getUnit1(layerPrec, 16, activation="sigmoid")  # era linear

    inter = []
    if indexUNet in interSupervisions:
        inter.append(getUnit1(layerPrec, 16, activation="sigmoid"))

    return l, inter


def getCUNet(shape, m, n, nUNet, nBlocks, interSupervisions,initializeLayers = False):

    nets = {}
    nets["layers"] = {}
    nets["layers"]["bn"] = []

    for i in range(nUNet):
        nets[f"unet{i}"] = None

    for j in range(nBlocks):
        nets["layers"][f"down{j}"] = []
        nets["layers"][f"up{j}"] = []

    input = Input(shape=shape)

    t_input = trasformationInput(input, m)

    output = []
    for i in range(nUNet):
        nets[f"unet{i}"], inter = getUNet(t_input, m, n, i, nUNet, nBlocks,nets,interSupervisions)
        if len(inter)>0:
            output.extend(inter)
    output.append(nets[f"unet{nUNet - 1}"])

    net = Model(inputs=input, outputs=output)

    if initializeLayers:

        layers = net.layers

        for i in range(len(layers)):
            if isinstance(layers[i], tf.python.keras.layers.convolutional.Conv2D):
                if isinstance(layers[i - 1].input, list):
                    in_chan = 0
                    for k in layers[i - 1].input:
                        in_chan = in_chan + k.shape[-1]
                    print(in_chan)
                else:
                    in_chan = layers[i - 1].input.shape[-1]

                n1 = layers[i].kernel_size[0] * layers[i].kernel_size[1] * in_chan
                stdv = 1 / math.sqrt(n1)
                layers[i].kernel_initializer = tf.keras.initializers.RandomUniform(minval=-stdv, maxval=stdv)

    return net