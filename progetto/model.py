import math
import tensorflow as tf
from keras.models import *
from keras.layers import *



nets = {}
nets["layers"] = {}
nets["layers"]["bn"] = []

def getUnit1(layerPrec, filters, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'):
    l = BatchNormalization()(layerPrec)
    l = Activation('relu')(l)
    l = Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, padding='same', use_bias=False)(l)
    return l


def getUnit2(layerPrec, filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'):
    l = BatchNormalization()(layerPrec)
    l = Activation('relu')(l)
    l = Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, padding='same', use_bias=False)(l)
    return l


def getDownBlock(layerPrec, m, n, indexBlock):
    l = getUnit1(layerPrec, 4 * n)
    l = getUnit2(l, n)
    nets["layers"][f"down{indexBlock}"].append(l)

    concat = Concatenate()([layerPrec, l])
    l = getUnit1(concat, m)
    maxPooling = MaxPool2D(padding='same')(l)
    return maxPooling, getUnit1(concat, m)


def getUpBlock(layerPrec, skipConn, m, n, indexBlock, upLayers=[]):
    l = getUnit1(layerPrec, m)
    l = UpSampling2D()(layerPrec)
    concat = Concatenate()([skipConn, l] + upLayers)
    l = getUnit1(concat, 4 * n)
    l = getUnit2(l, n)
    nets["layers"][f"up{indexBlock}"].append(l)
    concat = Concatenate()([concat, l])
    return concat


def getUNet(input, m, n, indexUNet, nUNet, nBlocks):
    layerPrec = input
    listSkipConn = []

    if indexUNet != 0:
        layerPrec = nets[f"unet{indexUNet - 1}"]
        # layerPrec = Concatenate()([input,layerPrec]) #l'abbiamo fatto giù con l'if dopo l'up
        layerPrec = getUnit1(layerPrec, m)

    # down
    for i in range(nBlocks):
        if nets["layers"][f"down{i}"]:
            layerPrec = Concatenate()([layerPrec] + nets["layers"][f"down{i}"])
        layerPrec, skipConn = getDownBlock(layerPrec, m, n, i)
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
        layerPrec = getUpBlock(layerPrec, listSkipConn[-(i + 1)], m, n, i, upLayers=nets["layers"][f"up{i}"])

    if indexUNet != nUNet - 1:
        l = Concatenate()([input, layerPrec])
    else:
        l = getUnit1(layerPrec, 16, activation="linear")

    return l


def trasformationInput(x, filters):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=(7, 7), strides=(2, 2), kernel_initializer="he_normal", padding='same',
               use_bias=False)(x)
    maxPooling = MaxPool2D(padding='same')(x)
    return maxPooling


def getCUNet(shape, m, n, nUNet, nBlocks):
    for i in range(nUNet):
        nets[f"unet{i}"] = None

    for j in range(nBlocks):
        nets["layers"][f"down{j}"] = []
        nets["layers"][f"up{j}"] = []

    input = Input(shape=shape)

    t_input = trasformationInput(input, m)  # per le heatmap da 64x64

    for i in range(nUNet):
        nets[f"unet{i}"] = getUNet(t_input, m, n, i, nUNet, nBlocks)

    output = nets[f"unet{nUNet - 1}"]
    net = Model(inputs=input, outputs=output)
    layers = net.layers
    for i in range(len(layers)):
        if isinstance(layers[i], tf.python.keras.layers.convolutional.Conv2D):
            if isinstance(layers[i - 1].input, list):
                in_chan = 0
                for i in layers[i - 1].input:
                    in_chan = in_chan + i.shape[-1]
                print(in_chan)
            else:
                in_chan = layers[i - 1].input.shape[-1]
                n1 = layers[i].kernel_size[0] * layers[i].kernel_size[1] * in_chan
                stdv = 1 / math.sqrt(n1)
                layers[i].kernel_initializer = tf.keras.initializers.RandomUniform(minval=-stdv, maxval=stdv)

    return net