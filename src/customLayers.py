from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def getUnit1(layerPrec, filters, kernel_size = (1, 1), activation='relu', kernel_initializer='he_normal'):
  l = BatchNormalization()(layerPrec)
  l = Activation(activation)(l)
  l = Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, padding='same',use_bias=False)(l)
  return l

def getUnit2(layerPrec, filters, kernel_size = (3, 3), activation='relu', kernel_initializer='he_normal'):
  l = BatchNormalization()(layerPrec)
  l = Activation(activation)(l)
  l = Conv2D(filters, kernel_size, kernel_initializer=kernel_initializer, padding='same',use_bias=False)(l)
  return l

def getDownBlock(layerPrec,m,n,indexBlock,nets):
  l = getUnit1(layerPrec,4*n)
  l = getUnit2(l,n)
  nets["layers"][f"down{indexBlock}"].append(l)

  concat = Concatenate()([layerPrec,l])
  l = getUnit1(concat,m)
  maxPooling = MaxPool2D(padding='same')(l)
  return maxPooling, getUnit1(concat,m)

def getUpBlock(layerPrec,skipConn,m,n,indexBlock,nets,upLayers=[]):
  l = getUnit1(layerPrec,m)
  l = UpSampling2D()(layerPrec)
  concat = Concatenate()([skipConn,l]+upLayers)
  l = getUnit1(concat,4*n)
  l = getUnit2(l,n)
  nets["layers"][f"up{indexBlock}"].append(l)
  concat = Concatenate()([concat,l])
  return concat


def trasformationInput(x, filters):
    x = Conv2D(filters * 2, kernel_size=(5, 5), strides=(1, 1), kernel_initializer="he_normal", padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer="he_normal", padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(padding='same')(x)

    return x