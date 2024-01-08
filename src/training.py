from customLoss import *
from customCallbacks import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
import pickle
import setGPU

def train(net,modelParams,trainParams):
    lr = trainParams["lr"]

    if trainParams["opt"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=lr, rho=0.99, momentum=0.0, epsilon=1e-08
        )
    if trainParams["loss"] == "weighted":
        net.compile(optimizer=optimizer, loss=weighted_mse_loss, metrics=[])
    else:
        net.compile(optimizer=optimizer, loss="mse", metrics=[])

    fileName = f"../cunet{modelParams['nUNet']}_{modelParams['m']}{modelParams['n']}_{len(modelParams['interSupervisions'])}S_log.txt"


    with open('../imgs_train_mpii128sx12_genhm.pickle', 'rb') as handle:
        train_images = pickle.load(handle)
    with open('../hms_train_mpii128sx12_genhm.pickle', 'rb') as handle:
        train_hms = pickle.load(handle)

    eval_images = np.load("../imgs_val_mpii128sx12_genhm.npz")['arr_0']
    eval_hms = np.load("../hms_val_mpii128sx12_genhm.npz")['arr_0']


    train_hms = np.transpose(train_hms, (0, 2, 3, 1))
    eval_hms2 = np.transpose(eval_hms, (0, 2, 3, 1))


    evalcallback = EvalCallBack(fileName,eval_images,eval_hms,modelParams)
    updateLR = CustomLearningRateScheduler(7,0.2,fileName=fileName)


    if len(modelParams["interSupervision"])==2:
        history = net.fit(train_images,[train_hms,train_hms,train_hms],
                          validation_data=(eval_images,eval_hms2),epochs=modelParams["epochs"], batch_size=modelParams["batch_size"],shuffle=True,verbose=1, callbacks=[evalcallback,updateLR])
    elif len(modelParams["interSupervision"])==1:
        history = net.fit(train_images, [train_hms, train_hms],
                          validation_data=(eval_images, eval_hms2), epochs=modelParams["epochs"],
                          batch_size=modelParams["batch_size"], shuffle=True, verbose=1,
                          callbacks=[evalcallback, updateLR])
    else:
        history = net.fit(train_images, train_hms,
                          validation_data=(eval_images, eval_hms2), epochs=modelParams["epochs"],
                          batch_size=modelParams["batch_size"], shuffle=True, verbose=1,
                          callbacks=[evalcallback, updateLR])

    net.save_weights(f"../cunet{modelParams['nUNet']}_{modelParams['m']}{modelParams['n']}_{len(modelParams['interSupervisions'])}S_last.h5", overwrite=True)

    np.save(f"../history_cunet{modelParams['nUNet']}_{modelParams['m']}{modelParams['n']}_{len(modelParams['interSupervisions'])}S.npy", history.history)