import numpy as np
import tensorflow as tf
import evalCallBack
from model import getCUNet
from customLoss import scheduler


if __name__ == '__main__':
    shape = (256, 256, 3)
    m = 64
    n = 16
    nUNet = 2
    nBlocks = 4
    net = getCUNet(shape, m, n, nUNet, nBlocks)

    net.summary()

    train_images = np.load("../imgs_train_mpii.npz")['arr_0']
    train_hms = np.load("../hms_train_mpii.npz")['arr_0']
    train_hms = np.transpose(train_hms, (0, 2, 3, 1))  # np.reshape(train_hms,(train_hms.shape[0],64,64,16))

    evalcallback = evalCallBack.EvalCallBack((256, 256))

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=2.5e-4, rho=0.99, momentum=0.0, epsilon=1e-08
    )

    updateLR = tf.keras.callbacks.LearningRateScheduler(scheduler)

    net.compile(optimizer=optimizer, loss="mse", metrics=[])
    history = net.fit(train_images, train_hms, epochs=150, batch_size=32, shuffle=True, verbose=1, callbacks=[updateLR,evalcallback])  # use_multiprocessing=True,workers=15)#[checkpointer,updateLR])
    # np.savez_compressed("../history",history)
    net.save_weights(f'../cunet{nUNet}_{m}{n}_last.h5', overwrite=True)
    np.save('../history.npy', history.history)