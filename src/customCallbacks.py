from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
import gc
from evaluation import accuracy



global_best_acc = 0.0

def get_normalize(input_shape):

    scale = float((input_shape[0] + input_shape[1]) / 2) / 256.0

    return 6.4*scale

class EvalCallBack(tf.keras.callbacks.Callback):
    def __init__(self, fileName, imgs, hms, modelParams):
        self.best_acc = 0.0
        self.fileName = fileName
        self.eval_images = imgs
        self.eval_hms = hms
        self.modelParams = modelParams
        self.listAcc = []


    def on_epoch_end(self, epoch, logs=None):
        global global_best_acc
        output = []

        # [2] dipende dal numero di supervisioni
        output.append(self.model.predict(self.eval_images[0:2407])[2])
        output.append(self.model.predict(self.eval_images[2407:4814])[2])
        output.append(self.model.predict(self.eval_images[4814:7221])[2])

        output = np.array(output)
        print(output.shape)
        output = output.reshape((7221, 64, 64, 16))
        print(output.shape)
        output = np.transpose(output, (0, 3, 1, 2))
        val_acc = accuracy(output, self.eval_hms)
        self.listAcc.append(val_acc[0])
        print('\nvalidate accuracy:\n', val_acc, '@epoch', epoch)
        f = open(self.fileName, "a")
        if val_acc[0] > self.best_acc:

            self.model.save_weights(f"../cunet{self.modelParams['nUNet']}_{self.modelParams['m']}{self.modelParams['n']}_{len(self.modelParams['interSupervisions'])}S_best.h5")
            print('Epoch {epoch:03d}: val_acc improved from {best_acc:.3f} to {val_acc:.3f}'.format(epoch=epoch + 1,
                                                                                                    best_acc=self.best_acc,
                                                                                                    val_acc=val_acc[
                                                                                                        0]))
            message = 'Epoch {epoch:03d}: val_acc improved from {best_acc:.3f} to {val_acc:.3f}. Train loss is {loss:.3f}\n'.format(
                epoch=epoch + 1, best_acc=self.best_acc, val_acc=val_acc[0], loss=logs.get('loss'))
            self.best_acc = val_acc[0]
            global_best_acc = val_acc[0]


        else:
            message = 'Epoch {epoch:03d}: val_acc did not improve from {best_acc:.3f}. Train loss is {loss:.3f}\n'.format(
                epoch=epoch + 1, best_acc=self.best_acc, loss=logs.get('loss'))
            print('Epoch {epoch:03d}: val_acc did not improve from {best_acc:.3f}'.format(epoch=epoch + 1,
                                                                                          best_acc=self.best_acc))
        f.write(message)
        f.close()
        if epoch == 200:
            np.savez_compressed(f"../cunet{self.modelParams['nUNet']}_{self.modelParams['m']}{self.modelParams['n']}_{len(self.modelParams['interSupervisions'])}S_accs",
                                self.listAcc)
        gc.collect()


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):

    def __init__(self, patience, factor, fileName=None):
        super(CustomLearningRateScheduler, self).__init__()
        self.counter = 0
        self.patience = patience
        self.fileName = fileName
        self.factor = factor
        self.best_acc = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        global global_best_acc
        print(f"Counter: {self.counter}, Global: {global_best_acc}, MyBest: {self.best_acc}\n")
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)

    def schedule(self, epoch, lr):
        global global_best_acc
        if self.counter == self.patience:
            self.counter = 0
            print("Epoch %03d: Updating Learning rate.. New value is %f" % (epoch, lr * self.factor))
            message = "Epoch %03d: Updating Learning rate.. New value is %f" % (epoch, lr * self.factor)
            f = open(self.fileName, "a")
            f.write(message)
            f.close()
            return lr * self.factor
        if self.best_acc == global_best_acc:
            self.counter = self.counter + 1
        elif self.best_acc < global_best_acc:
            self.counter = 0
            self.best_acc = global_best_acc
        return lr
