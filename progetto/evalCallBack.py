import keras.callbacks
from customAccuracy import accuracy
from imageProcessing import *


class EvalCallBack(keras.callbacks.Callback):
    def __init__(self, model_input_shape):
        self.normalize = get_normalize(model_input_shape)
        self.model_input_shape = model_input_shape
        self.best_acc = 0.0

        self.eval_images = np.load("../imgs_val_mpii.npz")['arr_0']
        self.eval_hms = np.load("../hms_val_mpii.npz")['arr_0']

    def on_epoch_end(self, epoch, logs=None):
        output = self.model.predict(self.eval_images)
        output = np.transpose(output,(0,3,1,2))
        val_acc = accuracy(output,self.eval_hms)
        print('validate accuracy:\n', val_acc, '@epoch', epoch)

        if val_acc[0] > self.best_acc:
            # Save best accuray value and model checkpoint
            self.model.save(f"../modelsave/ep{epoch}_acc{val_acc[0]}.h5")
            print('Epoch {epoch:03d}: val_acc improved from {best_acc:.3f} to {val_acc:.3f}'.format(epoch=epoch+1, best_acc=self.best_acc, val_acc=val_acc[0]))# checkpoint_dir=checkpoint_dir))
            self.best_acc = val_acc[0]
        else:
            print('Epoch {epoch:03d}: val_acc did not improve from {best_acc:.3f}'.format(epoch=epoch+1, best_acc=self.best_acc))