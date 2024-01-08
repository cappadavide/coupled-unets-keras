import tensorflow as tf
import tensorflow.keras.backend as K

def heatmapLoss(y_true,y_pred):
    l = tf.math.square((y_pred - y_true))
    l = tf.reduce_mean(l,axis=3)
    l = tf.reduce_mean(l,axis=2)
    l = tf.reduce_mean(l,axis=1)
    return l

def euclidean_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred)))

def dice(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)

def weighted_mse_loss(y_true, y_pred):

    heatmap_sum = K.sum(K.sum(y_true, axis=1, keepdims=True), axis=2, keepdims=True)
    keypoint_weights = 1.0 - K.cast(K.equal(heatmap_sum, 0.0), 'float32')

    return K.sqrt(K.mean(K.square((y_true - y_pred) * keypoint_weights)))

def smooth_l1_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), 'float32')
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

    return loss