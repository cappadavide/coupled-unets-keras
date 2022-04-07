import tensorflow as tf
import keras.backend as K


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
    """
    apply weights on heatmap mse loss to only pick valid keypoint heatmap
    since y_true would be gt_heatmap with shape
    (batch_size, heatmap_size[0], heatmap_size[1], num_keypoints)
    we sum up the heatmap for each keypoints and check. Sum for invalid
    keypoint would be 0, so we can get a keypoint weights tensor with shape
    (batch_size, 1, 1, num_keypoints)
    and multiply to loss
    """
    heatmap_sum = K.sum(K.sum(y_true, axis=1, keepdims=True), axis=2, keepdims=True)

    keypoint_weights = 1.0 - K.cast(K.equal(heatmap_sum, 0.0), 'float32')

    return K.sqrt(K.mean(K.square((y_true - y_pred) * keypoint_weights)))

def smooth_l1_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), 'float32')
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

    return loss

def scheduler(epoch, lr):
  if epoch < 30:
    return lr
  elif epoch in [60,90]:
    return lr * 0.5
  elif epoch == 30:
    return lr * 0.2
  else:
    return lr


parts = {'mpii':['rank', 'rkne', 'rhip',
                 'lhip', 'lkne', 'lank',
                 'pelv', 'thrx', 'neck', 'head',
                 'rwri', 'relb', 'rsho',
                 'lsho', 'lelb', 'lwri']}

flipped_parts = {'mpii':[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

part_pairs = {'mpii':[[0, 5], [1, 4], [2, 3], [6], [7], [8], [9], [10, 15], [11, 14], [12, 13]]}

pair_names = {'mpii':['ankle', 'knee', 'hip', 'pelvis', 'thorax', 'neck', 'head', 'wrist', 'elbow', 'shoulder']}