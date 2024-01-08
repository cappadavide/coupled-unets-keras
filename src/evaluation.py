import numpy as np

def get_max_preds(heatmaps):
    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds


def calc_dists(preds, target, normalize, use_zero=False):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    normalize = normalize.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    if use_zero:
        boundary = 0
    else:
        boundary = 1
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > boundary and target[n, c, 1] > boundary:
                dists[c, n] = np.linalg.norm((preds[n, c, :] - target[n, c, :]) / normalize[n])  # axis ricavato da solo
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    # Return percentage below threshold while ignoring values with a -1

    if (dists != -1).sum() > 0:

        return ((dists <= thr) == (dists != -1)).sum().astype(np.float32) / (dists != -1).sum().astype(np.float32)

    else:

        return -1


def accuracy(output, target, thr=0.5):

    idkp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    preds = get_max_preds(output)
    gts = get_max_preds(target)
    norm = np.ones(preds.shape[0]) * output.shape[3] / 10

    dists = calc_dists(preds, gts, norm)

    acc = np.zeros(len(idkp) + 1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idkp)):
        acc[i + 1] = dist_acc(dists[idkp[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt

    return acc