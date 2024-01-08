import numpy as np
import cv2

parts = {'mpii':['rank', 'rkne', 'rhip',
                 'lhip', 'lkne', 'lank',
                 'pelv', 'thrx', 'neck', 'head',
                 'rwri', 'relb', 'rsho',
                 'lsho', 'lelb', 'lwri']}

flipped_parts = {'mpii':[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

part_pairs = {'mpii':[[0, 5], [1, 4], [2, 3], [6], [7], [8], [9], [10, 15], [11, 14], [12, 13]]}

pair_names = {'mpii':['ankle', 'knee', 'hip', 'pelvis', 'thorax', 'neck', 'head', 'wrist', 'elbow', 'shoulder']}



def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)


def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    return cv2.resize(new_img, res)


def inv_mat(mat):
    ans = np.linalg.pinv(np.array(mat).tolist() + [[0, 0, 1]])
    return ans[:2]


def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot(np.concatenate((kpt, kpt[:, 0:1] * 0 + 1), axis=1), mat.T).reshape(shape)


def resize(im, res):
    return np.array([cv2.resize(im[i], res) for i in range(im.shape[0])])


def generateHeatmap(keypoints, output_res, num_parts):
    # Init
    sigma = output_res / 64
    # size = 6*sigma+3
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 2 * sigma + 1, 2 * sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / ((3 * sigma) ** 2))
    # Generation
    hms = np.zeros(shape=(num_parts, output_res, output_res),
                   dtype=np.float32)  # crea vettore (16,64,64), cioè 16 heatmaps nere
    for p in keypoints:
        for idx, pt in enumerate(p):  # ottiene id + [x,y] di ogni keypoint
            x, y = int(pt[0]), int(pt[1])
            if x <= 0 or y <= 0 or x >= output_res or y >= output_res:  # allora rimane heatmap idx-esima  tutta a 0
                continue
            # ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
            # br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

            ul = int(x - 2 * sigma - 1), int(y - 2 * sigma - 1)
            br = int(x + 2 * sigma + 2), int(y + 2 * sigma + 2)

            c, d = max(0, -ul[0]), min(br[0], output_res) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], output_res) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], output_res)
            aa, bb = max(0, ul[1]), min(br[1], output_res)
            hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], g[a:b, c:d])
    return hms


def getImgHms(img, c, s, keypoints, inp_res=(128, 128), out_res=(64, 64)):
    cropped = crop(img, c, s, inp_res)
    orig_keypoints = []
    for i in keypoints:
        orig_keypoints.append(np.array([i["x"], i["y"]]))
    orig_keypoints = np.array(orig_keypoints).reshape((1, 16, 2))
    kptmp = np.copy(orig_keypoints)
    for i in range(orig_keypoints.shape[1]):
        if orig_keypoints[0, i, 0] > 0:
            orig_keypoints[0, i, :2] = transform(orig_keypoints[0, i, :2], c, s, inp_res)
    keypoints = np.copy(orig_keypoints)
    h, w = cropped.shape[0:2]
    center = np.array((w / 2, h / 2))
    scale = max(h, w) / 200
    aug_rot = (np.random.random() * 2 - 1) * 30
    aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
    scale *= aug_scale

    mat_mask = get_transform(center, scale, out_res, aug_rot)[:2]
    mat = get_transform(center, scale, inp_res, aug_rot)[:2]
    inp = cv2.warpAffine(cropped, mat, inp_res).astype(np.float32) / 255

    keypoints[:, :, 0:2] = kpt_affine(keypoints[:, :, 0:2], mat_mask)


    # Flip 50% probability
    if np.random.randint(2) == 0:
        inp = inp[:, ::-1]
        keypoints = keypoints[:, flipped_parts['mpii']]
        keypoints[:, :, 0] = 64 - keypoints[:, :, 0]
        orig_keypoints = orig_keypoints[:, flipped_parts['mpii']]
        orig_keypoints[:, :, 0] = 128 - orig_keypoints[:, :, 0]


        for i in range(np.shape(orig_keypoints)[1]):
            if kptmp[0, i, 0] == 0 and kptmp[0, i, 1] == 0:
                keypoints[0, i, 0] = 0
                keypoints[0, i, 1] = 0
                orig_keypoints[0, i, 0] = 0
                orig_keypoints[0, i, 1] = 0

    heatmaps = generateHeatmap(keypoints, out_res[0], 16)
    return inp, heatmaps