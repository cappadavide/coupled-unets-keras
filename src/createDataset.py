import json
import imageio
import numpy as np
from imgProcessing import getImgHms
import pickle

path = "../images_mpii/"

f = open('datasets.json')
datasets = json.load(f)


images_mpii = []
hms_mpii = []
data = datasets['dataset']['MPII']["people"]

for p in range(len(data)):
    img = imageio.imread(f"{path}{data[p]['filepath']}")
    if np.max(img)>1:
        img = img/255
    c = [data[p]["objpos"]["x"],data[p]["objpos"]["y"]]
    s = data[p]["scale"]
    croppedImg, hms = getImgHms(img,c,s*1.2,data[p]["keypoints"])
    if np.max(croppedImg)>1:
        croppedImg = croppedImg/255
    images_mpii.append(croppedImg)
    hms_mpii.append(hms)


images_mpii =  np.array(images_mpii)
hms_mpii =  np.array(hms_mpii)

images_train_mpii = images_mpii[:-7221]
images_val_mpii = images_mpii[-7221:]
images_test_mpii = images_train_mpii[-3249:]
images_train_mpii = images_train_mpii[:-3249]

hms_train_mpii = hms_mpii[:-7221]
hms_val_mpii = hms_mpii[-7221:]
hms_test_mpii = hms_train_mpii[-3249:]
hms_train_mpii = hms_train_mpii[:-3249]

with open("../imgs_train_mpii128sx12_genhm.pickle", 'wb') as pfile:
    pickle.dump(images_train_mpii, pfile, protocol=pickle.HIGHEST_PROTOCOL)
with open("../hms_train_mpii128sx12_genhm.pickle", 'wb') as pfile:
    pickle.dump(hms_train_mpii, pfile, protocol=pickle.HIGHEST_PROTOCOL)
np.savez_compressed('../imgs_val_mpii128sx12_genhm', images_val_mpii)
np.savez_compressed('../hms_val_mpii128sx12_genhm', hms_val_mpii)
np.savez_compressed('../imgs_test_mpii128sx12_genhm', images_test_mpii)
np.savez_compressed('../hms_test_mpii128sx12_genhm', hms_test_mpii)