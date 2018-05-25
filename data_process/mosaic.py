from skimage import img_as_float
import matplotlib.pyplot as plt
from skimage import io
import random
import numpy as np
import cv2
import os

def mosaic(image, half_patch=4):


    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out = image.copy()

    row, col, channel = image.shape

    for i in range(half_patch, row - 1 - half_patch, half_patch):
        for j in range(half_patch, col - 1 - half_patch, half_patch):
            k1 = random.random() - 0.5
            k2 = random.random() - 0.5
            m = np.floor(k1 * (half_patch * 2 + 1))
            n = np.floor(k2 * (half_patch * 2 + 1))
            h = int((i + m) % row)
            w = int((j + n) % col)

            img_out[i - half_patch:i + half_patch, j - half_patch:j + half_patch, :] = \
                image[h, w, :]
    return img_out

if __name__ == "__main__":
    # source_dir = "/home1/fsb/project/LPR/plate_dataset_new/crop_plate"
    # out_dir = "/home1/fsb/project/LPR/plate_dataset_new/crop_plate_mosaic"
    source_dir = "/home1/fsb/project/LPR/plate_dataset_new/license"
    out_dir = "/home1/fsb/project/LPR/plate_dataset_new/license_mosaic"
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    source_files = os.listdir(source_dir)

    def run(source_dir, name, out_dir):

        file_name = os.path.join(source_dir, name)
        img = cv2.imread(file_name)
        img = cv2.resize(img, ())
        img_out = mosaic(img, 4)

        plt.figure(3)
        dst_image = img_out
        alpha = 0.7
        dst_image = cv2.addWeighted(img, alpha, dst_image, 1 - alpha, 0.)
        # img_out = cv2.resize(img_out, (629, 195))
        cv2.imwrite(os.path.join(out_dir, name), dst_image)


    for i, file in enumerate(source_files):
        run(source_dir, file, out_dir)
        print(i)
        break

