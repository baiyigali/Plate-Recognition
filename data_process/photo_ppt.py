import os, sys, cv2
import time
photo_dir = "/home1/fsb/project/LPR/plate_dataset_new/license"

files = os.listdir(photo_dir)
for i, file_name in enumerate(files):
    if i < 10000:
        file_path = os.path.join(photo_dir, file_name)
        print(i, file_path, file_name)
        file_image = cv2.imread(file_path)
        cv2.namedWindow("new")
        cv2.imshow("new", file_image)
        cv2.waitKey(500)
        # cv2.destroyWindow("new")
        # time.sleep(1)