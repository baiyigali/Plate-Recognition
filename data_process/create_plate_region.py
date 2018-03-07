# encoding=utf-8
import glob
import cv2
import os
import concurrent.futures

# 创建车牌位置的标签数据集
class create_region():
    def __init__(self):
        pass

    # 选取图像区域
    def cut_image(self, image, crop):
        row, col, _ = image.shape
        crop_img = image[crop[1]: crop[3], crop[0]: crop[2]]
        # self.show_image(crop_img)
        return crop_img

    # 生成位置坐标 一个正样本 一个负样本
    def create_coord(self, image, plate):
        image_h, image_w, _ = image.shape
        plate_h, plate_w, _ = plate.shape
        plate = cv2.resize(plate, (plate_h / 2., plate_w / 2.))



    def set_logo(self, bg_image, logo, coord):
        pose_letter_h, pose_letter_w, _ = logo.shape
        pose_coord = coord
        bg_image[
        int(pose_coord[0]): int(pose_coord[0] + pose_letter_h),
        int(pose_coord[1]): int(pose_coord[1] + pose_letter_w),
        :
        ] = logo[:, :, 0:3]
        return bg_image

    def save_image(self, image, folder, name):
        if not folder is None:
            if not os.path.exists(folder):
                os.makedirs(folder)
            path = os.path.join(folder, name)
        else:
            path = name
        cv2.imwrite(path, image)
        print("save image {}".format(path))

    def show_image(self, image):
        cv2.namedWindow('demo')
        cv2.imshow('demo', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# def do(file_name):
#     cr = create_region()
#     folder = 'E:\\workspace\\github\\plate_dataset\\car_image'
#     try:
#         image = cv2.imread(file_name)
#         row, col, _ = image.shape
#     except:
#         print("{} is broken".format(file_name))
#         return
#     d_x = int(col * 0.2)
#     d_y = int(row * 0.2)
#     crop = (d_x, d_y, col - d_x, row - d_y)
#     crop_image = cr.cut_image(image, crop)
#     cr.save_image(crop_image, None, file_name)
#     return file_name
#
#
# image_files = glob.glob('E:\\workspace\\github\\plate_dataset\\car_image\\*.jpg')
# print(image_files)
# with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
#     for t in executor.map(do, image_files):
#         print(t)

