# encoding=utf-8
import numpy as np
import os
import sys
sys.path.append("..")
sys.path.append("../..")


class write_label_file():
    def __init__(self):
        pass

    def image2label(self, path):
        self.label_dict = {}
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                group = line.split(' ')
                name = group[0]
                label = group[1]
                self.label_dict[name] = label

        print(self.label_dict)

    # get all files list of the folder_path with iteration
    def gci(self, path, file_list):
        parents = os.listdir(path)
        for parent in parents:
            child = os.path.join(path, parent)
            if os.path.isdir(child):
                self.gci(child, file_list)
            else:
                str = os.path.normpath(child)
                file_list.append(str)
        return file_list

    # 只读一层目录
    def create_file(self, folder_path):
        files = os.listdir(folder_path)
        files_list = []
        for file in files:
            str = os.path.abspath(os.path.join(folder_path, file)) + ' ' + file.split('.')[0]
            files_list.append(str)
            print(str)
        return files_list

    # write file_list into txt_file
    def write_file(self, txt_file, file_list):
        with open(txt_file, 'w') as file:
            np.random.shuffle(file_list)
            for fn in file_list:
                try:
                    file.write(os.path.abspath(fn))
                    print("write path {}".format(fn))
                    file.writelines('\n')
                except:
                    print("path error {}".format(fn))

    # 将一个list按照 6:1 划分训练集和测试集
    def write_file_with_split(self, file_list, train_txt, test_txt):
        np.random.shuffle(file_list)
        with open(train_txt, 'w') as train_file:
            with open(test_txt, 'w') as test_file:
                len = file_list.__len__()
                split_num = len / 7.
                for i in range(len):
                    print(file_list[i])
                    if i < split_num:
                        test_file.writelines(file_list[i])
                        test_file.writelines('\n')
                    else:
                        train_file.write(file_list[i])
                        train_file.writelines('\n')


# folder_path = '../../plate_dataset/plate_process_image'
# # folder_path = '../../plate_dataset/plate_process_image_without_shape'
# # folder_path = '../../plate_dataset/plate_process_image_with_shear'
# # folder_path = '../../plate_dataset/license'
# #
# train_txt = '../path/plate_process_image/train.txt'
# valid_txt = '../path/plate_process_image/valid.txt'
# wl = write_label_file()
# file_list = wl.create_file(folder_path)
# wl.write_file_with_split(file_list, train_txt, valid_txt)
#
# folder_path = '../../plate_dataset/plate_process_image'
# test_txt = '../path/test.txt'
# wl = write_label_file()
# file_list = wl.create_file(folder_path)
# wl.write_file(test_txt, file_list)
