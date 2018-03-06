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
        label_dict = {}
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                group = line.split(' ')
                name = group[0]
                label = group[1]
                label_dict[name] = label

        print(label_dict)

    def gci(self, path, file_list):
        parents = os.listdir(path)
        for parent in parents:
            child = os.path.join(path, parent)
            if os.path.isdir(child):
                self.gci(child, file_list)
            else:
                str = os.path.normpath(child) + ' ' + os.path.split(os.path.dirname(child))[-1]
                # str = os.path.normpath(child) + ' ' + label

                # str = os.path.normpath('../' + child)
                file_list.append(str)
        return file_list

    def write_file(self, txt_file, file_list):
        with open(txt_file, 'w') as file:
            np.random.shuffle(file_list)
            for fn in file_list:
                try:
                    file.write(os.path.abspath(fn))
                    print("wtite path {}".format(fn))
                    file.writelines('\n')
                except:
                    print("path error {}".format(fn))

    def write_file_with_split(self, train_txt, test_txt, file_list):
        np.random.shuffle(file_list)
        with open(train_txt, 'w') as train_file:
            with open(test_txt, 'w') as test_file:
                len = file_list.__len__()
                split_num = len / 7.
                for i in range(len):
                    print(file_list[i])
                    if i < split_num:
                        test_file.write(file_list[i])
                        test_file.writelines('\n')
                    else:
                        train_file.write(file_list[i])
                        train_file.writelines('\n')


wl = write_label_file()
wl.image2label('./file_label.txt')
# file_list = []
# wl.gci('./plate_process_image', file_list)
# print(file_list)

