import sys
sys.path.append("..")
sys.path.append("../..")
import xml.etree.ElementTree as ET
import numpy as np
import data_process.label as dlabel

class read_xml():
    def __init__(self, batch_size, num_digit=8, num_classes=82):
        self.batch_size = batch_size
        self.num_digit = num_digit
        self.num_classes = num_classes
        pass

    def read_file(self, path):
        tree = ET.ElementTree(file=path)
        # root = tree.getroot()
        label_dict = {}
        str_dict = {}
        for child in tree.iter():
            xml_tag = child.tag
            if xml_tag == 'item':
                label_dict[child.text] = child.attrib['idx']
                str_dict[child.attrib['idx']] = child.text

        return label_dict, str_dict

    # Input
    # Output one_hot_label = [batch_size, letter_count=8, classes_count]
    def plate2label(self, plate):
        one_hot_label = np.zeros((1, self.num_digit, self.num_classes))
        for i, s in enumerate(plate):
            # print(label_dict[s])
            if i is 0:
                one_hot_label[0, i, int(dlabel.chinese_dict[s])] = 1
            else:
                one_hot_label[0, i, int(dlabel.letter_dict[s])] = 1

        # print(one_hot_label)
        return one_hot_label

    def label2plate(self, one_hot_label):

        y_label = np.argmax(one_hot_label[0, :, :], axis=1)

        for i, l in enumerate(y_label):
            if i is 0:
                str = dlabel.chinese_char[l]
            else:
                str += dlabel.letter_char[l]
        print(str)

# rx = read_xml(batch_size=1, num_digit=8, num_classes=34)
# # label_dict, str_dict = rx.read_file('./label_name.xml')
# # # print(label_dict)
# one_hot_label = rx.plate2label('京A1R93TX') # 输入: '京A2354UH' 输出: [1, 8, 80]
# # print(one_hot_label)
# label = rx.label2plate(one_hot_label) # 输入: [1, num_digit, num_classes] 输出: '京A1R93TX'
