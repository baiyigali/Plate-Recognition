import xml.etree.ElementTree as ET
import numpy as np
import data_process.label as data_label

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

    # Input：京A1R93TX
    # Output： one_hot_label =
    def plate2label(self, plate):
        one_hot_label = np.zeros((1, 1, self.num_digit, self.num_classes))
        for i, s in enumerate(plate):
            one_hot_label[0, 0, i, int(data_label.class_dict[s])] = 1
        # print(one_hot_label)
        return one_hot_label

    #Input: [1, self.num_digit, 1, self.num_classes]
    #Output: 京A1R93TX
    def label2plate(self, one_hot_label):
        label = np.argmax(one_hot_label, axis=3)
        print(label.shape)
        plate_char = []
        for l in label[0, :, 0]:
            plate_char.append(data_label.class_char[l])
        return ''.join(plate_char)

# rx = read_xml(batch_size=1, num_digit=8, num_classes=65)
# # label_dict, str_dict = rx.read_file('./label_name.xml')
# # # print(label_dict)
# one_hot_label = rx.plate2label('京A1R93TX')
# # print(one_hot_label)
# label = rx.label2plate(one_hot_label)
# print(label)
