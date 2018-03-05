import xml.etree.ElementTree as ET
import numpy as np


class read_xml():
    def __init__(self, batch_size, classes_count, letter_count=8):
        self.batch_size = batch_size
        self.classes_count = classes_count
        self.letter_count = letter_count
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
    def plate2label(self, plate, label_dict):
        one_hot_label = np.zeros((1, self.letter_count, self.classes_count))
        for i, s in enumerate(plate):
            one_hot_label[0, i, int(label_dict[s])] = 1
        # print(one_hot_label)
        return one_hot_label

    def label2plate(self, one_hot_label, str_dict):

        for i in range(self.batch_size):
            label = np.argmax(one_hot_label[i], axis=1)

        for l in label:
            print(str_dict[str(l)], end='')

rx = read_xml(1, 89, 8)
label_dict, str_dict = rx.read_file('label_name.xml')
one_hot_label = rx.plate2label('京A2354UH', label_dict)
label = rx.label2plate(one_hot_label, str_dict)
