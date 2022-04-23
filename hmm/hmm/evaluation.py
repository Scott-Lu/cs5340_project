import os
import numpy as np
import itertools
from matplotlib import pyplot as plt


class Confusion:
    def __init__(self, tag_map):
        self.tag_map_len = len(tag_map)
        self.conf_matrix = np.zeros((self.tag_map_len, self.tag_map_len))
        tag_id_map = {k: v for v, k in tag_map.items()}
        label_name = []
        for i in range(len(tag_id_map)):
            label_name.append(tag_id_map[i])
        self.label_name = label_name

    def confusion_matrix(self, ground_truth, prediction):
        for g, p in zip(ground_truth, prediction):
            self.conf_matrix[g, p] += 1
        return self.conf_matrix

    def plot_confusion_matrix(self, normalize=False, title='Confusion matrix', color_map=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Input
        - normalize : True: display the percent, False: display the count
        """
        if normalize:
            cm = np.asarray(self.conf_matrix, dtype=np.float)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            cm = np.asarray(self.conf_matrix, dtype=np.int32)

        plt.figure(figsize=(30, 10))
        plt.imshow(cm, interpolation='nearest', cmap=color_map)
        plt.title(title, size=10)
        plt.colorbar()
        tick_marks = np.arange(len(self.label_name))
        plt.xticks(tick_marks, self.label_name, rotation=90, size=6)
        plt.yticks(tick_marks, self.label_name, size=6)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", size=7, color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel('Ground Truth', size=8)
        plt.ylabel('Predictions', size=8)

        plt.show()


def evaluation(output_file, ref_file, tag_map):
    confusion = Confusion(tag_map)

    with open(output_file) as output:
        out_lines = output.readlines()

    with open(ref_file) as reference:
        ref_lines = reference.readlines()

    if len(out_lines) != len(ref_lines):
        print('Error: No. of lines in output file and reference file do not match.')
        exit(0)

    total_tags = 0
    matched_tags = 0
    for i in range(0, len(out_lines)):
        out_line = out_lines[i].strip()
        out_tags = out_line.split(' ')
        ref_line = ref_lines[i].strip()
        ref_tags = ref_line.split(' ')
        total_tags += len(ref_tags)

        for j in range(0, len(ref_tags)):
            if out_tags[j] == ref_tags[j]:
                matched_tags += 1
            out_tag = out_tags[j].split('/')
            ref_tag = ref_tags[j].split('/')

            out_tag_id = tag_map[out_tag[len(out_tag)-1]]
            ref_tag_id = tag_map[ref_tag[len(ref_tag)-1]]
            confusion.conf_matrix[ref_tag_id][out_tag_id] += 1
    print("Accuracy={}%".format("%.6f" % ((float(matched_tags) / total_tags)*100)))
    confusion.plot_confusion_matrix()


if __name__ == "__main__":
    Curr_Path = os.getcwd()
    Tag_map = {'CC': 0, 'CD': 1, 'DT': 2, 'EX': 3, 'FW': 4, 'IN': 5, 'JJ': 6, 'JJR': 7, 'JJS': 8, 'LS': 9,
            'MD': 10, 'NN': 11, 'NNP': 12, 'NNPS': 13, 'NNS': 14, 'PDT': 15, 'POS': 16, 'PRP': 17, 'PRP$': 18, 'RB': 19,
            'RBR': 20, 'RBS': 21, 'RP': 22, 'SYM': 23, 'TO': 24, 'UH': 25, 'VB': 26, 'VBD': 27, 'VBG': 28, 'VBN': 29,
            'VBP': 30, 'VBZ': 31, 'WDT': 32, 'WP': 33, 'WP$': 34, 'WRB': 35, '#': 36, '$': 37, "''": 38, '-LRB-': 39,
            '-RRB-': 40, ', ': 41, '.': 42, ':': 43, '``': 44, ',': 45}

    Data_Path = os.path.join(os.path.abspath(os.path.dirname(Curr_Path)), "data")
    Output_File = os.path.join(Data_Path, "sents.output_hmm")
    Ref_File = os.path.join(Data_Path, "sents.answer")
    evaluation(Output_File, Ref_File, Tag_map)


