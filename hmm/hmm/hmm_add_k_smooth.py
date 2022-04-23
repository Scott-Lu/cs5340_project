import os
import math
import sys
import datetime
import numpy as np
import pickle
import evaluation as eva


PennTreebank_Tag = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS",
                    "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN",
                    "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "#", "$", "''", "-LRB-", "-RRB-", ", ", ".", ":", "``"]


class PosInfo:
    def __init__(self):
        self.word_map = {}
        self.tag_map = {}
        # first word of tag in the sentence
        self.first_tag = None
        # transition probability matrix
        self.a_matrix = None
        # observation likelihood matrix
        self.b_matrix = None
        for value in PennTreebank_Tag:
            self.tag_map[value] = len(self.tag_map)


def init() -> PosInfo:
    pos_info = PosInfo()
    return pos_info


# training process
def extract_word_and_tag_sub(item, pos_info):
    word = ''
    ele = item.split('/')
    if len(ele) > 2:
        tag = ele[len(ele) - 1].rstrip('\n')
        for i in range(len(ele) - 2):
            word += ele[i] + '/'
        word = word + ele[len(ele) - 2]
    else:
        word, tag = ele[0], ele[1].rstrip('\n')

    if word not in pos_info.word_map:
        pos_info.word_map[word] = len(pos_info.word_map)
    if tag not in pos_info.tag_map:
        pos_info.tag_map[tag] = len(pos_info.tag_map)


def calculate_row_ratio_of_matrix(row_num, matrix):
    for i in range(row_num):
        row_sum = np.sum(matrix[i])
        if row_sum != 0:
            matrix[i] = matrix[i] / row_sum

    return matrix


def extract_word_and_tag(train_data_file, pos_info):
    with open(train_data_file) as tdf:
        train_data = tdf.readlines()

    for line in train_data:
        word_by_tag = line.split(' ')

        for item in word_by_tag:
            extract_word_and_tag_sub(item, pos_info)

    # to solve unseen words
    pos_info.word_map['UNK'] = len(pos_info.word_map)


def hmm_calculate_word_tag(ele, pos_info):
    word = ''
    if len(ele) > 2:
        tag_id, tag = pos_info.tag_map[ele[len(ele) - 1].rstrip('\n')], ele[len(ele) - 1].rstrip('\n')
        for i in range(len(ele) - 2):
            word += ele[i] + '/'
        word = word + ele[len(ele) - 2]
        word_id = pos_info.word_map[word]
    else:
        word_id, tag_id, tag = pos_info.word_map[ele[0]], pos_info.tag_map[ele[1].rstrip('\n')], ele[1].rstrip('\n')

    return word_id, tag_id, tag


def smooth(matrix):
    return matrix + 0.1


def hmm(train_data_file, pos_info):
    word_num = len(pos_info.word_map)
    tag_num = len(pos_info.tag_map)

    # first word of tag in the sentence
    first_tag = np.zeros(tag_num)
    # transition probability matrix
    a_matrix = np.zeros((tag_num, tag_num))
    # observation likelihood
    b_matrix = np.zeros((tag_num, word_num))

    with open(train_data_file) as tdf:
        train_data = tdf.readlines()

    for line in train_data:
        pre_tag = ''
        word_by_tag = line.split(' ')
        for item in word_by_tag:
            ele = item.split('/')
            word_id, tag_id, tag = hmm_calculate_word_tag(ele, pos_info)
            if pre_tag == '':
                first_tag[tag_id] += 1
                b_matrix[tag_id][word_id] += 1
            else:
                a_matrix[pos_info.tag_map[pre_tag]][tag_id] += 1
                b_matrix[tag_id][word_id] += 1
            pre_tag = tag

    # a_matrix = smooth(a_matrix)
    b_matrix = smooth(b_matrix)

    pos_info.first_tag = first_tag / np.sum(first_tag)
    pos_info.a_matrix = calculate_row_ratio_of_matrix(tag_num, a_matrix)
    pos_info.b_matrix = calculate_row_ratio_of_matrix(tag_num, b_matrix)


def train_model(train_file, pos_info):

    extract_word_and_tag(train_file, pos_info)
    hmm(train_file, pos_info)

    print('Train finished...')
#  Training process


# Test process
def viterbi(ele, first_tag, a_matrix, b_matrix, word_map):
    word_id = np.zeros(len(ele), dtype=np.int32)
    for i in range(len(ele)):
        if ele[i] not in word_map:
            word_id[i] = word_map['UNK']
        else:
            word_id[i] = word_map[ele[i]]

    observation_len = len(word_id)
    state_graph_len = len(a_matrix[0])
    viterbi_matrix = np.zeros((state_graph_len, observation_len))
    back_pointer = np.zeros((state_graph_len, observation_len), dtype=np.int16)

    for i in range(state_graph_len):
        viterbi_matrix[i][0] = first_tag[i] * b_matrix[i][word_id[0]]

    for i in range(1, observation_len):
        for j in range(state_graph_len):
            for k in range(state_graph_len):
                tmp = viterbi_matrix[k][i - 1] * a_matrix[k][j] * b_matrix[j][word_id[i]]
                if tmp > viterbi_matrix[j][i]:
                    viterbi_matrix[j][i] = tmp
                    back_pointer[j][i] = k

    tag_seq = np.zeros(observation_len, dtype=np.int16)
    tag_seq[observation_len - 1] = np.argmax(viterbi_matrix[:, observation_len - 1])

    for i in range(observation_len - 2, -1, -1):
        tag_seq[i] = back_pointer[tag_seq[i + 1]][i + 1]

    return tag_seq


def extract_test_file(test_file, pos_info, out_file):
    out_str = ''

    tagid_map = {k: v for v, k in pos_info.tag_map.items()}
    with open(test_file) as tf:
        test_data = tf.readlines()

    for line in test_data:
        ele = line.split(' ')
        for i in range(len(ele)):
            ele[i] = ele[i].rstrip('\n')

        tag_seq = viterbi(ele, pos_info.first_tag, pos_info.a_matrix, pos_info.b_matrix, pos_info.word_map)
        for i in range(len(ele) - 1):
            out_str += (ele[i] + '/' + tagid_map[tag_seq[i]] + ' ')
        out_str += (ele[len(ele) - 1] + '/' + tagid_map[tag_seq[len(ele) - 1]] + '\n')

    with open(out_file, 'w') as output:
        output.write(out_str)


def test_model(test_file, model_file, out_file):
    extract_test_file(test_file, model_file, out_file)

    print('Test finished...')
# Test process


if __name__ == "__main__":
    Curr_Path = os.getcwd()
    Data_Path = os.path.join(os.path.abspath(os.path.dirname(Curr_Path)), "data")
    Train_File = os.path.join(Data_Path, "sents.train")
    Test_File = os.path.join(Data_Path, "sents.test")
    Output_File = os.path.join(Data_Path, "sents.output_hmm")

    Pos_Info = init()

    Start_Time = datetime.datetime.now()
    train_model(Train_File, Pos_Info)
    End_Time = datetime.datetime.now()
    print('Training time:', End_Time - Start_Time)

    Start_Time = datetime.datetime.now()
    test_model(Test_File, Pos_Info, Output_File)
    End_Time = datetime.datetime.now()
    print('Test time:', End_Time - Start_Time)

    Ref_File = os.path.join(Data_Path, "sents.answer")
    eva.evaluation(Output_File, Ref_File, Pos_Info.tag_map)

