import os


def calculate_word_number(file_path):
    word_id_map = {}
    word_counts = 0
    with open(file_path) as fp:
        train_data = fp.readlines()
    for line in train_data:
        words = line.split(' ')

        for item in words:
            ele = item.split('/')
            if len(ele) > 2:
                for i in range(len(ele) - 2):
                    word += ele[i] + '/'
                word = word + ele[len(ele) - 2]
            else:
                word = ele[0]
            word_counts += 1
            if word not in word_id_map:
                word_id_map[word] = len(word_id_map)

    return word_counts, len(word_id_map)


if __name__ == "__main__":
    Curr_Path = os.getcwd()
    Data_Path = os.path.join(os.path.abspath(os.path.dirname(Curr_Path)), "data")
    Train_File = os.path.join(Data_Path, "sents.train")
    Test_File = os.path.join(Data_Path, "sents.test")

    train_word_num, train_word_class = calculate_word_number(Train_File)
    print("Training Dataset has {} words and {} classes.".format(train_word_num, train_word_class))

    test_word_num, test_word_class = calculate_word_number(Test_File)
    print("Test Dataset has {} words and {} classes.".format(test_word_num, test_word_class))