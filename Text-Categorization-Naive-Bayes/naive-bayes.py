from math import log
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')

"""
This function reads a file for a given folder with its name
and return a list of strings of all the words in the document
"""
def return_words(folder, file_name):
    address = folder + '/' + str(file_name) + '.clean'
    file = open(address,'r')
    words = file.read().split()
    return words

"""
This function reads the label files (index_train and index_test)
"""
def return_class(folder, file_name, size):
    dir = folder + '/' + file_name
    file = open(dir, 'r')
    l = file.read().split("\n")
    if "" in l:
        l.remove("")

    N = int(size * len(l))
    label = {}
    label_pos = {}
    label_neg = {}
    for line in l[0:N]:
        [key, val] = line.split("|")[0:2]
        label[int(key)] = val
        if val == "yes":
            label_pos[int(key)] = val
        else:
            label_neg[int(key)] = val

    return label, label_pos, label_neg

"""
This function trains our model with two types of variant
"""
def smoothing_terms(folder, size, m, train_type):
    train_labels, label_pos, label_neg = return_class(folder, "index_train", size)

    words = []
    for i in train_labels.keys():
        words += return_words(folder, i)
    all_words = set(words)

    vocab_pos = dict.fromkeys(all_words, 0)
    vocab_neg = dict.fromkeys(all_words, 0)

    if train_type == 1:
        # All "yes" class files
        for i in label_pos.keys():
            text = return_words(folder, i)
            # For all words in in "yes" class
            for word in text:
                vocab_pos[word] += 1  # (w^c), Count for each word
        # All "no" class files
        for i in label_neg.keys():
            text = return_words(folder, i)
            # For all words in in "no" class
            for word in text:
                vocab_neg[word] += 1  # (w^c), Count for each word

        # Calculating P(yes) and P(no)
        num_c_pos = sum(vocab_pos.values()) #c
        num_c_neg = sum(vocab_neg.values()) #c
        V = len(all_words) #V

        # Calculating P(word|yes) and P(word|no)
        for key in vocab_pos.keys():
            vocab_pos[key] = float(vocab_pos[key] + m) / (num_c_pos + m * V)
        for key in vocab_neg.keys():
            vocab_neg[key] = float(vocab_neg[key] + m) / (num_c_neg + m * V)

    if train_type == 2:
        for i in label_pos.keys():
            text = return_words(folder, i)
            for word in set(text):
                vocab_pos[word] += 1  # (w^c)
        for i in label_neg.keys():
            text = return_words(folder, i)
            for word in set(text):
                vocab_neg[word] += 1  # (w^c)

        num_c_pos = len(label_pos) #c
        num_c_neg = len(label_neg) #c
        V = 2 #V

        # Calculating P(word|yes) and P(word|no)
        for key in vocab_pos.keys():
            vocab_pos[key] = float(vocab_pos[key] + m) / (num_c_pos + m * V)
        for key in vocab_neg.keys():
            vocab_neg[key] = float(vocab_neg[key] + m) / (num_c_neg + m * V)

    pos_rate = float(len(label_pos)) / len(train_labels)
    neg_rate = float(len(label_neg)) / len(train_labels)

    return vocab_pos, vocab_neg, pos_rate, neg_rate

"""
This function makes the prediction on the test data
"""
def predict_class(folder, size, m, train_type):
    test_label, test_pos, test_neg = return_class(folder, "index_test", 1)

    # calculate all the terms ((w^c), c, m, v) for smoothing
    vocab_pos, vocab_neg, pos_rate, neg_rate = smoothing_terms(folder, size, m, train_type)

    predicted_labels = dict.fromkeys(test_label.keys(), "")

    if train_type == 1:
        for i in test_label.keys():
            text = return_words(folder, i)

            # Taking log of P(yes) and P(no)
            score_pos = np.log(pos_rate)
            score_neg = np.log(neg_rate)
            for word in text:
                if (word in vocab_pos) or (word in vocab_neg):
                    score_pos += np.log(vocab_pos[word])
                    score_neg += np.log(vocab_neg[word])

            # Prediction
            if score_pos > score_neg:
                predicted_labels[i] = "yes"
            else:
                predicted_labels[i] = "no"

    if train_type == 2:
        for i in test_label.keys():
            text = return_words(folder, i)

            score_pos = math.log(pos_rate)
            score_neg = math.log(neg_rate)

            # For all words in "yes" class
            for word in vocab_pos.keys():
                if word in text:
                    score_pos += np.log(vocab_pos[word])
                    score_neg += np.log(vocab_neg[word])
                else:
                    score_pos += np.log(1 - vocab_pos[word])
                    score_neg += np.log(1 - vocab_neg[word])

            # Prediction
            if score_pos > score_neg:
                predicted_labels[i] = "yes"
            else:
                predicted_labels[i] = "no"

    return test_label, predicted_labels

"""
This function compute the accuracy of our algorithm
"""
def accuracy(test_labels, predicted_labels):
    acc = 0
    for key in test_labels.keys():
        if test_labels[key] == predicted_labels[key]:
            acc += 1
    acc = float(acc) / len(test_labels)
    return acc

def main():
    print "Evaluating 1:"
    curve_ibmmac_221 = []
    curve_ibmmac_222 = []
    curve_ibmmac_223 = []
    curve_ibmmac_224 = []
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for s in sizes:
        test_labels, predicted_labels = predict_class(folder="ibmmac", size=s, m=0, train_type=1)
        acc = accuracy(test_labels, predicted_labels)
        curve_ibmmac_221.append(acc)

    for s in sizes:
        test_labels, predicted_labels = predict_class(folder="ibmmac", size=s, m=1, train_type=1)
        acc = accuracy(test_labels, predicted_labels)
        curve_ibmmac_222.append(acc)

    for s in sizes:
        test_labels, predicted_labels = predict_class(folder="ibmmac", size=s, m=0, train_type=2)
        acc = accuracy(test_labels, predicted_labels)
        curve_ibmmac_223.append(acc)

    for s in sizes:
        test_labels, predicted_labels = predict_class(folder="ibmmac", size=s, m=1, train_type=2)
        acc = accuracy(test_labels, predicted_labels)
        curve_ibmmac_224.append(acc)

    print curve_ibmmac_221
    print curve_ibmmac_222
    print curve_ibmmac_223
    print curve_ibmmac_224

    curve_sport_221 = []
    curve_sport_222 = []
    curve_sport_223 = []
    curve_sport_224 = []
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for s in sizes:
        test_labels, predicted_labels = predict_class(folder="sport", size=s, m=0, train_type=1)
        acc = accuracy(test_labels, predicted_labels)
        curve_sport_221.append(acc)

    for s in sizes:
        test_labels, predicted_labels = predict_class(folder="sport", size=s, m=1, train_type=1)
        acc = accuracy(test_labels, predicted_labels)
        curve_sport_222.append(acc)

    for s in sizes:
        test_labels, predicted_labels = predict_class(folder="sport", size=s, m=0, train_type=2)
        acc = accuracy(test_labels, predicted_labels)
        curve_sport_223.append(acc)

    for s in sizes:
        test_labels, predicted_labels = predict_class(folder="sport", size=s, m=1, train_type=2)
        acc = accuracy(test_labels, predicted_labels)
        curve_sport_224.append(acc)

    print curve_sport_221
    print curve_sport_222
    print curve_sport_223
    print curve_sport_224


    print "Evaluating 2:"
    m_vec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
             1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.]
    curve_m_ibmmac_type1 = []
    curve_m_ibmmac_type2 = []
    curve_m_sport_type1 = []
    curve_m_sport_type2 = []

    for m_i in m_vec:
        test_labels, predicted_labels = predict_class(folder="ibmmac", size=1, m=m_i, train_type=1)
        acc = accuracy(test_labels, predicted_labels)
        curve_m_ibmmac_type1.append(acc)
    for m_i in m_vec:
        test_labels, predicted_labels = predict_class(folder="ibmmac", size=1, m=m_i, train_type=2)
        acc = accuracy(test_labels, predicted_labels)
        curve_m_ibmmac_type2.append(acc)
    for m_i in m_vec:
        test_labels, predicted_labels = predict_class(folder="sport", size=1, m=m_i, train_type=1)
        acc = accuracy(test_labels, predicted_labels)
        curve_m_sport_type1.append(acc)
    for m_i in m_vec:
        test_labels, predicted_labels = predict_class(folder="sport", size=1, m=m_i, train_type=2)
        acc = accuracy(test_labels, predicted_labels)
        curve_m_sport_type2.append(acc)

    print curve_m_ibmmac_type1
    print curve_m_ibmmac_type2
    print curve_m_sport_type1
    print curve_m_sport_type2

if __name__ == "__main__":
    main()