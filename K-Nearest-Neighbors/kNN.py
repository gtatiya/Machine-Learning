import numpy as np

"""
Reading data files
This function read arff file
"""
def read_dataset(file_name):
    file = open("./dataset/" + file_name)
    sample = []
    for line in file:
        if line != "" and line[0].isdigit():
            line = line.strip("\n")
            line = line.split(',')
            sample.append(line)
    for line in sample:
        for i in xrange(0, len(line) - 1):
            line[i] = float(line[i])
    return sample

# Implementing kNN
"""
This function split the data into data points and one column of classes
"""
def split_points_class(data):
    example = []
    classes = []
    for line in data:
        example.append(line[0:len(line) - 1])
        classes.append(line[len(line) - 1])
    return np.array(example), np.array(classes)

"""
This function performs the kNN and compute the accuracy on test data
"""
def kNN(test_data, train_data, k):

    train_points, train_classes = split_points_class(train_data)
    test_points, test_classes = split_points_class(test_data)

    test_classes_hat = []
    for point in test_points:
        distances = np.sqrt(np.sum((point - train_points) ** 2, 1))
        ind = np.argsort(distances)[:k]

        vote = dict.fromkeys(set(train_classes), 0)
        for indice in ind:
            vote[train_classes[indice]] += 1
        test_classes_hat.append(max(vote, key=vote.get))

    accuracy = (sum(np.array(test_classes_hat) == np.array(test_classes)) + 0.0) / len(test_classes)
    return accuracy

"""
This function compute the SplitGain for the ith feature
"""
def compute_gain(train_data, i):
    train_points, train_labels = split_points_class(train_data)
    labels = set(train_labels)
    N = len(train_data)

    ind = np.argsort(train_points[:, i])

    count = []
    gain = 0
    for l in labels:
        count.append(list(train_labels).count(l))
    for p in count:  # compute the gain
        pk = (p + 0.0) / sum(count)
        if pk != 1 and pk != 0:
            gain += - pk * np.log2(pk)

    for k in range(0, 5):
        ind_k = ind[k * N / 5:(k + 1) * N / 5]
        count_k = []
        gain_k = 0
        for l in labels:
            count_k.append(list(train_labels[ind_k]).count(l))
        for p in count:
            pk = (p + 0.0) / sum(count_k)
            if pk != 1 and pk != 0:
                gain_k += - pk * np.log2(pk)
        gain += - ((len(ind_k) + 0.0) / N) * gain_k
    return gain

"""
This function evalutate kNN with the top n selected features
"""
def choose_feature(train_data, test_data, n):
    split_gain = []
    for i in xrange(0, len(train_data[0]) - 1):
        split_gain.append(compute_gain(train_data, i))
    ind = np.argsort(split_gain)[-n:]

    train_points, train_labels = split_points_class(train_data)
    test_points, test_labels = split_points_class(test_data)
    train_points = train_points[:, ind]
    test_points = test_points[:, ind]

    test_labels_hat = []
    for point in test_points:
        distances = np.sqrt(np.sum((point - train_points) ** 2, 1))
        ind = np.argsort(distances)[:5]

        vote = dict.fromkeys(set(train_labels), 0)
        for indice in ind:
            vote[train_labels[indice]] += 1
        test_labels_hat.append(max(vote, key=vote.get))

    accuracy = (sum(np.array(test_labels_hat) == np.array(test_labels)) + 0.0) / len(test_labels)
    return accuracy

def main():
    ionosphere = 0.91453
    irrelevant = 0.645
    mfeat_fourier = 0.746627
    spambase = 0.915906

    print "Decision Trees (J48) Accuracy Results by Weka"
    print "ionosphere = ", ionosphere
    print "irrelevant = ", irrelevant
    print "mfeat_fourier = ", mfeat_fourier
    print "spambase = ", spambase

    ionosphere_train = read_dataset("ionosphere_train.arff")
    irrelevant_train = read_dataset("irrelevant_train.arff")
    mfeat_fourier_train = read_dataset("mfeat-fourier_train.arff")
    spambase_train = read_dataset("spambase_train.arff")

    ionosphere_test = read_dataset("ionosphere_test.arff")
    irrelevant_test = read_dataset("irrelevant_test.arff")
    mfeat_fourier_test = read_dataset("mfeat-fourier_test.arff")
    spambase_test = read_dataset("spambase_test.arff")

    print "Evaluating kNN with respect to k"

    accuracy_list = []
    for k in xrange(1, 26):
        accuracy = kNN(ionosphere_test, ionosphere_train, k)
        accuracy_list.append(accuracy)
    print "Accuracy of ionosphere:", accuracy_list

    accuracy_list = []
    for k in xrange(1, 26):
        accuracy = kNN(irrelevant_test, irrelevant_train, k)
        accuracy_list.append(accuracy)
    print "Accuracy of irrelevant:", accuracy_list

    accuracy_list = []
    for k in xrange(1, 26):
        accuracy = kNN(mfeat_fourier_test, mfeat_fourier_train, k)
        accuracy_list.append(accuracy)
    print "Accuracy of mfeat-fourier:", accuracy_list

    accuracy_list = []
    for k in xrange(1, 26):
        accuracy = kNN(spambase_test, spambase_train, k)
        accuracy_list.append(accuracy)
    print "Accuracy of spambase:", accuracy_list


    print "Feature Selection for kNN"
    accuracy_list = []
    n_vec = np.arange(1, len(ionosphere_train[0]) - 1)
    for n in n_vec:
        accuracy = choose_feature(ionosphere_train, ionosphere_test, n)
        accuracy_list.append(accuracy)
    print "Accuracy of ionosphere:", accuracy_list

    accuracy_list = []
    n_vec = np.arange(1, len(irrelevant_train[0]) - 1)
    for n in n_vec:
        accuracy = choose_feature(irrelevant_train, irrelevant_test, n)
        accuracy_list.append(accuracy)
    print "Accuracy of irrelevant:", accuracy_list

    accuracy_list = []
    n_vec = np.arange(1, len(ionosphere_train[0]) - 1)
    for n in n_vec:
        accuracy = choose_feature(mfeat_fourier_train, mfeat_fourier_test, n)
        accuracy_list.append(accuracy)
    print "Accuracy of mfeat-fourier:", accuracy_list

    accuracy_list = []
    n_vec = np.arange(1, len(ionosphere_train[0]) - 1)
    for n in n_vec:
        accuracy = choose_feature(spambase_train, spambase_test, n)
        accuracy_list.append(accuracy)
    print "Accuracy of spambase:", accuracy_list

if __name__ == "__main__":
    main()