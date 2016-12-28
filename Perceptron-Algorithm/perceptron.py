import numpy as np

I = 50

"""
This function reads arff file
Returns a 2D list: [no. of examples][example]
"""
def read_dataset(file_name):

    file = open( "./dataset/" + file_name,"rb")
    data_found = False
    data = []

    for line in file:
        if data_found and (len(line) > 0):
            temp_line = line.strip().split(",")
            temp_data = [float(i) for i in temp_line]
            data.append(temp_data)
        if line.strip().startswith("@DATA"):
            data_found = True

    return data

# Primal Perceptron with Margin
def Primal_PwM(train_dataset):
    m = len(train_dataset[0])
    A = 0.0
    examples = []
    classes = []

    for i in xrange(len(train_dataset)):
        tempExample = train_dataset[i][0:m - 1]
        tempExample.insert(0, 1)  # add a feature with constant value 1

        tempLabel = train_dataset[i][-1]
        A += np.linalg.norm(tempExample)  # A is the norm of training examples
        """
        Storing examples and classes separately
        """
        examples.append(tempExample)
        classes.append(tempLabel)

    A = A / len(train_dataset)  # A is the average norm of training examples
    T = 0.1 * A
    w = [0.0 for a in xrange(m)]

    for i in xrange(I):
        for j in xrange(len(train_dataset)):
            if classes[j] * np.dot(w, examples[j]) < T:
                for k in xrange(m):
                    w[k] += classes[j] * examples[j][k]
    return w

def test_Primal_PwM(test_dataset, weights):
    test_data_len = len(test_dataset)
    accuracy = 0.0
    m = len(test_dataset[0])

    for i in xrange(test_data_len):
        tempExample = test_dataset[i][0:m - 1]
        tempExample.insert(0, 1)  # add a feature with constant value 1
        tempLabel = test_dataset[i][-1]

        sign = np.sign(np.dot(weights, tempExample))
        if tempLabel == sign:
            accuracy += 1

    return accuracy / test_data_len

def dual_sign(alphas, classes, examples, test_example, d, s):
    temp = []

    for k in xrange(len(examples)):

        if d < 0:
            norm = (np.linalg.norm(np.array(examples[k]) - np.array(test_example)))
            div = - (norm*norm) / (2.0 * (s*s))
            kernel = np.exp(div)
        else:
            kernel = (np.dot(examples[k], test_example) + 1) ** d

        temp.append(kernel)

    temp = np.array(temp)

    multi = np.multiply(np.array(alphas), np.array(classes))
    multi = np.multiply(multi, temp)

    return np.sum(multi)

# Kernel Perceptron with Margin
def Dual_PwM(train_dataset, d, s):
    m = len(train_dataset[0]) - 1
    A = 0.0
    examples = []
    classes = []

    for i in xrange(len(train_dataset)):
        tempExample = train_dataset[i][0:m]
        tempLabel = train_dataset[i][-1]

        if d < 0:
            norm = (np.linalg.norm(np.array(tempExample) - np.array(tempExample)))
            div = - (norm*norm) / (2.0 * (s*s))
            kernel = np.exp(div)
        else:
            kernel = (np.dot(tempExample, tempExample) + 1) ** d

        A += np.sqrt(kernel)

        examples.append(tempExample)
        classes.append(tempLabel)

    A = A / len(train_dataset)
    T = 0.1 * A

    alphas = [0.0 for a in xrange(len(train_dataset))]

    for i in xrange(I):
        for j in xrange(len(train_dataset)):
            if (classes[j] * dual_sign(alphas, classes, examples, examples[j], d, s)) < T:
                alphas[j] += 1
    return alphas

def test_Dual_PwM(examples, classes, test_dataset, alphas, d, s):
    accuracy = 0.0
    m = len(test_dataset[0]) - 1

    for i in xrange(len(test_dataset)):
        tempExample = test_dataset[i][0:m]
        tempLabel = test_dataset[i][-1]

        sign = np.sign(dual_sign(alphas, classes, examples, tempExample, d, s))
        if tempLabel == sign:
            accuracy += 1

    return accuracy / len(test_dataset)

def main():
    train_dataset_file = ["ATrain.arff", "BTrain.arff", "CTrain.arff", "backTrain.arff", "breastTrain.arff", "sonarTrain.arff"]
    test_dataset_file = ["ATest.arff", "BTest.arff", "CTest.arff", "backTest.arff", "breastTest.arff", "sonarTest.arff"]

    d_values = [1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1]
    s_values = [1, 1, 1, 1, 1, 0.1, 0.5, 1, 2, 5, 10]

    for i in xrange(len(train_dataset_file)):
        train_dataset = read_dataset(train_dataset_file[i])
        test_dataset = read_dataset(test_dataset_file[i])

        """
        Separating examples and classes
        """
        trainExamples = []
        trainclasses = []

        for line in xrange(len(train_dataset)):
            trainExamples.append(train_dataset[line][0:(len(train_dataset[line]) - 1)])
            trainclasses.append(train_dataset[line][-1])

        weights = Primal_PwM(train_dataset)
        accuracy = test_Primal_PwM(test_dataset, weights)
        print "File Name:", train_dataset_file[i]
        print "Accuracy of Primal Perceptron with Margin:", accuracy * 100, "%"

        print "Accuracy of Dual Perceptron with Margin:"

        for j in xrange(len(d_values)):
            alphas = Dual_PwM(train_dataset, d_values[j], s_values[j])
            accuracy = test_Dual_PwM(trainExamples, trainclasses, test_dataset, alphas, d_values[j], s_values[j])

            if d_values[j] > 0:
                print "For d: ", d_values[j], ", Accuracy: ", accuracy * 100, "%"
            else:
                print "For s: ", s_values[j], ", Accuracy: ", accuracy * 100, "%"

if __name__ == "__main__":
    main()