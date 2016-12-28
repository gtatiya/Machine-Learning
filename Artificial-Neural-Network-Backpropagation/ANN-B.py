import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

INPUT_SIZE = 8 ** 2
OUT_SIZE = 10
LEARNING_FACTOR = .1
TRAIN_SIZE = 3823
TEST_SIZE = 1797

def read_dataset(file_name, testing_set=False):
    with open(file_name, "r") as dataFl:  # Opens the ARFF file
        lines = [line for line in dataFl]  # Stores all the lines in a list
    attribute = []
    l = []  # for data in 2D list
    for line in lines:
        if line.startswith("%") or 'data' in line or line == '\n':  # this is a comment or the data line
            pass
        elif line.startswith("@"):
            if "relation" in line:
                relationName = line.split(" ")[1]
            elif "attribute" in line:
                attribute.append(line.split(" ")[1])
                if 'class' in line:
                    line = line.split(" ")
                    b = "\n{}"  # Remove this characters
                    for char in b:
                        line[2] = line[2].replace(char, "")
                    classes = line[2].split(',')
        else:
            line = line.replace('\n', ' ')
            line = line.replace(',', ' ')
            line_split = line.split(' ')
            l.extend(line_split)

    for k in range(l.count('')):
        l.remove('')
    l = [int(i) for i in l]

    if testing_set:
        iters = 1797
    else:
        iters = 3823

    matrices = []
    num_list = []
    for i in range(iters):
        mini_list = l[i * 65:(i + 1) * 65]
        num_list.append(mini_list.pop())
        mat = []
        for ind in range(1, 9):
            mat.append(mini_list[8 * (ind - 1):8 * ind])
        matrices.append(mat)

    return relationName, attribute, classes, matrices, num_list

def sig(x):
    res = 1.0 / (1 + np.e ** (-x))
    return res

class ANN:
    def __init__(self, input_size, depth_size, width_size, output_size, input_matrix, input_values, test_matrix, test_values):

        self.input_size = input_size
        self.depth_size = depth_size
        self.width_size = width_size
        self.output_size = output_size

        self.input_matrix = input_matrix
        self.input_vals = input_values
        self.test_matrix = test_matrix
        self.test_vals = test_values
        self.build()

    # This function generate weights with independent random numbers uniformly sampled in the range [-0.1, 0.1]
    def build(self):
        self.hidden_vect = np.zeros(self.width_size)
        self.out_vect = np.zeros(self.output_size)
        self.in_final_vect = np.zeros(self.width_size)

        if self.depth_size == 0:
            self.out_weights = np.random.uniform(-0.1, 0.1, size=(self.input_size, self.output_size))
        elif self.depth_size == 1:
            self.in_weights = np.random.uniform(-0.1, 0.1, size=(self.input_size, self.width_size))
            self.out_weights = np.random.uniform(-0.1, 0.1, size=(self.width_size, self.output_size))
        elif self.depth_size >= 2:
            self.in_weights = np.random.uniform(-0.1, 0.1, size=(self.input_size, self.width_size))

            self.hidden_weights = defaultdict(list)
            for i in xrange(1, self.depth_size):
                self.hidden_weights[i] = np.random.uniform(-0.1, 0.1, size=(self.width_size, self.width_size))

            self.out_weights = np.random.uniform(-0.1, 0.1, size=(self.width_size, self.output_size))

    def train(self, iters):
        for iter in xrange(iters):
            for mat_index in range(len(self.input_matrix)):
                self.feed(mat_index)

                expected_vect = np.zeros(self.output_size)
                expected_vect[self.input_vals[mat_index]] = 1.0

                # Backpropogate
                # Get output error
                out_err_vect = (-1) * (expected_vect - self.out_vect) * self.out_vect * (1.0 - self.out_vect)

                # Get hidden node error and update weight
                if self.depth_size == 0:
                    out_weights_delta = (self.in_vect).transpose().dot(out_err_vect)

                    self.out_weights = self.out_weights - (out_weights_delta * LEARNING_FACTOR)

                elif self.depth_size == 1:
                    sm = self.out_weights.dot(out_err_vect.transpose())
                    hidden_err = sm * (self.in_final_vect.transpose()) * (1.0 - self.in_final_vect.transpose())

                    in_weights_delta = (hidden_err.dot(self.in_vect)).transpose()
                    out_weights_delta = self.in_final_vect.transpose().dot(out_err_vect)

                    self.in_weights = self.in_weights - (in_weights_delta * LEARNING_FACTOR)
                    self.out_weights = self.out_weights - (out_weights_delta * LEARNING_FACTOR)

                elif self.depth_size >= 2:
                    sm = self.out_weights.dot(out_err_vect.transpose())
                    hidden_err = defaultdict(list)
                    hidden_err[self.depth_size-1] = sm * (self.hidden_vect[len(self.hidden_vect)].transpose()) * (1.0 - self.hidden_vect[len(self.hidden_vect)].transpose())

                    for i in xrange(self.depth_size-2, 0, -1):
                        sm = self.hidden_weights[i].dot(hidden_err[i+1])
                        hidden_err[i] = sm * (self.hidden_vect[i].transpose()) * (1.0 - self.hidden_vect[i].transpose())

                    in_weights_delta = (hidden_err[1].dot(self.in_vect)).transpose()

                    hidden_weights_delta = defaultdict(list)

                    for i in xrange(1, self.depth_size):  # starts from "2"
                        hidden_weights_delta[i] = self.hidden_vect[i].transpose().dot(np.array(hidden_err[i]).transpose())

                    out_weights_delta = self.hidden_vect[len(self.hidden_vect)].transpose().dot(out_err_vect)

                    self.in_weights = self.in_weights - (in_weights_delta * LEARNING_FACTOR)

                    for i in xrange(2, self.depth_size):
                        self.hidden_weights[i] = self.hidden_weights[i] - (hidden_weights_delta[i] * LEARNING_FACTOR)

                    self.out_weights = self.out_weights - (out_weights_delta * LEARNING_FACTOR)

    def test(self):
        num_right = 0
        for i in range(len(self.test_matrix)):
            self.feed(i, False)
            out_list = self.out_vect.tolist()[0]

            maxval = max(out_list)

            maxindex = out_list.index(maxval)

            if maxindex == self.test_vals[i]:
                num_right += 1

        accuracy = (num_right * 100.0 / (len(self.test_matrix)))
        return accuracy

    def feed(self, index, train=True):
        # index is index of input_matrix to choose for inputs
        if train:
            in_mat = np.reshape(self.input_matrix[index], (1, INPUT_SIZE))
        else:
            in_mat = np.reshape(self.test_matrix[index], (1, INPUT_SIZE))

        if self.depth_size == 0:
            out_final = in_mat.dot(self.out_weights)
            out_final = sig(out_final)
        elif self.depth_size == 1:
            in_final = in_mat.dot(self.in_weights)
            in_final = sig(in_final)
            out_final = in_final.dot(self.out_weights)
            out_final = sig(out_final)

            self.in_final_vect = in_final
        elif self.depth_size >= 2:
            in_final = in_mat.dot(self.in_weights)
            in_final = sig(in_final)

            hidden_final = defaultdict(list)
            hidden_final[1] = in_final.dot(self.hidden_weights[1])
            hidden_final[1] = sig(hidden_final[1])

            for i in xrange(2, self.depth_size):
                hidden_final[i] = np.array(hidden_final[i-1]).dot(self.hidden_weights[i])
                hidden_final[i] = sig(hidden_final[i])

            out_final = hidden_final[len(hidden_final)].dot(self.out_weights)
            out_final = sig(out_final)

            self.hidden_vect = hidden_final  # self.hidden_vect is a dictionary
            self.in_final_vect = in_final

        self.in_vect = in_mat
        self.out_vect = out_final

def main():

    relationName, attribute, classes, matrices, num_list = read_dataset("optdigits_train.arff")

    print("Relation Name is : %s" % relationName)
    print("Attributes are " + ', '.join(attribute))
    print "classes are:", classes
    print "Number of classes:", len(classes)

    relationName, attribute, classes, test_mat, test_nums = read_dataset("optdigits_test.arff", True)

    depth = [0, 1, 2, 3, 4]
    res_list = []
    for i in depth:
        if i == 0:
            width = [0]
        else:
            width = [1, 2, 5, 10]
        for j in width:
            n = ANN(INPUT_SIZE, i, j, OUT_SIZE, matrices, num_list, test_mat, test_nums)
            n.train(200)
            res = n.test()
            if i > 0:
                res_list.append(res)
                print "For depth", i, ", width", j, ", accuracy is", res, "%"
            else:
                print "For depth", i, ", no width, accuracy is", res, "%"
        if width == [1, 2, 5, 10]:
            width = (width)
            res_list = (res_list)
            plt.ylabel("Accuracy", fontsize=16)
            plt.xlabel("Width", fontsize=16)
            plt.title("Accuracy curve for depth:" + str(i), fontsize=16)
            plt.ylim(0, 100)
            plt.plot(width, res_list)
            plt.show()
            res_list = []

if __name__ == "__main__":
    main()