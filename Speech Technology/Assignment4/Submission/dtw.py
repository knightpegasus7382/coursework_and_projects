import glob
import numpy as np

def dtwarr(a, b):
    # dtw of 2d arrays

    n, m = len(a), len(b)
    dtw_matrix = np.matrix(np.ones((n + 1, m + 1)) * np.inf)
    dtw_matrix[0, 0] = 0

    initial_cost_matrix = np.zeros((n, m))

    for i in range(1, n+1):
        for j in range(1, m+1):
            # Calculate the cost between any two vectors as the square of the Euclidean distance between them
            initial_cost_matrix[i-1, j-1] = np.sum((a[i-1]-b[j-1])**2)
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = initial_cost_matrix[i-1, j-1] + last_min

    return dtw_matrix[-1, -1]

def result(score_list, class_list, actual_class, q=5):
    # gets the result based on top20 similar samples
    # also gives ratio of samples which were of actual class of example and ratio of samples for top class
    sorted_res20 = np.argsort(np.array(score_list))[:q]
    res_dict = {}
    for si in sorted_res20:
        if class_list[si] in res_dict.keys():
            res_dict[class_list[si]] += 1
        else:
            res_dict[class_list[si]] = 1
    max_class = max(res_dict, key=lambda p: res_dict[p])
    accuracy = res_dict[actual_class] / q
    prec = res_dict[max_class] / q
    return max_class, accuracy, prec


# folders of train data

train_list = [r"features/train/2/*.mfcc",
              r"features/train/3/*.mfcc",
              r"features/train/4/*.mfcc",
              r"features/train/5/*.mfcc",
              r"features/train/z/*.mfcc"]
# folders of test data

test_list = [r"features/test/2/*.mfcc",
             r"features/test/3/*.mfcc",
             r"features/test/4/*.mfcc",
             r"features/test/5/*.mfcc",
             r"features/test/z/*.mfcc"]

# extracting files
x_train = []
y_train = []
x_test = []
y_test = []
test_file_list = {}
for i in range(len(train_list)):
    file_list_train = glob.glob(train_list[i])
    for file in file_list_train:
        x_train.append(np.loadtxt(file, skiprows=1))
        y_train.append(i)
    file_list_test = glob.glob(test_list[i])
    for file in file_list_test:
        x_test.append(np.loadtxt(file, skiprows=1))
        y_test.append(i)
        test_file_list[len(y_test) - 1] = file

# running dtw for the train data
mistakes = 0
for i in range(len(x_test)):
    mini = dtwarr(x_test[i], x_train[0])
    outc = y_train[0]
    res_list = []
    out_list = []
    for j in range(len(x_train)):
        resu = dtwarr(x_test[i], x_train[j])
        res_list.append(resu)
        out_list.append(y_train[j])
        if resu < mini:
            mini = resu
            outc = y_train[j]
    res20, acc, prec = result(res_list, out_list, y_test[i], q=20)
    if res20 != y_test[i]:
        mistakes += 1
        print('Actual:', y_test[i])
        print('Predicted 20:', res20)
        print('Ratio for actual result:', acc)
        print('Ratio for the given result:', prec)

print('Accuracy is ', (1 - (mistakes / len(x_test))) * 100, "%")
