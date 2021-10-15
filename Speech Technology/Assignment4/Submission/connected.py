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
            # Calculate the cost between any two vectors as the square of Euclidean distance between them
            initial_cost_matrix[i-1, j-1] = np.sum((a[i-1]-b[j-1])**2)
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = initial_cost_matrix[i-1, j-1] + last_min

    return np.array(dtw_matrix[1:, 1:])

# Folders of train data

train_list = [r"features/train/2/*.mfcc",
              r"features/train/3/*.mfcc",
              r"features/train/4/*.mfcc",
              r"features/train/5/*.mfcc",
              r"features/train/z/*.mfcc"]

# Folders of blind-test data

blind_test_list = [r"features/connected/blind_test/*.mfcc"]

# Extracting train data
x_train = []
y_train = []

for i in range(len(train_list)):
    file_list_train = glob.glob(train_list[i])
    for file in file_list_train:
        x_train.append(np.loadtxt(file, skiprows=1))
        y_train.append(i)

# Separating the train data into the train data for '2', train data for '3', etc. for 2LDP
x_train_2 = [x for x,y in zip(x_train, y_train) if y == 0]
x_train_3 = [x for x,y in zip(x_train, y_train) if y == 1]
x_train_4 = [x for x,y in zip(x_train, y_train) if y == 2]
x_train_5 = [x for x,y in zip(x_train, y_train) if y == 3]
x_train_z = [x for x,y in zip(x_train, y_train) if y == 4]

x_train_lists = [x_train_2, x_train_3, x_train_4, x_train_5, x_train_z]

# Creating a mapping from indices of loops to the digits
digits_dict = {0:'2', 1:'3', 2:'4', 3:'5', 4:'z'}

# Extracting blind-test data
blind_x_tests = []
file_list_blind_test = glob.glob(blind_test_list[0])
for file in file_list_blind_test:
    blind_x_tests.append(np.loadtxt(file, skiprows=1))

# SCROLL PAST THIS COMMENTED SECTION TO GO TO THE CODE FOR PREDICTIONS ON THE BLIND-TESTS T1, T2, T3, T4, T5.
# UNCOMMENT THE BELOW SECTION OF THE CODE IN ORDER TO TEST ANY PARTICULAR FILES IN THE dev DIRECTORY.
# SELECT THE REQUIRED FILES IN THE dev DIRECTORY BY TWEAKING THE 'test_start_index' AND 'num_test_samples' PARAMETERS.

# The comments explaining the below section of code will be similar to those explaining the similar code for the blind-tests T1, T2, T3, T4, T5.
# Please refer to that section for commented code.

test_list = [r"features/connected/dev/*.mfcc"]

test_labels_file = r"features/connected/dev/truthlabels.txt"

test_labels = []
with open (test_labels_file, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        test_labels.append(line.split())

# Extracting dev test data

x_test = []
y_test = []

file_list_test = glob.glob(test_list[0])
for ind, file in enumerate(file_list_test):
    x_test.append(np.loadtxt(file, skiprows=1))
    y_test.append(test_labels[ind])

test_start_index = 4
num_test_samples = 1
test_end_index = test_start_index + num_test_samples

for ind, test in enumerate(x_test[test_start_index: test_end_index]):
    #print("Length of the current sequence =", len(test))

    starts_dict = {'dum':0}
    path_min_distances = {'dum':0}

    final_paths = set()

    while True:
        starting_points = starts_dict.values()
        starting_points = np.array(list(starting_points))

        starts_dict_copy = starts_dict.copy()
        for path_key, start_point in starts_dict.items():
            start_point = int(start_point)

            del starts_dict_copy[path_key]

            if start_point < len(test):
                prev_distance = path_min_distances[path_key]

                for key, digit in digits_dict.items():
                    min_distances_for_digit = []
                    
                    first_end_points = []

                    for example in x_train_lists[key]:
                        dtw_dists_for_example = []
                        first_end_point = int(start_point+len(example)/2)
                        last_end_point = int(start_point+2*len(example))
                        mid_point = int((first_end_point+last_end_point)/2)
                        first_end_points.append(first_end_point)

                    if np.mean(first_end_points) >= len(test):
                        if path_key != 'dum':
                            final_paths.add(path_key)
                    else:
                        for example in x_train_lists[key]:

                            dtw_dists_for_example = []
                            first_end_point = int(len(example)/2)
                            last_end_point = int(2*len(example))
                            end_points = np.arange(first_end_point, min(len(test)-start_point, last_end_point + 1))

                            if len(end_points) > 0:

                                largest_piece = test[start_point : start_point+end_points[-1]+1]
                                largest_dtw_matrix = dtwarr(example, largest_piece)
                                dtw_dists_for_example.append(largest_dtw_matrix[-1, end_points[0]:])
                                dtw_dists_for_example.append(start_point+end_points)
                                dtw_dists_for_example = np.array(dtw_dists_for_example)

                                min_distances_for_digit.append([np.min(dtw_dists_for_example[0,:]), dtw_dists_for_example[1, np.argmin(dtw_dists_for_example[0,:])]])

                        if len(min_distances_for_digit) > 0:
                            min_distances_for_digit = np.array(min_distances_for_digit)
                            final_min_distance_for_digit = [np.min(min_distances_for_digit[:,0]), min_distances_for_digit[np.argmin(min_distances_for_digit[:,0]), 1]]
                            
                            if type(path_key)==tuple:
                                new_key = path_key+(digit,)
                            else:
                                new_key = (path_key, digit)
                            
                            starts_dict_copy[new_key] = final_min_distance_for_digit.pop()
                            path_min_distances[new_key] = prev_distance + final_min_distance_for_digit.pop()

            if 'dum' in starts_dict_copy:
                del starts_dict_copy['dum']
            if 'dum' in path_min_distances:
                del path_min_distances['dum']
        #print(starts_dict_copy)

        if not bool(starts_dict_copy):
            break

        starts_dict = starts_dict_copy.copy()
    
    selected_paths = {}
    for path in final_paths:
        selected_paths[path] = path_min_distances[path]
    #print("FINAL PATH DISTANCES:", selected_paths)

    first_pred_key = min(selected_paths, key=selected_paths.get)
    del selected_paths[first_pred_key]
    second_pred_key = min(selected_paths, key=selected_paths.get)
    del selected_paths[second_pred_key]
    third_pred_key = min(selected_paths, key=selected_paths.get)

    first_pred = list(first_pred_key[1:])
    second_pred = list(second_pred_key[1:])
    third_pred = list(third_pred_key[1:])

    print("True Labels = ", y_test[test_start_index+ind])
    print("1st Prediction = ", first_pred)
    print("2nd Prediction = ", second_pred)
    print("3rd Prediction = ", third_pred)

    if y_test[test_start_index+ind] in [first_pred, second_pred, third_pred] :
        print("Correct prediction has been made in top 3!")
    else:
        print("All predictions are WRONG")


# SECTION OF THE CODE THAT PREDICTS FOR THE BLIND-TESTS
# CAN BE COMMENTED OUT IF WE WISH TO RUN THE ABOVE SECTION OF TESTING SPECIFIC EXAMPLES IN dev DIRECTORY
"""
# Iterating through the blind-tests
for ind, blind_test in enumerate(blind_x_tests):
    #print("Length of the current sequence =", len(blind_test))


    # Through iterations, starts_dict stores the path of digits as the key, and the corresponding endpoints (next start points for further DTWs) as the value
    # Initialised with a dummy digit as the key, and 0 as starting point
    starts_dict = {'dum':0}

    # Similarly for path_min_distances
    path_min_distances = {'dum':0}

    # A set to store the final candidate paths that could be the answer
    final_paths = set()

    while True:
        starting_points = starts_dict.values()
        starting_points = np.array(list(starting_points))

        # Creating a copy of starts_dict to dynamically remove a path whenever that path can be extended to one more digit reasonably certainly
        starts_dict_copy = starts_dict.copy()

        # For each path that already exists in our stored paths, and the corresponding endpoint
        for path_key, start_point in starts_dict.items():
            start_point = int(start_point)

            # The currently selected path will be evaluated for all possible extensions by one future digit, and the best DTW distances will be stored.
            # The path will even be stored as a final candidate path if it can be one. Therefore, all analysis of this path will be done here.
            # Therefore, there is no need to evaluate the path in future iterations and it can be removed.
            del starts_dict_copy[path_key]

            # Start point must not be beyond the length of the entire blind-test sequence
            if start_point < len(blind_test):
                # Storing previous distance of the path so far, to accumulate path distance for any digit added
                prev_distance = path_min_distances[path_key]

                # Iterating through all the digits in the vocabulary
                for key, digit in digits_dict.items():
                    min_distances_for_digit = []
                    
                    first_end_points = []

                    # Running through all train examples of the digit once, to figure out if 
                    # the next endpoints will reach the end of the test sequence on average
                    for example in x_train_lists[key]:
                        dtw_dists_for_example = []
                        first_end_point = int(start_point+len(example)/2)
                        last_end_point = int(start_point+2*len(example))
                        first_end_points.append(first_end_point)

                    # If the average across training samples, of the FIRST end point in the b+l/2 to b+2l range is already beyond the sequence
                    # add the path to the set of candidate final paths to evaluate, because it is possible that this path is already the right one
                    if np.mean(first_end_points) >= len(blind_test):
                        if path_key != 'dum':
                            final_paths.add(path_key)
                    
                    # If the average of the FIRST end point is still inside the sequence, there is most likely still space inside the sequence
                    # to probe the path further
                    else:
                        # Once again iterate through all training examples of the digit
                        for example in x_train_lists[key]:
                            

                            # Initialise a list to store the DTW distances for this training example
                            dtw_dists_for_example = []

                            # Generate the range of endpoints to check the best DTW distance over
                            first_end_point = int(len(example)/2)
                            last_end_point = int(2*len(example))
                            end_points = np.arange(first_end_point, min(len(blind_test)-start_point, last_end_point + 1))

                            # If the range of endpoints is non-zero
                            if len(end_points) > 0:
                                
                                # Obtain DTW for the largest piece of the test sequence [b:b+2l] directly, since the DTW distances for n smaller subsequences
                                # can be obtained by taking the bottom-right n elements in the last row
                                largest_piece = blind_test[start_point : start_point+end_points[-1]+1]
                                largest_dtw_matrix = dtwarr(example, largest_piece)
                                # Append the list, of DTW distances over the range of endpoints, and the corresponding endpoints
                                dtw_dists_for_example.append(largest_dtw_matrix[-1, end_points[0]:])
                                dtw_dists_for_example.append(start_point+end_points)
                                dtw_dists_for_example = np.array(dtw_dists_for_example)

                                # Store the least DTW distance for the training example, and the corresponding endpoint
                                min_distances_for_digit.append([np.min(dtw_dists_for_example[0,:]), dtw_dists_for_example[1, np.argmin(dtw_dists_for_example[0,:])]])

                        if len(min_distances_for_digit) > 0:
                            min_distances_for_digit = np.array(min_distances_for_digit)
                            # Select the training example with the least DTW distance, as the best match from this particular digit of the vocabulary
                            # Store the corresponding least DTW distance, and the corresponding endpoint of this training example
                            final_min_distance_for_digit = [np.min(min_distances_for_digit[:,0]), min_distances_for_digit[np.argmin(min_distances_for_digit[:,0]), 1]]
                            
                            # Generate a new path key by appending current digit to previous path key
                            if type(path_key)==tuple:
                                new_key = path_key+(digit,)
                            else:
                                new_key = (path_key, digit)
                            
                            # Map the new key (tuple of new path of digits) to the corresponding best endpoint and store
                            starts_dict_copy[new_key] = final_min_distance_for_digit.pop()
                            # Map the new key (tuple of new path of digits) to the corresponding best cumulative DTW distance and store
                            path_min_distances[new_key] = prev_distance + final_min_distance_for_digit.pop()

            # Removal of the dummy digit
            if 'dum' in starts_dict_copy:
                del starts_dict_copy['dum']
            if 'dum' in path_min_distances:
                del path_min_distances['dum']

        #print(starts_dict_copy)

        # Break the loop when all the existing paths in the starts_dict have been evaluated and removed, but no further extensions of these paths have been added.
        # Therefore, the trees of these paths have stopped branching out.
        if not bool(starts_dict_copy):
            break

        # Else set starts_dict to the new modified starts_dict_copy and continue
        starts_dict = starts_dict_copy.copy()
    
    # From the path_min_distances dict of all possible paths, select only the distances corresponding to the candidate final paths
    selected_paths = {}
    for path in final_paths:
        selected_paths[path] = path_min_distances[path]
    #print("FINAL PATH DISTANCES:", selected_paths)

    # Select 1st, 2nd, and 3rd predictions from the candidate paths: take the prediction with least cumulative DTW distance, remove it, and repeat
    first_pred_key = min(selected_paths, key=selected_paths.get)
    del selected_paths[first_pred_key]
    second_pred_key = min(selected_paths, key=selected_paths.get)
    del selected_paths[second_pred_key]
    third_pred_key = min(selected_paths, key=selected_paths.get)

    # Obtain the predictions by taking the key and removing the first element (dummy digit), and converting to list
    first_pred = list(first_pred_key[1:])
    second_pred = list(second_pred_key[1:])
    third_pred = list(third_pred_key[1:])

    # Printing the top 3 predictions
    # Since the .mfcc files are read in alphabetical order, we can directly assume that the first blind_test is T1.mfcc, second is T2.mfcc, etc.
    print("PREDICTIONS FOR T"+str(ind+1)+".mfcc")
    print("1st Prediction = ", first_pred)
    print("2nd Prediction = ", second_pred)
    print("3rd Prediction = ", third_pred)
    """