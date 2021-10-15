import glob
import numpy as np

# Function to train a K-Means Clustering Algorithm given a set of vectors and number of clusters
# Returns a set of centroids of the clusters

def kMeansTrain(vectors, num_clusters):
    centroids = vectors[np.random.choice(vectors.shape[0], size=num_clusters, replace=False)]
    max_iter = 10000
    iter = 0
    new_centroids = np.zeros((num_clusters, vectors.shape[1]))
    while np.sum((new_centroids - centroids)**2) > 10**(-2):
        if iter > 1:
            centroids = new_centroids
        if iter > max_iter:
            break
        iter += 1
        distances = []
        for i in range(num_clusters):
            difference_square_norm = np.sum((vectors - centroids[i]) ** 2, axis=1)
            distances.append(difference_square_norm)
        distances = np.array(distances)
        cluster_labelling = np.argmin(distances, axis=0)
        for i in range(num_clusters):
            new_centroids[i] = np.mean(vectors[cluster_labelling == i], axis=0)

    return new_centroids

# Function to predict clusters of test vectors using centroids of a K-Means Clustering Algorithm

def kMeansPredict(centroids, test_vectors):
    distances = []
    for i in range(len(centroids)):
        difference_square_norm = np.sum((test_vectors - centroids[i]) ** 2, axis=1)
        distances.append(difference_square_norm)
    distances = np.array(distances)
    cluster_labelling = np.argmin(distances, axis=0)

    return cluster_labelling



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

# reading the data
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


#combine all the training data
cluster_train = np.concatenate(x_train)

# k-means clustering

clusters_num = int(len(cluster_train)/100)
centroid = kMeansTrain(cluster_train, clusters_num)

train_2 = []
train_3 = []
train_4 = []
train_5 = []
train_z = []
test_set = []


# get sequences for data
for i in range(len(x_train)):
    if y_train[i] == 0:
        train_2.append(kMeansPredict(centroid, x_train[i]))
    elif y_train[i] == 1:
        train_3.append(kMeansPredict(centroid, x_train[i]))
    elif y_train[i] == 2:
        train_4.append(kMeansPredict(centroid, x_train[i]))
    elif y_train[i] == 3:
        train_5.append(kMeansPredict(centroid, x_train[i]))
    else:
        train_z.append(kMeansPredict(centroid, x_train[i]))

for i in range(len(x_test)):
    test_set.append(kMeansPredict(centroid, x_test[i]))

# save the sequences to files for hmm
f = open('train_2.hmm.seq', 'w')
for ele in train_2:
    f.write(str(list(ele))[1:-1].replace(',', '') + '\n')

f.close()

f = open('train_3.hmm.seq', 'w')
for ele in train_3:
    f.write(str(list(ele))[1:-1].replace(',', '') + '\n')

f.close()

f = open('train_4.hmm.seq', 'w')
for ele in train_4:
    f.write(str(list(ele))[1:-1].replace(',', '') + '\n')

f.close()

f = open('train_5.hmm.seq', 'w')
for ele in train_5:
    f.write(str(list(ele))[1:-1].replace(',', '') + '\n')

f.close()

f = open('train_z.hmm.seq', 'w')
for ele in train_z:
    f.write(str(list(ele))[1:-1].replace(',', '') + '\n')

f.close()

f = open('test_all.hmm.seq', 'w')
for ele in test_set:
    f.write(str(list(ele))[1:-1].replace(',', '') + '\n')

f.close()