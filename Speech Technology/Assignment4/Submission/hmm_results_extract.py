import numpy as np

files = ['alphas_2.txt', 'alphas_3.txt', 'alphas_4.txt', 'alphas_5.txt', 'alphas_z.txt']

all_alphas = []

for i, file in enumerate(files):
    alphas = []
    with open (file, 'r') as f:
        while True:
            line = f.readline()
            if len(line.split())>0:
                if(line.split()[0] == 'alpha'):
                    alphas.append(float(line.split()[4]))
            if not line:
                break
    all_alphas.append(alphas)

all_alphas = np.array(all_alphas)

index_to_digit_dict = {0:'2', 1:'3', 2: '4', 3: '5', 4: 'z'}

highest_likelihood_indices = np.argmax(all_alphas, axis = 0)

predictions = [index_to_digit_dict[ind] for ind in highest_likelihood_indices]

correct_labels = 12*['2'] + 12*['3'] + 12*['4'] + 12*['5'] + 12*['z']

accuracy = sum([pred == correct_lbl for pred, correct_lbl in zip(predictions, correct_labels)])/len(predictions)*100

print("Predicted digit labels:", predictions)
print("True labels:", correct_labels)
print("Accuracy =", accuracy, "%")