import os

directory = r"./features/connected/dev"

truth_labels = []

for filename in os.listdir(directory):
    if filename[-4:] == "mfcc":
        truth_labels.append(list(filename[:-6]))

print(truth_labels)

with open (directory+r"/truthlabels.txt", 'w') as f:
    for label in truth_labels:
        for digit in label:
            f.write(digit+" ") 
        f.write('\n')