import os
from collections import Counter

def check_labels(label_dir):
    class_counts = Counter()
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
    return class_counts

train_labels = check_labels("DataSets\\Fisheye8K_all_including_train&test\\train\\labels")
test_labels = check_labels("DataSets\\Fisheye8K_all_including_train&test\\test\\labels")
print("Train label distribution:", dict(train_labels))
print("Test label distribution:", dict(test_labels))
