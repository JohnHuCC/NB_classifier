from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi
import pandas as pd
import numpy as np

# Load a CSV file


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        next(file)
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            # Convert string column to float
            for i in range(0, len(row)):
                row[i] = float(row[i])
            # print(row)
            dataset.append(row)
    return dataset

# Split a dataset into k folds


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation s plit


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Split the dataset by class values, returns a dictionary


def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

# use
# Calculate the mean of a list of numbers


def mean(numbers):
    return sum(numbers)/float(len(numbers))

# use
# Calculate the standard deviation of a list of numbers


def stdev(numbers):
    EPS = 1e-5
    avg = mean(numbers)
    variance = (sum([(x-avg)**2 for x in numbers]) + EPS) / \
        (float(len(numbers)-1) + EPS)
    return sqrt(variance)

# use
# Calculate the mean, stdev and count for each column in a dataset


def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column))
                 for column in zip(*dataset)]
    del(summaries[-1])
    return summaries

# use
# Split dataset by class then calculate statistics for each row


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

# Calculate the Gaussian probability distribution function for x


def calculate_probability(x, mean, stdev):
    #EPS = 1e-5
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    if((1 / (sqrt(2 * pi) * stdev)) * exponent == 0):
        prob = (1 + 1 / (sqrt(2 * pi) * stdev)+2)
    else:
        prob = (1 / (sqrt(2 * pi) * stdev))
    return prob

# Calculate the probabilities of predicting each class for a given row


def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / \
            float(total_rows)
        print(len(class_summaries))
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
# 			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

# Predict the class for a given row


def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

# Naive Bayes Algorithm


def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return(predictions)


# Test Naive Bayes on Iris Dataset
# xlsfile_path = './Project_NB_Tr.xlsx'
csvfile_path = '/Users/johnnyhu/Desktop/Dataset/AI_project3_train01.csv'
csvfile_path_ts = '/Users/johnnyhu/Desktop/Dataset/AI_project3_test01.csv'
# csvfile_path2 = '/Users/johnnyhu/Desktop/Dataset/Project_NB_Tr.csv'
# df_val = pd.read_excel(xlsfile_path)
# df_val = df_val.fillna(-999)
# X = df_val.drop(columns=['No', 'Target'])
# X['Laterality'] = X['Laterality'].replace('L', 0)
# X['Laterality'] = X['Laterality'].replace('R', 1)
# y = df_val['Target']
# X['Target'] = y
# X.to_csv(csvfile_path,header=False, index=False)

seed(1)
dataset = load_csv(csvfile_path)
dataset_ts = load_csv(csvfile_path_ts)
summarize = summarize_by_class(dataset)
for row in dataset_ts:
    output = calculate_class_probabilities(summarize, row)
    print(output)
# n_folds = 10
# scores = evaluate_algorithm(dataset, naive_bayes, n_folds)

print(naive_bayes(dataset, dataset_ts))
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
