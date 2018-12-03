import copy
import math
import numpy as np

def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1


def data_in_x_y_format (filename, no_of_columns):
  no_of_rows = file_len (filename)
  data = np.zeros ((no_of_rows, no_of_columns))

  row_no = 0
  with open(filename) as f:
    for line in f:
      words = line.split()
      start = 1
      for word in words:
        if  start == 1:
          start = 0
          data[row_no][0] = int (word)
        else:
          parts = word.split(':')
          column = int (parts[0])
          value = float (parts[1])
          data[row_no][column] = value
      row_no += 1

  X = data
  Y = list(range (1, no_of_columns+1))
  return X, Y

def compute_likelihood_per_feature (X, Y, feature_index):
  unique_lables = np.unique (X[:, 0])
  unique_values = np.unique (X[:, feature_index])
  per_value_dict = {}
  per_value_per_label_dict = {}

  print (unique_values)
  for value in unique_values:
    for label in unique_lables:
      per_value_per_label_dict[label] = 0 

    temp_dict = copy.deepcopy (per_value_per_label_dict)
    per_value_dict[value] = temp_dict

#  print (per_value_dict)
#  for value in unique_values:
#    print (value)
#    print (per_value_dict[value])

def compute_likelihood_of_features (X, Y):
  no_of_features = len (Y)
  for i in range (0, no_of_features):
    compute_likelihood_per_feature (X, Y, i)

def get_prior_label_prob (X, Y):
  unique_lables = np.unique (X[:, 0])
  no_of_rows = X.shape[0]

  label_count = {}
  label_prob = {}

  for label in unique_lables:
    label_count[label] = 0
     
  for i in range (0, no_of_rows):
    label_count[X[i, 0]] = label_count[X[i, 0]] + 1

  for label in unique_lables:
    label_prob[label] = label_count[label]/no_of_rows

  print ("Label Probability")
  print (label_prob)

  return unique_lables, label_prob

def main_func ():
  X, Y = data_in_x_y_format ("tennis.data", 6)
  unique_labels, label_prob_dict = get_prior_label_prob (X, Y)
  compute_likelihood_of_features (X, Y)

main_func ()
