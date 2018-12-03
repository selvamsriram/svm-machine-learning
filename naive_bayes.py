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
  no_of_rows = X.shape[0]
  unique_lables = np.unique (X[:, 0])
  unique_values = np.unique (X[:, feature_index])
  per_value_dict = {}
  per_value_per_label_dict = {}

  #print ("Feature index : ", feature_index, "Unique Values : ", unique_values)
  for value in unique_values:
    for label in unique_lables:
      per_value_per_label_dict[label] = 0 

    temp_dict = copy.deepcopy (per_value_per_label_dict)
    per_value_dict[value] = temp_dict

  for i in range (0, no_of_rows):
    label_dict = per_value_dict[X[i][feature_index]]
    label_dict[X[i][0]] += 1

  return per_value_dict

def compute_likelihood_of_features (X, Y):
  no_of_features = len (Y)
  feature_data_list = [None, None]
  for i in range (2, no_of_features):
    per_feature_dict = compute_likelihood_per_feature (X, Y, i)
    feature_data_list.append (per_feature_dict)

  return feature_data_list

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

  #print ("Label Probability")
  #print (label_prob)

  return unique_lables, label_prob, label_count

def predict_per_row (X, Y, lamda, unique_lables, label_prob_dict, label_count, feature_data_list):
  no_of_features = len (Y)
  perdicted_label_prob_dict = {}

  for label in unique_lables:
    perdicted_label_prob_dict[label] = label_prob_dict[label]

  for label in unique_lables:
    for i in range (2, no_of_features):
      #Compute P(Xi | Y = label)
      fdata = feature_data_list [i]
      if X[i] in fdata: 
        f_value_data = fdata [X[i]]
        count_Xi_Y = f_value_data [label]
      else:
        count_Xi_Y = 0
      count_y    = label_count [label]
      s_i        = len (fdata)
      perdicted_label_prob_dict [label] *= ((count_Xi_Y + lamda) / (count_y + (s_i * lamda))) 

  biggest = 0
  for label in unique_lables:
    current = perdicted_label_prob_dict[label]
    if (current > biggest):
      biggest = current
      best_label = label

  return best_label
def test_naive_bayes (X, Y, lamda, label_prob_dict, label_count, feature_data_list):
  no_of_rows = X.shape[0]
  unique_lables = np.unique (X[:, 0])
  mistakes = 0
  for i in range (0, no_of_rows):
    predicted_label = predict_per_row (X[i], Y, lamda, unique_lables, label_prob_dict, label_count, feature_data_list)
    if (predicted_label != X[i][0]):
      mistakes += 1

  print ("Number of mistakes : ", mistakes)

def main_func ():
  lamda = 1

#  X, Y = data_in_x_y_format ("tennis.data", 6)
  X, Y = data_in_x_y_format ("train.liblinear", 220)
  unique_labels, label_prob_dict, label_count = get_prior_label_prob (X, Y)
  feature_data_list = compute_likelihood_of_features (X, Y)
 
  X, Y = data_in_x_y_format ("test.liblinear", 220)
  test_naive_bayes (X, Y, lamda, label_prob_dict, label_count, feature_data_list)

main_func ()
