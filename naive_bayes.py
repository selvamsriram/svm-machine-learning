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
  true_positive = 0
  false_positive = 0
  false_negative = 0
  mistakes = 0
  
  for i in range (0, no_of_rows):
    predicted_label = predict_per_row (X[i], Y, lamda, unique_lables, label_prob_dict, label_count, feature_data_list)
    if (predicted_label != X[i][0]):
      mistakes += 1
      if (X[i,0] > 0):
        false_negative += 1
      else:
        false_positive += 1
    else:
      if (X[i][0] > 0):
        true_positive += 1

  if (true_positive != 0) or (false_positive != 0):
    precision = (true_positive/(true_positive+false_positive))
  else:
    precision = 0

  if (true_positive != 0) or (false_negative != 0):
    recall = (true_positive/(true_positive+false_negative))
  else:
    recall = 0

  if (precision != 0) or (recall != 0):
    F1 = 2 * ((precision * recall) / (precision + recall))
  else:
    F1 = 0
  print (" Mistakes   : ", mistakes)
  print (" Precision  : ", precision)
  print (" Recall     : ", recall)
  print (" F Value    : ", F1)
  print ("")

  return true_positive, false_positive, false_negative, precision, recall, F1

def train_and_test_nb (train_file, test_file, lamda, no_of_columns):
  X, Y = data_in_x_y_format (train_file, no_of_columns)
  unique_labels, label_prob_dict, label_count = get_prior_label_prob (X, Y)
  feature_data_list = compute_likelihood_of_features (X, Y)
 
  X, Y = data_in_x_y_format (test_file, no_of_columns)
  true_positive, false_positive, false_negative, precision, recall, F1  = test_naive_bayes (X, Y, lamda, label_prob_dict, label_count, feature_data_list)
  return true_positive, false_positive, false_negative, precision, recall, F1

 
def cross_validation (kfold, lamda, fname_partial, no_of_columns):
  consolidated_F1 = 0
  for i in range (0, kfold):
    training_filenames = []
    temp_arr_start = True

    for j in range (0, kfold):
      if (i != j):
        training_filenames.append (fname_partial + str(j)+'.data')

    with open ('temporary.data', 'w') as temp_file:
      for fname  in training_filenames:
        with open(fname) as iterfile:
          for line in iterfile:
            temp_file.write (line)

    #Cross Validation Training
    true_positive, false_positive, false_negative, precision, recall, F1 = train_and_test_nb ('temporary.data', fname_partial+str(i)+'.data', lamda, no_of_columns) 
    consolidated_F1 += F1
  return (consolidated_F1/kfold)

def naive_bayes_train_test (train_file, test_file, lamdas, kfold, no_of_columns):
  best_f1 = 0

  for lamda in lamdas:
    f1 = cross_validation (kfold, lamda, "training0", no_of_columns)
    if f1 > best_f1:
      best_f1 = f1
      best_lamda = lamda
  
  print ("#############################################")
  print ("Cross validation results ")
  print ("   Best Lamda          : ", best_lamda)
  print ("   Yielded F1          : ", best_f1)
  print ("#############################################")

  true_positive, false_positive, false_negative, precision, recall, F1 = train_and_test_nb (train_file, test_file, best_lamda, no_of_columns) 
def main_func ():
  kfold          = 5
  lamdas         = [2, 1.5, 1, 0.5]
  no_of_columns  = 220
  train_file     = "train.liblinear"
  test_file      = "test.liblinear"

  naive_bayes_train_test (train_file, test_file, lamdas, kfold, no_of_columns)

main_func ()
