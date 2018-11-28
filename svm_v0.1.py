import copy
import math
import numpy as np

# Peceptron Tester Function
#--------------------------------------------------------------------------------------------------
def svm_test(X, Y, W):
  rows = X.shape[0]
  true_positive = 0
  false_positive = 0
  false_negative = 0
  mistakes = 0

  for i in range (0, rows):
    val = (np.dot(X[i], W))
    if (val * Y[i,0]) < 0:
      mistakes += 1
      if (Y[i,0] > 0):
        false_negative += 1
      else:
        false_positive += 1
    else:
      if (Y[i,0] > 0):
        true_positive += 1

  print ("Total Mistakes : ", mistakes)
  return true_positive, false_positive, false_negative

# Perceptron Learner Function
#--------------------------------------------------------------------------------------------------
def svm_invoke(X, Y, W, C, l_rate, epochs, count_corrections):
    update_count = 0
    cols = X.shape[1]
    rows = X.shape[0]
    for t in range(0, epochs):
        randomize = np.arange (X.shape[0])
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
        rate = (l_rate/(1+t))
        
        for i in range (0, rows):
          if ((np.dot(X[i], W)*Y[i,0]) <= 1):
            W = ((1-rate)*W) + (rate * C * Y[i,0] * X[i])
          else:
            W = (1-rate)*W

    return W

# Train and test function
#--------------------------------------------------------------------------------------------------
def train_and_test_svm (train_filename, test_filename, no_of_columns, W, C,
                        l_rate, epochs, count_corrections):
  print (train_filename, test_filename, no_of_columns, C, l_rate, epochs)
  X, Y = data_in_x_y_format (train_filename, no_of_columns)
  new_W = svm_invoke(X, Y, W, C, l_rate, epochs, count_corrections)

  X, Y = data_in_x_y_format (test_filename, no_of_columns)
  true_positive, false_positive, false_negative = svm_test (X, Y, new_W)

  print (true_positive, false_positive, false_negative)
  if (true_positive != 0) and (false_positive != 0):
    precision = (true_positive/(true_positive+false_positive))
  else:
    precision = 0

  if (true_positive != 0) and (false_negative != 0):
    recall    = (true_positive/(true_positive+false_negative))
  else:
    recall = 0

  print (precision, recall)
  if (precision != 0) and (recall != 0):
    F1        = 2 * ((precision * recall) / (precision + recall))
  else:
    F1 = 0
  return new_W, precision, recall, F1 

# Utility Function
#--------------------------------------------------------------------------------------------------
def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

# Utility Function
#--------------------------------------------------------------------------------------------------
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

  raw_Y = copy.deepcopy(data)
  Y = np.delete (raw_Y, np.s_[1:no_of_columns], axis=1)
  X = np.delete (data, 0, axis=1)
  return X, Y

# Cross validation function
#--------------------------------------------------------------------------------------------------
def cross_validation (kfold, C, l_rate, epochs, no_of_columns, W):
  precision = 0
  consolidated_F1 = 0
  for i in range (0, kfold):
    training_filenames = []
    for j in range (0, kfold):
      if (i != j):
        training_filenames.append ('training0'+str(j)+'.data')

    with open ('temporary.data', 'w') as temp_file:
      for fname  in training_filenames:
        with open(fname) as iterfile:
          for line in iterfile:
            temp_file.write (line)

    #Cross Validation Training
    new_W, precision, recall, F1 = train_and_test_svm ('temporary.data', 'training0'+str(i)+'.data',
                                                  no_of_columns, W, C, l_rate, epochs, 0)
    consolidated_F1 += F1 
  return (consolidated_F1/kfold)

# Mother ship
#--------------------------------------------------------------------------------------------------
def train_test_request_processor (kfold, learn_rates, tradeoff_params, epochs,
                                  no_of_columns, W):
  best_f1 = 0

  for C in tradeoff_params:
    for l_rate  in learn_rates:
      W_copy = copy.deepcopy (W)
      f1 =  cross_validation (kfold, C, l_rate, epochs, no_of_columns, W_copy)
      if (f1 > best_f1):
        best_f1 = f1 
        best_C = C
        best_l_rate = l_rate 


  print ("Cross validation results ")
  print ("   Best Learning Rate  : ", best_l_rate)
  print ("   Best tradeof Param  : ", best_C)
  print ("   Yielded F1          : ", f1)
  # Re-init for future use
  best_f1 = 0
  best_epoch = 0
  best_w = np.zeros(no_of_columns - 1)

  print ("Development set results")
  #Train for each epoch and test in development data for each of them and measure accuracy
  for i in range (1, 21):
    new_W, precision, recall, f1 = train_and_test_svm ('train.liblinear', 'test.liblinear', no_of_columns,
                                                   W, best_C, best_l_rate, i, 0)
    print ("   Epoch : %-4d   Precision : %-10.10f  Recall : %-10.10f F1 : %-10.10f" % (i, precision, recall, f1))

    if (f1 > best_f1):
      best_f1 = f1 
      best_epoch = i
      best_w = copy.deepcopy (new_W)

  print ("   Best epoch                   : ", best_epoch)
  print ("   Best F1                      : ", best_f1, "%")

  return best_f1 

# Majority baseline
#------------------------------------------------------------------------------------------------
def majority_baseline (train_file, dev_file, test_file, no_of_columns):
 positive_count = 0
 negative_count = 0
 error_count = 0
 prediction = 0

 X, Y = data_in_x_y_format (train_file, no_of_columns)
 for i in range (0, Y.shape[1]):
   if (Y[i] < 0):
     negative_count += 1
   else:
     positive_count += 1

   if (positive_count > negative_count):
     prediction = 1
   else:
     prediction = -1

 X, Y = data_in_x_y_format (dev_file, no_of_columns)
 for i in range (0, X.shape[0]):
   if (float(Y[i]) != prediction):
     error_count += 1

 print ("Dev File Accuracy   :", (1 - (error_count/X.shape[0]))*100)

 error_count = 0
 X, Y = data_in_x_y_format (test_file, no_of_columns)
 for i in range (0, X.shape[0]):
   if (Y[i] != prediction):
     error_count += 1

 print ("Test File Accuracy  :", (1 - (error_count/X.shape[0]))*100)

# Main Function Starts here 
#--------------------------------------------------------------------------------------------------
def main_function (seed_value):
  kfold           = 5
  no_of_columns   = 220
  np.random.seed (seed_value)
  W               = np.zeros (no_of_columns-1)
  epochs          = 2 
  precision       = 0
  learn_rates     = [1, 0.1, 0.01, 0.001, 0.0001]
  tradeoff_params = [10, 1, 0.1, 0.01, 0.001, 0.0001]

  print ("******************Basic SVM Start *******************")
  print ("******************Seed Value", seed_value, "*******************")
  precision = train_test_request_processor (kfold, learn_rates, tradeoff_params, epochs, no_of_columns, W)
  print ("******************Basic SVM End *********************")
  return precision 


# Start of operation
main_function (1)
