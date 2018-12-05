import copy
import math
import numpy as np
import decision_tree as dtree

# Peceptron Tester Function
#--------------------------------------------------------------------------------------------------
def logistic_regression_test(X, Y, W):
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

  print (" Mistakes   : ", mistakes)
  return true_positive, false_positive, false_negative

# Perceptron Learner Function
#--------------------------------------------------------------------------------------------------
def logistic_regression_invoke(X, Y, W, tradeoff, l_rate, epochs, count_corrections):
  update_count = 0
  cols = X.shape[1]
  rows = X.shape[0]
  for t in range(0, epochs):
#    randomize = np.arange (X.shape[0])
#    np.random.shuffle(randomize)
#    X = X[randomize]
#    Y = Y[randomize]
    rate = (l_rate/(1+t))

    for i in range (0, rows):
      try:
        gradient = ((-1 * X[i] * Y[i,0])/(1 + math.exp (Y[i] * np.dot (W, X[i])))) + ((2*W)/(tradeoff))
      except OverflowError:
        gradient = ((2*W) / tradeoff)

      W = W - rate * gradient
  return W

# Train and test function
#--------------------------------------------------------------------------------------------------
def train_and_test_logistic_regression (train_filename, test_filename, no_of_columns, W, C,
                        l_rate, epochs, count_corrections):
  print (" Train File : ", train_filename, " | Test File : ", test_filename, " | Epochs :", epochs, " | Trade off : ", C, " | Learn Rate : ", l_rate)

  if ("trans_" in test_filename):
    transformed = True
  else:
    transformed = False

  # Separate data handlers for transformed and non transformed data
  if (transformed == True):
    X, Y = trans_data_in_x_y_format (train_filename, no_of_columns)
    new_W = logistic_regression_invoke(X, Y, W, C, l_rate, epochs, count_corrections)

    X, Y = trans_data_in_x_y_format (test_filename, no_of_columns)
    true_positive, false_positive, false_negative = logistic_regression_test (X, Y, new_W)
  else:
    X, Y = data_in_x_y_format (train_filename, no_of_columns)
    new_W = logistic_regression_invoke(X, Y, W, C, l_rate, epochs, count_corrections)

    X, Y = data_in_x_y_format (test_filename, no_of_columns)
    true_positive, false_positive, false_negative = logistic_regression_test (X, Y, new_W)

  print (" True +ve   : ", true_positive)
  print (" False +ve  : ", false_positive)
  print (" False -ve  : ", false_negative)

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
  print (" Precision  : ", precision)
  print (" Recall     : ", recall)
  print (" F Value    : ", F1)
  print ("")
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
def trans_data_in_x_y_format (filename, no_of_columns):
  data = np.load (filename)
  raw_Y = copy.deepcopy(data)
  Y = np.delete (raw_Y, np.s_[1:no_of_columns], axis=1)
  X = np.delete (data, 0, axis=1)
  return X, Y

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
def cross_validation (kfold, C, l_rate, epochs, no_of_columns, W, fname_partial):
  precision = 0
  consolidated_F1 = 0

  if ("trans_" in fname_partial):
    transformed = True
  else:
    transformed = False

  for i in range (0, kfold):
    training_filenames = []
    temp_arr_start = True
    for j in range (0, kfold):
      if (i != j):
        training_filenames.append (fname_partial + str(j)+'.data')

    if (transformed == False):
      with open ('temporary.data', 'w') as temp_file:
        for fname  in training_filenames:
          with open(fname) as iterfile:
            for line in iterfile:
              temp_file.write (line)
    else:
      for fname in training_filenames:
        transient_arr = np.load (fname)
        if temp_arr_start == True:
          temp_arr_start = False
          temp_arr = copy.deepcopy (transient_arr)
        else:
          temp_arr = np.concatenate ((temp_arr, transient_arr))
      temp_arr.dump ("temporary.data")

    #Cross Validation Training
    new_W, precision, recall, F1 = train_and_test_logistic_regression ('temporary.data', fname_partial+str(i)+'.data',
                                                  no_of_columns, W, C, l_rate, epochs, 0)
    consolidated_F1 += F1 
  return (consolidated_F1/kfold)

# Mother ship
#--------------------------------------------------------------------------------------------------
def train_test_request_processor (kfold, learn_rates, tradeoff_params, epochs,
                                  no_of_columns, W):
  '''
  best_f1 = 0
  for C in tradeoff_params:
    for l_rate  in learn_rates:
      print ("")
      print (" Cross Validating values C: ", C, "Rate : ", l_rate)
      print ("-----------------------------------------------")
      W_copy = copy.deepcopy (W)
      f1 =  cross_validation (kfold, C, l_rate, epochs, no_of_columns, W_copy, "training0")
      if (f1 > best_f1):
        best_f1 = f1 
        best_tradeoff = C
        best_l_rate = l_rate 


  print ("#############################################")
  print ("Cross validation results ")
  print ("   Best Learning Rate  : ", best_l_rate)
  print ("   Best tradeof Param  : ", best_tradeoff)
  print ("   Yielded F1          : ", best_f1)
  print ("#############################################")
  '''
  # Re-init for future use
  best_f1 = 0
  best_epoch = 0
  best_precision = 0
  best_recall = 0
  best_w = np.zeros(no_of_columns - 1)
  
  print ("Test results")
  #Train for each epoch and test in development data for each of them and measure accuracy
  for i in range (1, 2):
    print ("")
    print (" Epoch      :", i)
    best_tradeoff = 10
    best_l_rate = 1
    new_W, precision, recall, f1 = train_and_test_logistic_regression ('train.liblinear', 'test.liblinear', no_of_columns,
                                                   W, best_tradeoff, best_l_rate, i, 0)
    print (" Precision  : ", precision)
    print (" Recall     : ", recall)
    print (" F1         : ", f1)
    if (f1 > best_f1):
      best_f1 = f1 
      best_precision = precision
      best_recall = recall
      best_epoch = i
      best_w = copy.deepcopy (new_W)

  print ("##############################################")
  print ("   Best epoch                   : ", best_epoch)
  print ("   Best F1                      : ", best_f1, "%")
  print ("##############################################")

  return best_f1 

# Main Function Starts here 
#--------------------------------------------------------------------------------------------------
def main_function (seed_value):
  train_file      = "train.liblinear"
  test_file       = "test.liblinear"
  kfold           = 5
  no_of_columns   = 220
  np.random.seed (seed_value)
  W               = np.zeros (no_of_columns-1)
  epochs          = 5 
  precision       = 0
  learn_rates     = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
  tradeoff_params = [0.1, 1, 10, 100, 1000, 10000]

  print ("******************Seed Value", seed_value, "*******************")
  precision = train_test_request_processor (kfold, learn_rates, tradeoff_params, epochs, no_of_columns, W)
  print ("******************Basic SVM End *********************")
  return precision 


# Start of operation
main_function (11)
