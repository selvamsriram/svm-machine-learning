import copy
import math
import numpy as np
import decision_tree as dtree

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

  print (" Mistakes   : ", mistakes)
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
  print (" Train File : ", train_filename, " | Test File : ", test_filename, " | Epochs :", epochs, " | Trade off : ", C, " | Learn Rate : ", l_rate)

  if ("trans_" in test_filename):
    transformed = True
  else:
    transformed = False

  # Separate data handlers for transformed and non transformed data
  if (transformed == True):
    X, Y = trans_data_in_x_y_format (train_filename, no_of_columns)
    new_W = svm_invoke(X, Y, W, C, l_rate, epochs, count_corrections)

    X, Y = trans_data_in_x_y_format (test_filename, no_of_columns)
    true_positive, false_positive, false_negative = svm_test (X, Y, new_W)
  else:
    X, Y = data_in_x_y_format (train_filename, no_of_columns)
    new_W = svm_invoke(X, Y, W, C, l_rate, epochs, count_corrections)

    X, Y = data_in_x_y_format (test_filename, no_of_columns)
    true_positive, false_positive, false_negative = svm_test (X, Y, new_W)

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
def get_data_and_features (filename, no_of_columns):
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
  Y = list (range (0,220))
  return X, Y

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
    new_W, precision, recall, F1 = train_and_test_svm ('temporary.data', fname_partial+str(i)+'.data',
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
      print ("")
      print (" Cross Validating values C: ", C, "Rate : ", l_rate)
      print ("-----------------------------------------------")
      W_copy = copy.deepcopy (W)
      f1 =  cross_validation (kfold, C, l_rate, epochs, no_of_columns, W_copy, "training0")
      if (f1 > best_f1):
        best_f1 = f1 
        best_C = C
        best_l_rate = l_rate 


  print ("#############################################")
  print ("Cross validation results ")
  print ("   Best Learning Rate  : ", best_l_rate)
  print ("   Best tradeof Param  : ", best_C)
  print ("   Yielded F1          : ", best_f1)
  print ("#############################################")
  # Re-init for future use
  best_f1 = 0
  best_epoch = 0
  best_precision = 0
  best_recall = 0
  best_w = np.zeros(no_of_columns - 1)

  print ("Test results")
  #Train for each epoch and test in development data for each of them and measure accuracy
  for i in range (1, 21):
    print ("")
    print (" Epoch      :", i)
    new_W, precision, recall, f1 = train_and_test_svm ('train.liblinear', 'test.liblinear', no_of_columns,
                                                   W, best_C, best_l_rate, i, 0)
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

def random_forest (train_file, sample_size, no_of_dtree, no_of_columns, depth):
  # Bagging
  print (train_file, sample_size, no_of_dtree, no_of_columns, depth)
  X, features = get_data_and_features (train_file, no_of_columns)
  randomize = np.arange (X.shape[0])

  dtree_list = []
  for i in range (0, no_of_dtree):
    np.random.shuffle(randomize)
    X = X[randomize]
    sample = X[0:sample_size]
    print ("Creating Dtree", i, "with no of features = ", len (features), "Depth : ", depth)
    root = dtree.add_node (sample, features, 0, depth)
    dtree_list.append(root)

  return dtree_list

# Construct transformed features
def transform_features_with_random_forest (input_file, transformed_file, no_of_columns_in_input, dtree_list): 
  X, features = get_data_and_features (input_file, no_of_columns_in_input)
  no_of_rows = X.shape[0]
 
  no_of_dtree = len (dtree_list)
  transformed_data = np.zeros ((no_of_rows, no_of_dtree+1))

  index_column_dict = dict(enumerate(features))
  column_index_dict = {v: k for k, v in index_column_dict.items()}
  for i in range (0, no_of_rows):
    print ("Transforming train row : ", i)
    transformed_data[i][0] = X[i][0]
    for j in range (0, no_of_dtree):
      root = dtree_list[j]
      transformed_data[i][1+j] = dtree.test_dtree_per_row (root, X[i], index_column_dict, column_index_dict)

  transformed_data.dump (transformed_file)
#  readback = np.load ("transformed_data.txt")
#  print (readback)

def svm_over_trees (train_file, test_file, sample_size, no_of_dtrees, depths, kfold, learn_rates, tradeoff_params, epochs, no_of_columns, W):
  best_f1     = 0
  best_C      = 0
  best_l_rate = 0
  best_depth  = 0
  W           = np.zeros (no_of_dtrees)

  # Cross validation Loop
  for depth in depths:
    # Create a tree with given depth
    dtree_list = random_forest(train_file, sample_size, no_of_dtrees, no_of_columns, depth)
    
    # Transform the K fold training data with the constructed decision trees
    transform_features_with_random_forest ("training00.data", "trans_training00.data", no_of_columns, dtree_list) 
    transform_features_with_random_forest ("training01.data", "trans_training01.data", no_of_columns, dtree_list) 
    transform_features_with_random_forest ("training02.data", "trans_training02.data", no_of_columns, dtree_list) 
    transform_features_with_random_forest ("training03.data", "trans_training03.data", no_of_columns, dtree_list) 
    transform_features_with_random_forest ("training04.data", "trans_training04.data", no_of_columns, dtree_list) 
    # Begin cross validation
    for C in tradeoff_params:
      for l_rate  in learn_rates:
        print ("")
        print (" Cross Validating values Depth : ", depth, "C: ", C, "Rate : ", l_rate)
        print ("------------------------------------------------------")
        W_copy = copy.deepcopy (W)
        f1 =  cross_validation (kfold, C, l_rate, epochs, no_of_dtrees+1, W_copy, "trans_training0")
      
        if (f1 > best_f1):
          best_f1 = f1 
          best_C = C
          best_l_rate = l_rate 
          best_depth = depth

  print ("#############################################")
  print ("Cross validation results ")
  print ("   Best Depth          : ", best_depth)
  print ("   Best Learning Rate  : ", best_l_rate)
  print ("   Best tradeof Param  : ", best_C)
  print ("   Yielded F1          : ", best_f1)
  print ("#############################################")

  # Transform all the training data
  trans_train_file = "trans_train.liblinear"
  trans_test_file  = "trans_test.liblinear"
  transform_features_with_random_forest (train_file, trans_train_file, no_of_columns, dtree_list) 
  transform_features_with_random_forest (test_file, trans_test_file, no_of_columns, dtree_list) 
  print (" SVM over Trees test results")
  #Train for each epoch and test in development data for each of them and measure accuracy
  best_f1 = 0
  for i in range (1, 21):
    print ("")
    print (" Epoch      :", i)
    new_W, precision, recall, f1 = train_and_test_svm (trans_train_file, trans_test_file, no_of_columns,
                                                       W, best_C, best_l_rate, i, 0)
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
  epochs          = 10 
  precision       = 0
  sample_size     = 2000
  no_of_dtrees    = 200
  depths          = [10, 20, 30]
  learn_rates     = [1, 0.1, 0.01, 0.001, 0.0001]
  tradeoff_params = [10, 1, 0.1, 0.01, 0.001, 0.0001]

  print ("******************Basic SVM Start *******************")
  print ("******************Seed Value", seed_value, "*******************")
  precision = train_test_request_processor (kfold, learn_rates, tradeoff_params, epochs, no_of_columns, W)
  print ("******************Basic SVM End *********************")

  # Random Forest data collection
  # SVM over trees
  #svm_over_trees (train_file, test_file, sample_size, no_of_dtrees, depths, kfold, learn_rates, tradeoff_params, epochs, no_of_columns, W)
  return precision 


# Start of operation
main_function (11)
