import copy
import math
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------
#   This program does the following
#   1. Uses the ID3 algorithm and constructs the decision tree with provided training data (train.csv)
#   2. Supports validating the tree with given test data. (test.csv)
#   3. Allows K-Fold cross validation testing, file names should be like (fold1.csv, fold2.csv ...)
#   4. Supports limiting the depth of the tree add_node () gives user an option to enter allowed depth in code.
#   5. Displays average accuracy and standard deviation of results using K-Fold cross validation and depth limiting.
#
#   Notes : 
#   train.csv and test.csv files should have the label in the column Zero.
#------------------------------------------------------------------------------------------------------------------------

#1 = enabled 0 = disabled
debug = 1

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

#  raw_Y = copy.deepcopy(data)
#  Y = np.delete (raw_Y, np.s_[1:no_of_columns], axis=1)
  X = data
  Y = list(range (1, 220))
#  X = np.delete (data, 0, axis=1)
  return X, Y

class Node:
    def __init__(self, feature):
        self.feature = feature 
        self.children = []
        self.decision = ""

    def __str__(self):
        return self.feature

class Attribute:
	pass

class Data:
	
	def __init__(self, *, fpath = "", data = None):
		
		if not fpath and data is None:
			raise Exception("Must pass either a path to a data file or a numpy array object")

		self.raw_data, self.attributes, self.features, self.index_column_dict, \
		self.column_index_dict = self._load_data(fpath, data)

	def _load_data(self, fpath = "", data = None):
		
		if data is None:
			data = np.loadtxt(fpath, delimiter=',', dtype = str)

		header = data[0]
		index_column_dict = dict(enumerate(header))

		#Python 2.7.x
		# column_index_dict = {v: k for k, v in index_column_dict.items()}

		#Python 3+
		column_index_dict = {v: k for k, v in index_column_dict.items()}

		data = np.delete(data, 0, 0)

		attributes = self._set_attributes_info(index_column_dict, data)

		return data, attributes, header, index_column_dict, column_index_dict
	
	def _set_attributes_info(self, index_column_dict, data):
		attributes = dict()

		for index in index_column_dict:
			column_name = index_column_dict[index]
			if column_name == 'label':
				continue
			attribute = Attribute()
			attribute.name = column_name
			attribute.index = index - 1
			attribute.possible_vals = np.unique(data[:, index])
			attributes[column_name] = attribute

		return attributes

def information_gain_per_column (subset_data, features, col):
    # Compute the entropy of the labels first
    dict = {}
    unique_labels = np.unique (subset_data[:, 0])
    if unique_labels.shape[0] == 1:
        label_entropy = 0
        return label_entropy
    else:
        count = np.zeros ((unique_labels.shape[0], 1))
        for i in range (0, unique_labels.shape[0]):                     # For each unique label
            for j in range (0, subset_data[:, 0].shape[0]):             # For each row in the label column
                if subset_data[j, 0] == unique_labels[i]:
                    count [i] += 1                                      # Count of +ve or -ve

    row_size = subset_data[:, 0].shape[0]

    i = 0
    label_entropy = 0
    for i in range (0, unique_labels.shape[0]):
        division_result = (count[i]/row_size)
        label_entropy -= (division_result) * math.log (division_result, 2)

    #print ("Label Entropy : ", label_entropy)

	# See how many different inputs are there in the column
    unique_values = np.unique (subset_data[:, col])

	# For each different type of input in a column, compute the entropy

    # Find how many different types of values are there and how many times each value occurs
    i = 0
    no_of_occurence_per_value = np.zeros (unique_values.shape[0])

    for i in range (0, unique_values.shape[0]):
        for j in range (0, subset_data[:, 0].shape[0]):
            if subset_data [j, col] == unique_values [i]:
                no_of_occurence_per_value [i] += 1

    # Debug print
    #for i in range (0, unique_values.shape[0]):
    #   print (features[col], ":", unique_values[i], ":", no_of_occurence_per_value[i])

    entropy_per_attr_value = np.zeros (unique_values.shape[0])
	# Find the the +ve and -ve outcomes for each different type of input
    for i in range (0, unique_values.shape[0]):
        # Get the subset rows to compute entropy of this attribute value
        count = np.zeros ((unique_labels.shape[0], 1))
        for j in range (0, unique_labels.shape[0]):
            #Check how many times particular label comes for each attribute value
            for k in range (0, subset_data[:, 0].shape[0]):
                if subset_data [k, col] == unique_values[i]:
                    if subset_data [k, 0] == unique_labels[j]:
                        count [j] += 1
        #print (features[col], "Value : ", unique_values[i], unique_labels, count)
        for j in range (0, unique_labels.shape[0]):
            if count[j] != 0:
                division_result = (count[j]/no_of_occurence_per_value[i])
                entropy_per_attr_value [i] -= (division_result) * math.log (division_result, 2)

	# Compute expected entropy
    expected_entropy_for_column = 0
    for i in range (0, unique_values.shape[0]):
        expected_entropy_for_column += (no_of_occurence_per_value[i]/row_size) * entropy_per_attr_value [i]

    #print ("Expected Entropy", expected_entropy_for_column)

	# Compute Information Gain = H (Label) - Expected entropy (Column)
    info_gain = label_entropy - expected_entropy_for_column

    #print ("Information Gain for :", features[col], ":", info_gain)
	# Returns information gain per column
    return info_gain 

def information_gain_all_columns (subset_data, features):
    # Find how many different columns are there and loop for all of them
    no_of_col = subset_data.shape[1]

    # Create a single dimensional array to save all the information gain computed
    info_gain_per_column = np.zeros (no_of_col)

    # Call information_gain_per_column (subset_data, col)
    for col in range (1, no_of_col):
        info_gain_per_column[col] = information_gain_per_column (subset_data, features, col)

    # Return the information_gain_per_column array
    return info_gain_per_column 

def get_next_feature (subset_data, features):
    # Call information_gain_all_columns (subset_data, features)
    # information_gain_all_columns (subset_data, features)
    # Select the attribute that has the maximum value in information_gain_per_column []
    # Return the Attribute Name and Column index
    info_gain_per_column = information_gain_all_columns (subset_data, features)
    next_col_index = np.argmax (info_gain_per_column)
    return next_col_index, features[next_col_index] 

def get_row_subset(data, col_index, attribute_value):
    subset_data = copy.deepcopy(data)
    subset_data = subset_data [subset_data[:, col_index] == attribute_value]
    return subset_data

def add_node (subset_data, features, current_depth, allowed_depth):
    # If the every label is either +ve or -ve then the tree is complete at this branch, add the decision and return """

    # Check if the depth limit has been reached
    if current_depth >= allowed_depth:
        # We have got a problem here, depth has run out.
        # See how many +ve and -ve outcomes are there, select the majority
        unique_labels = np.unique (subset_data[:, 0])
        if unique_labels.shape[0] == 1:
            node = Node ("")
            node.decision = np.unique (subset_data[:, 0])[0]
            return node
        else:
            count = np.zeros ((unique_labels.shape[0], 1))
            for i in range (0, unique_labels.shape[0]):                     # For each unique label
                for j in range (0, subset_data[:, 0].shape[0]):             # For each row in the label column
                    if subset_data[j, 0] == unique_labels[i]:
                        count [i] += 1                                      # Count of +ve or -ve
        decision_index = np.argmax (count)
        node = Node ("")
        node.decision = unique_labels[decision_index]
        return node
 
    # Take the label column and check if everything is same """
    unique_labels = np.unique (subset_data[:, 0]).shape[0]
    if unique_labels == 1:
        # Take action now, and assign this as the decision for this branch
        #if debug == 1:
        #    print ("Labels are the same")
        node = Node ("")
        node.decision = np.unique (subset_data[:, 0])[0]
        return node
    #elif debug == 1:
     #       print ("unique labels", unique_labels)

    #The tree is not complete yet. Because the labels differ, branching possible """

    #Check which can be the new feature that is added to the tree by information gain """
    selected_col_index, selected_feature = get_next_feature (subset_data, features)
    selected_feature_unique_values = np.unique (subset_data[:, selected_col_index])

    node = Node (selected_feature)
    #print ("selected_col_index : ", selected_col_index, ",selected_feature : ", selected_feature, ",Value Count", selected_feature_unique_values.shape[0])
    for i in range (0, selected_feature_unique_values.shape[0]):
        # Alter the subset data here
        new_subset = get_row_subset (subset_data, selected_col_index, selected_feature_unique_values[i]) 
        # Delete the selected column from the new_subset
        new_subset = np.delete(new_subset, selected_col_index, 1)
        # Delete from the feature subset also
        new_feature_subset = copy.deepcopy (features)
        new_feature_subset = np.delete (new_feature_subset, selected_col_index, 0) 
        # Call the function again with subset data
        child_node = add_node (new_subset, new_feature_subset, current_depth+1, allowed_depth)
        # Add the node to the tree
        node.children.append((selected_feature_unique_values[i], child_node))

    return node

def test_dtree_per_row (node, test_row, col_index_feature_dict, feature_col_index_dict):
    if node.decision != "":
        return node.decision

    col_index = feature_col_index_dict[node.feature]
    col_value = test_row[col_index]

    for child in node.children :
        if col_value == child[0]:
            return test_dtree_per_row (child[1], test_row, col_index_feature_dict, feature_col_index_dict)

def test_dtree(root, test_data, features):
    no_of_rows = test_data.shape[0]
    index_column_dict = dict(enumerate(features))
    column_index_dict = {v: k for k, v in index_column_dict.items()}
    mismatch = 0
    match = 0
    for i in range (0, no_of_rows):
        id3_result = test_dtree_per_row (root, test_data [i], index_column_dict, column_index_dict)
        #print ("Original Result", test_data [i][0], "ID3 Result", id3_result)
        if id3_result != test_data [i][0]:
            mismatch += 1
        else:
            match += 1
    
    #print ("Mismatch : ", mismatch, "Match", match)
    accuracy = match/no_of_rows
    #print ("Accuracy : ", accuracy*100, "%")
    return (accuracy*100)

'''
#-------------------------------- Main function calls start below this ---------------
mango = Data (fpath = "train.csv")
root = add_node (mango.raw_data, mango.features, 0, 100)
test = Data (fpath = "test.csv")
accuracy = test_dtree (root, test.raw_data, test.features)
print ("Depth              : Unlimited")
print ("Training File      : train.csv")
print ("Testing File       : test.csv")
print ("Accuracy           :", accuracy, "%\n")

#----------------------------------- K Fold and Depth Limiting -----------------------
depths = [10, 20, 30]
kfold  = 5

print (kfold, "Fold cross validation Results as follows,")
for depth in depths:
    accuracy = np.zeros(5)
    for i in range (1, kfold+1):
        tmp_array = []
        for j in range (1, kfold+1):
            if (i != j):
                if not tmp_array:
                    tmp_array = [np.loadtxt ('fold'+str(j)+'.csv', delimiter=',', dtype = str)]
                else:
                    tmp_array.append (np.loadtxt ('fold'+str(j)+'.csv', delimiter=',', dtype = str, skiprows = 1))

        train_data = np.concatenate (tmp_array)
        mango = Data (data = train_data)
        root = add_node (mango.raw_data, mango.features, 0, depth)
        test_data = Data (fpath = 'fold'+str(i)+'.csv')
        accuracy[i - 1] = test_dtree (root, test_data.raw_data, test_data.features)

    avg_accuracy = np.mean (accuracy)
    std_dev = np.std (accuracy)
    print ("Depth              :", depth)
    print ("Average Accuracy   :", avg_accuracy, "%")
    print ("Standard Deviation :", std_dev, "\n")
# ------------------------ K Fold and Depth Limiting code ends -------------------------
'''
