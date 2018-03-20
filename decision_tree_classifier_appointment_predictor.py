# CART algorithm Implementation

import pandas as pd 
import numpy as np 

training_data = pd.read_csv("Kaggle_appointment.csv")
print(training_data.head())
print(training_data.info())
#print(training_data.describe())
patient_id = training_data['PatientId']
appointment_ID = training_data['AppointmentID']
ScheduledDay = training_data['ScheduledDay']
AppointmentDay = training_data['AppointmentDay']
training_data.drop(['PatientId','AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)


print(training_data.head())
#print(training_data['Neighbourhood'].count_values())

header = []
for index in training_data.columns:
	header.append(index)

X = np.array(training_data)
print(X)

def unique_vals(rows, col):
	return(set(row[col] for row in rows))

print("This is the uniques values set\n",unique_vals(training_data,1))

def class_count(rows):
	count = {}
	for row in rows:
		label = row[-1]
		if label not in count:
			count[label] = 0
		count[label] +=1
	return count

print("This is the label class count\n",class_count(X))

def is_numeric(val):
	return(isinstance(val, int) or isinstance(val,float))

class Question:

	def __init__ (self, column, value):
		self.column = column
		self.value = value


	def match(self, example):
		val = example[self.column]
		if is_numeric(val):
			return val >= self.value
		else:
			return val >= self.value

	def __repr__ (self):
		condition = "=="
		if is_numeric(self.value):
			condition =">="
		return "Is %s %s %s?" %(header[self.column], condition, str(self.value))

#print(training_data.columns)
#print(training_data[0])
print(Question(1,35))
print(Question(0, "M"))

def partition(rows,question):
	true_rows, false_rows = [], []
	for row in rows:
		if question.match(row):
			true_rows.append(row)
		else:
			false_rows.append(row)
	return(true_rows, false_rows)

true_rows, false_rows = partition(training_data, Question(0, ""))
print(true_rows)
print(false_rows)

def gini(rows):
	count = class_count(rows)
	impurity = 1
	for lbl in count:
		prob_of_lbl = count[lbl]/float(len(rows))
		impurity -= prob_of_lbl**2

def info_gain(left, right, current_uncertainty):
	p = float(len(left)/len(right))
	return current_uncertainty - p*gini(left) - (1-p)*gini(right)

def find_best_split(rows):
	best_gain = 0
	best_question = 0
	n_feature = len(rows[0])-1
	for col in range(n_feature):
		values = set([row[col] for row in rows])

		for val in values:
			question = Question(col,val)
			true_rows, false_rows = partition(rows, question)

			if len(true_rows) == 0 or len(false_rows) == 0:
				continue
			gain = info_gain(true_rows, false_rows, current_uncertainty)
			if gain >= best_gain:
				best_gain = gain
				best_question = question
	return best_gain, best_question


best_gain_1, best_question_1 = find_best_split(training_data)
print(best_gain_1)
print(best_question_1)


class leaf:
	def __init__(self, rows):
		self.predictions = class_count(rows)

class Decision_node:

	def __init__(self, question, true_branch, false_branch):
		self.question = question
		self.true_branch = true_branch
		self.false_branch = false_branch


def build_tree(rows):
	gain, question = find_best_split(rows)

	if gain == 0:
		return(leaf(rows))
	true_rows, false_rows = partition(rows, question)
	true_branch = build_tree(true_rows)
	false_branch = build_tree(false_rows)

	return(Decision_node(question, true_branch, false_branch))

def print_tree(node, spacing=" "):
	if isinstance(node,leaf):
		print(spacing+ "Predict", node.predictions)
		return

	print(spacing+str(node.question))

	print(spacing+'--->True:')
	print_tree(node.true_branch, spacing+" ")

	print(spacing+'---> False')
	print_tree(node.false_branch, spacing+" ")

def classify(row, node):
	if isinstance(node, leaf):
		return node.predictions

	if node.question.match(row):
		return classify(row, node.true_branch)

	else:
		return classify(row, node.false_branch)

my_tree = build_tree(training_data)
print(my_tree)
print(print_tree(my_tree))


def print_leaf(counts):
	total = sum(counts.values())*1.0
	probs = {}
	for lbl in counts.keys():
		probs[lbl] = str(int(counts[lbl]/total*100))+"%"
	return probs

