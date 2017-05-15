# import all needed libraries
from math import exp # exponentiation for the sigmoid function
from sklearn import cross_validation # split dataset into train and test
import pandas # read CSV files (dataset)
import numpy # linear algebra / calculus project
from flask import Flask, request, render_template # python web framework

# Make a prediction with coefficients
def predict(features, coefficients):
	u = coefficients[0] # bias (constant term)
	for i in range(len(features)-1):
		u += coefficients[i + 1] * features[i] # c_i * x_i
	return 1.0 / (1.0 + exp(-u)) # sigmoid function

# Estimate logistic regression coefficients using stochastic gradient descent
def gradient_descent(X_train, Y_train, l_rate, n_epoch):
	coefficients = [0.0 for i in range(len(X_train[0])+1)] # initialize coefficients with 0 (temporary)
	for epoch in range(n_epoch): # epoch = number of times to iterate over training set
		for row_ind in range(len(X_train)): # loop over each training instance (row)
			y_pred = predict(X_train[row_ind], coefficients) # use sigmoid function to predict presence of breast cancer
			error = Y_train[row_ind] - y_pred # compute error (difference between prediction and actual value)
			coefficients[0] = coefficients[0] + l_rate * error * y_pred * (1.0 - y_pred) # update bias with gradient descent
			for i in range(len(X_train[row_ind])): # loop over other constants
				coefficients[i + 1] = coefficients[i + 1] + l_rate * error * y_pred * (1.0 - y_pred) * X_train[row_ind][i] # update constant with gradient descent
	return coefficients # return computed coefficients

# Linear Regression Algorithm
def logistic_regression(X_validation, l_rate, n_epoch, coefficients): # define a function to test the algorithm
    predictions = list() # create an empty list to store predictions
    # make predictions (validate)
    for row in X_validation: # loop over each testing instance
        y_pred = round(predict(row, coefficients)) # rounded prediction
        predictions.append(y_pred) # add it to prediction list
    return(predictions) # return prediction list

# load dataset
filename = "data.csv"
dataset = pandas.read_csv(filename, dtype=numpy.float32).values

# X stores inputs
X = dataset[:,0:9]
# Y stores output
Y = dataset[:,9]
# dataset partitioned into 75% training, 25% testing
validation_size = 0.25
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size = validation_size, random_state = 1)

# evaluate algorithm
l_rate = 0.1
n_epoch = 100
# train the model
coefficients = gradient_descent(X_train, Y_train, l_rate, n_epoch)
# make predictions
predictions = logistic_regression(X_validation, l_rate, n_epoch, coefficients)
# determine accuracy
correct = 0
for i in range(len(Y_validation)):
    if Y_validation[i] == predictions[i]:
        correct += 1

# Print statistics
# print('Accuracy:', correct / float(len(Y_validation)) * 100.0)
# print(coefficients)

# set up website
app = Flask(__name__)
# route for the input form
@app.route("/")
def form():
	return render_template("index.html")

# process the data
@app.route("/", methods=["POST"])
def form_post():
	clumpThickness = request.form.get("clumpThickness")
	# clumpThickness = request.form["clumpThickness"]
	# clumpThickness = float(request.form.get("clumpThickness"))
	# cellSize = float(request.form.get("cellSize"))
	# cellShape = float(request.form.get("cellShape"))
	# marginalAdhesion = float(request.form.get("marginalAdhesion"))
	# epithelialCellSize = float(request.form.get("epithelialCellSize"))
	# bareNuclei = float(request.form.get("bareNuclei"))
	# chromatin = float(request.form.get("chromatin"))
	# nucleoli = float(request.form.get("nucleoli"))
	# mitoses = float(request.form.get("mitoses"))
	data = {
		"title": "Diagnosis Results",
		"results": clumpThickness
		#"results": "Model prediction: " + predict([clumpThickness, cellSize, cellShape, marginalAdhesion, epithelialCellSize, bareNuclei, chromatin, nucleoli, mitoses], coefficients)
	}
	return render_template("index.html",**data)

# run the website
app.run()
