# Lets do our imports
from math import e
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neural_network
import openpyxl


# Make a function to display prediction results of this ML app
# Make things nicer for the user
def displayPredictionResults(numeric_result):
    if numeric_result[0] == 0:
        print("\nI predict you will not die of heart disease!\n\n")
    else:
        print("\nI predict you will likely die of heart disease!\n\n")


# define the file
xls_data_file = r"heart.xlsx"
# Read the file into a data frame
df = pd.read_excel(xls_data_file)
# Print out a demo for the user to see
print("\n\t*** Will you die of heart disease? *** \n")
# Select only the columns I want to use as features from the data frame
# which was from the excel file
features = df[["age", "anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium"]]
# Select only the columns I want to use as labels from the data frame
# which was from the excel file
labels = df[["DEATH_EVENT"]]
# Let's prepare to use a decision tree classifier to train our app here
clf = tree.DecisionTreeClassifier()
# Let's train our app here - This is the line that does the training, not the prediction!
clf = clf.fit(features, labels)
# First hard code some test examples to have this app predict
print("\n0 = No death, 1 = Death")
print("This should be an example of a death.")
print("Age = 90, Anaemia = 1, diabetes = 1, high_blood_pressure = 1, sex = 1, smoke = 1, creatinine_phosphokinase = 7932, ejection_fraction = 15, platelets = 162000, serum_creatinine = 1.1, serum_sodum = 100")
# Let's get a prediction using this data from our app
result = clf.predict([[90, 1, 1, 1, 1, 1, 7932, 15, 162000, 1.1, 100]])
# Show prediction results to user
displayPredictionResults(result)

print("_" * 50)
print("\n\t *** Training Data this app was trained on *** \n", df)
print("_" * 50)

# Putting together testing and training data for the score predictions
features_training_data, features_testing_data, labels_training_data, labels_testing_data = train_test_split(features, labels, test_size=.45)

# Assembling linear regression model and percentage
linear_regression = linear_model.LinearRegression()
# Inputting training data
linear_regression = linear_regression.fit(features_training_data, labels_training_data)
# Assembling prediction
predictions_for_linear_regression = linear_regression.predict(features_testing_data)
# Creating prediction percentage
predection_percent_score_LR = accuracy_score(labels_testing_data, predictions_for_linear_regression.round())
percentage = "{:.0%}​​".format(predection_percent_score_LR)
print("ALERT: The Linear Regression Accuracy Score of this model is: ", percentage)

# Assembling neural network model and percentage
classifier_neural_network = neural_network.MLPClassifier()
# Putting in training data
classifier_neural_network = classifier_neural_network.fit(features_training_data, np.ravel(labels_training_data))
# Creating prediction model
predictions_for_classifier_neural_network = classifier_neural_network.predict(features_testing_data)
# Putting together a percentage
predection_percent_score_ANN = accuracy_score(labels_testing_data, predictions_for_classifier_neural_network)
percentage = "{:.0%}​​".format(predection_percent_score_ANN)
print("ALERT: The Neural network Classifier Accuracy score is ", percentage)

print("\n\t *** Now type in your medical information ***\n")
print("\n\t *** For \"yes or no\" questions, please type 0 for \"no\" and 1 for \"yes.\"")
ageInput = input("What is your age?\n")
anaemiaInput = input("Do you have anaemia?\n")
diabetesInput = input("Do you have diabetes?\n")
bloodInput = input("Do you have high blood pressure?\n")
sexInput = input("What is your sex? (Male = 1, Female = 0\n")
smokeInput = input("Do you smoke?\n")
print("CPK (Creatine Phosphokinase) is an enzyme found in your heart, brain, and skeletal muscles. Levels of it can be tested by a blood test.\n")
creatinineInput = input("What is the level of CPK enzyme in your blood? (mcg\L)\n")
ejectionInput = input("What is the percetnage of blood leaving your heart at each contraction? (percentage)\n")
print("Platelets are cells that circulate within your blood and bind together when they recognize damaged blood vessels to clot them. You can test how many platelets you have via a blood test.")
plateletInput = input("How many platelets are in your blood? (kiloplatelets/mL)\n")
print("Creatinine is a chemical compound left over from energy-producing processes in your muscles. They are a measure of how well your kidneys filter waste from your blood. They can be tested through a blood test.")
serumCreatinineInput = input("What is the level of serum creatinine in your blood? (mg/dL)\n")
print("Sodium is an essential mineral to your body, but having too much of it can raise your blood pressure and increase the likelihood of heart disease. Levels of sodium can be measured via a blood test.")
serumSodiumInput = input("What is the level of serum sodium in your blood? (mEq/L)\n")
result = clf.predict([[ageInput, anaemiaInput, diabetesInput, bloodInput, sexInput, smokeInput, creatinineInput, ejectionInput, plateletInput, serumCreatinineInput, serumSodiumInput]])
displayPredictionResults(result)

print(" *** END OF PROGRAM *** ")