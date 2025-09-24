import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#from Linear_reg import x_train

data=pd.read_csv(".\\data\\Student\\student-mat-pass-or-fail.csv")
#print(data)
pd.set_option('display.max_columns', None)
#print(data)
pd.set_option('display.width', 300)
#print(data.columns)
to_keep=["failures","absences","G1","G2","studytime","pass"]
data_student=data.drop(columns=[c for c in data.columns if c not in to_keep])
#print(data_student)
#print(data_student.head())

x=data_student[["failures","absences","G1","G2","studytime"]]
y=data_student["pass"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=123)

#######Prediction using Decision Tree Model########

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=123)
# Train the model
dt_model.fit(x_train, y_train)
# Make predictions
y_pred_dt = dt_model.predict(x_test)
print(y_pred_dt)

#Print accuracy of prediction
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
#print confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
#Details of each matrix
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

#Find the important features
importances = dt_model.feature_importances_
feature_importances = sorted(zip(importances, x_train.columns), reverse=True)
#feat_importances = pd.Series(importances, index=x_train.columns)
for importance, name in feature_importances:
    print(name, ":", importance)




#######Prediction using Random Forest########
# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
# Train the model
rf_model.fit(x_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(x_test)

# Evaluate
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Feature Importance
importances=rf_model.feature_importances_
feature_importances=sorted(zip(importances, x_train.columns),reverse=True)
print(feature_importances)

for importance, feature in feature_importances:
    print(f"{feature}: {importance:.3f}")

# Comparing accuracy between Decision tree model and Random Forest Model
print("Decision Tree:", dt_model.score(x_test, y_test), " | Random Forest:", rf_model.score(x_test, y_test))

