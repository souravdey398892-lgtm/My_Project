import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Load Dataset
data=pd.read_csv("./data/titanic/train.csv")
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
#pd.set_option('display.max_colwidth',None)
pd.set_option('display.width',2000)
#Printing data
print(data)
#Fill the nAN values on 'Age' column
data['Age'].fillna(data['Age'].median(),inplace=True)
print(data)
#Removing unwanted columns
data.drop(["Name","Ticket"],axis=1, inplace=True)
print(data)
#Plotting bar graph
sns.barplot(x="Age",y="Survived",data=data)
plt.show()
#Plotting Bar Graph
sns.barplot(x="Pclass",y="Survived",data=data)
plt.show()
#Plotting Histogram
sns.histplot(data[data["Survived"]==1]["Age"],bins=20,kde=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()