import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from nltk import PorterStemmer
from scipy.stats import multinomial
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import porter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#*******Load CSV*****#
data=pd.read_csv("C:\\Users\\soura\\PycharmProjects\\Hello World\\data\\SMS Spam Collection Dataset\\spam.csv",encoding="latin1")
#print(data.head())
#Display maximum column width, with all the columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth',None)
#print(data.columns)
#print(data)
#print(data['v1'].value_counts(normalize=True)*100)



#****Pre-Process the file(clean unwanted data,Replace characters with numbers etc.)#
#Remove all columns apart from v1 and v2
data=data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'])
#print(data)
#print(data.isnull().sum())
sns.countplot(x='v1', data=data)
plt.title("Spam vs Ham Count")
#plt.show()
#Remove symbols & numbers in v2
data['v2']=data['v2'].str.replace(r'[^a-zA-Z]', ' ' ,regex=True)
#print(data)
#Convert to lower case in v2
data['v2']=data['v2'].str.lower()
#print(data)

#Remove stopwords in v2
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))
data['v2'] = data['v2'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
#print(data)

#Apply stemming
stemmer = PorterStemmer()
data['v2'] = data['v2'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))
#print(data)


#****Convert words to digits, split and train data#
#BoW
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(data['v2'])
#print(x)
#tfidf_vectorizer = TfidfVectorizer()
#y = data['v1']
#print(y)

#split for train and test
Y=data['v1']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

#Train data using Naive bayes
model=MultinomialNB()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

#*****Prediction*******#
# Accuracy of the model
print("Accuracy:", accuracy_score(Y_test, Y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

# Detailed classification report
print("Classification Report:\n", classification_report(Y_test, Y_pred))

sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

prediction_counts = pd.Series(Y_pred).value_counts()
print(prediction_counts)
actual_counts = Y_test.value_counts()
print("Actual counts:\n", actual_counts)





