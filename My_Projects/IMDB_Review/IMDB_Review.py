from itertools import count

import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,ConfusionMatrixDisplay
#Load CSV
data=pd.read_csv(".\\data\\IMDB_Dataset\\IMDB Dataset.csv")
#print(data)
#Check the count of positive and negative counts
print(data['sentiment'].value_counts())
#check for missing values
print(data.isnull().sum())
#Remove duplicate rows and check for duplicates
#print(data.duplicated().sum())
data=data.drop_duplicates()
print(data.duplicated().sum())

#Pre-Processing starts here*********************************#
#Remove symbols and numbers in review column
data['review']=data['review'].str.replace(r'[^a-zA-Z]',' ',regex=True)
#print(data['review'])
#print(data['review'].str.contains(r'<.*?>',regex=True).sum())
#convert all the words in lowercase letters
data['review']=data['review'].str.lower()
#Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['review']=data['review'].apply(lambda x:' '.join(word for word in x.split() if word not in stop_words))
#Apply lemmatization
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
data['review'] = data['review'].apply(
    lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split())
)
#Convert words to digits
#Apply BoW
# Create the vectorizer
vectorizer = CountVectorizer()
# Fit to your review column and transform it into BoW
X_Counts = vectorizer.fit_transform(data['review'])
#print(X.shape)
#Apply TF-IDF
tfidf=TfidfTransformer()
X = tfidf.fit_transform(X_Counts)
#print(X)
print(X.shape)

#****************Now Split data for training*************************#
Y=data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
#Model selection and prediction
model=MultinomialNB()
model.fit(X_train,y_train)

#**********************Predictions*************************#

y_pred=model.predict(X_test)
print(y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=['negative', 'positive']).plot(cmap='Blues')
print(confusion_matrix(y_test, y_pred))
plt.show()

#**********************Check if the model can predict random comments**********#
prediction=model.predict(tfidf.transform(vectorizer.transform(["This movie is great"])))
print(prediction)