\# 📩 SMS Spam Classifier



A simple machine learning project to classify SMS messages as \*\*Spam\*\* or \*\*Ham (Not Spam)\*\* using \*\*Naive Bayes\*\* and \*\*CountVectorizer\*\*.



---



\## 📘 Description

This project builds a spam detection model using text preprocessing, feature extraction, and machine learning. It trains on the \*\*SMS Spam Collection Dataset\*\* and predicts whether a message is spam or not.



---



\## 📊 Dataset

\- \*\*Source:\*\* SMS Spam Collection Dataset (`spam.csv`)

\- \*\*Columns Used:\*\*  

&nbsp; - `v1`: Label (spam/ham)  

&nbsp; - `v2`: Message text



---



\## 🧹 Preprocessing

\- Removed unwanted columns  

\- Cleaned text (removed symbols \& numbers)  

\- Converted to lowercase  

\- Removed stopwords  

\- Applied stemming (Porter Stemmer)



---



\## 🧠 Model Training

\- Feature extraction using \*\*CountVectorizer\*\*

\- Model: \*\*Multinomial Naive Bayes\*\*

\- Split: 80% training / 20% testing



---



\## 📈 Results

\- Accuracy: ~98%  

\- Evaluation: Confusion Matrix \& Classification Report



---



\## ⚙️ How to Run

```bash

Clone Repo: git clone <>


&nbsp;





\##Install Required libraries

pip install numpy pandas matplotlib seaborn scikit-learn nltk

