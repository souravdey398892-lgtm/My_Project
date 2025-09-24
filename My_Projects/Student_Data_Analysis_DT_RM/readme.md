Project Title:

Student Performance Classifier



Description:

This project predicts whether a student will \*\*pass or fail\*\* based on their study-related features.  

It uses \*\*Decision Tree\*\* and \*\*Random Forest\*\* models for classification, and compares their performance.  



Dataset

\- Dataset collected from \[Kaggle](https://www.kaggle.com/)  

\- After downloading, necessary cleaning was performed:

&nbsp; - Removed NA/missing values  

&nbsp; - Dropped unnecessary columns 



Preprocessing

1\. Cleaned each column to handle NA and irrelevant values  

2\. Divided dataset into input (features) and output (target: `pass/fail`)  

3\. Split data into training (80%) and testing (20%)





Model Training

\- Trained using:

&nbsp; - Decision Tree Classifier  

&nbsp; - Random Forest Classifier 



Results

\- Both models were evaluated using accuracy  

\- \*\*Random Forest performed better than Decision Tree\*\* 



How to Run

1\. Clone this repo  

&nbsp;  ```bash

&nbsp;  git clone https://github.com/souravdey398892-lgtm/My\_Project/tree/master/My\_Projects/Student\_Data\_Analysis\_DT\_RM.git



2\. Install required libraries:  

```bash

pip install -r requirements.txt

