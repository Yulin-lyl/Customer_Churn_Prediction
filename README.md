# Customer_Churn_Prediction

## Table of contents
* [General Info](#general-info)
* [Technologies](#Technologies)
* [Data Exploration](#data-exploration)
* [Feature Preprocessing](#feature-preprocessing)
* [Model Training and Results Evaluation](#model-training-and-results-evaluation)
* [Feature Importance](#feature-importance)

## General Info
I use supervised learning models to identify customers who are likely to churn in the future. Furthermore, I will analyze top factors that influence user retention
	
## Technologies
Project is created with:
* Jupyter Notebook: 6.0.3
* Python 3
	
## Data Exploration
### understand the raw dataset

```
print ("Num of rows: " + str(churn_df.shape[0])) # row count
print ("Num of columns: " + str(churn_df.shape[1])) # col count

# check data info
churn_df.info()

# check the unique values for each column
churn_df.nunique()
```
About 20% of customers are labeled churn, which is imbalanced.

### understand the features
I check the numerical and categorical feature distribution as well as correlation of features.

numerical feature distribution:

<img src="https://github.com/Yulin-lyl/Customer_Churn_Prediction/blob/master/numerical%20feature%20distribution.png" alt="numerical feature distribution" width="800" height="600">

categorical feature distribution:

<img src="https://github.com/Yulin-lyl/Customer_Churn_Prediction/blob/master/categorical%20feature%20distribution.png" alt="categorical feature distribution" width="800" height="600">

correlation of features:

<img src="https://github.com/Yulin-lyl/Customer_Churn_Prediction/blob/master/feature%20correlation.png" alt="correlation of features" width="550" height="500">

## Feature Preprocessing
### Split Dataset
```
# Reserve 20% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, stratify = y, random_state=1)
```
## Model Training and Results Evaluation
### Model Training and Selection
I use logistic regression, KNN and Random Forest to build machine learning models and train them.
And I use 5-fold Cross Validation to get the accuracy for different models.
Random Forest has the highest accuracy, which is 0.8628.

```
model_names = ['Logistic Regression','KNN','Random Forest']
model_list = [classifier_logistic, classifier_KNN, classifier_RF]
count = 0

for classifier in model_list:
    cv_score = model_selection.cross_val_score(classifier, X_train, y_train, cv=5)
    print(cv_score)
    print('Model accuracy of ' + model_names[count] + ' is ' + str(cv_score.mean()))
    count += 1
```

### Find Optimal Hyperparameters
I use grid search to find optimal hyperparameters for each models.
```
# Possible hyperparamter options for Random Forest
# Choose the number of trees
parameters = {
    'n_estimators' : [40,60,80]
}
Grid_RF = GridSearchCV(RandomForestClassifier(),parameters, cv=5)
Grid_RF.fit(X_train, y_train)
```
### Model Evaluation
TP: correctly labeled real churn

Precision(PPV, positive predictive value): tp / (tp + fp); Total number of true predictive churn divided by the total number of predictive churn; High Precision means low fp, not many return users were predicted as churn users.

Recall(sensitivity, hit rate, true positive rate): tp / (tp + fn) Predict most postive or churn user correctly. High recall means low fn, not many churn users were predicted as return users.

The best model is Random Forest.
Accuracy is: 0.86
precision is: 0.78
recall is: 0.44
Confusion Matrix:

<img src="https://github.com/Yulin-lyl/Customer_Churn_Prediction/blob/master/RF%20confusion%20matrix.png" alt="confusion matrix" width="550" height="500">

ROC Curve:

<img src="https://github.com/Yulin-lyl/Customer_Churn_Prediction/blob/master/RF%20ROC.png" alt="random forest ROC" width="550" height="500">

## Feature Importance
Feature importance ranking by Random Forest Model:
Age : 0.2404
EstimatedSalary : 0.1466
CreditScore : 0.1433
Balance : 0.1424
NumOfProducts : 0.1296
Tenure : 0.0816
IsActiveMember : 0.0395
Geography_Germany : 0.0217
Gender : 0.0185
HasCrCard : 0.0185
Geography_France : 0.0095
Geography_Spain : 0.0085

