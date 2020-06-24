# Customer_Churn_Prediction

## Table of contents
* [General Info](#general-info)
* [Technologies](#technologies)
* [Data Exploration](#data_exploration)

## General Info
I use supervised learning models to identify customers who are likely to churn in the future. Furthermore, I will analyze top factors that influence user retention
	
## Technologies
Project is created with:
* Jupyter Notebook: 6.0.3
* Python 3
	
## Data Exploration
- understand the raw dataset

```
print ("Num of rows: " + str(churn_df.shape[0])) # row count
print ("Num of columns: " + str(churn_df.shape[1])) # col count

# check data info
churn_df.info()

# check the unique values for each column
churn_df.nunique()
```
About 20% of customers are labeled churn, which is imbalanced.

- understand the features

I check the numerical and categorical feature distribution as well as correlation of features.

![categorical feature distribution](https://github.com/Yulin-lyl/Customer_Churn_Prediction/blob/master/categorical%20feature%20distribution.png)


