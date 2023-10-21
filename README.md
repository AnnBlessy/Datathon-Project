# Datathon-Project

# Logistic Regression Model to classify various kinds of diseases along with their symptoms in a given dataset

## Importing necessary libraries 
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```
## Reading the dataset
```
data = pd.read_csv("PROJECT/dataset.csv")
print("Disease Set")
data.head()
```
![op-1](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/2c477797-8ccf-4605-8b0d-a340786148fc)

## Checking of nulls and duplicates
```
print("\nChecking the null")
data.isnull().sum()
print("Data Duplicate")
data.duplicated().sum()
```

![op-2](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/bbac868b-08ba-44e9-82a9-a11805d2944e)

## Creating a copy of the dataset
```
data1=data.copy()
data1
```

![op-3](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/00f49ed1-4a86-41ca-9858-3dc533cb80a2)

## Reading dataset-2
```
sev_df = pd.read_csv('PROJECT/Symptom-severity.csv')
sev_df
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/e19eb352-271d-417b-9cc8-d41d6e88805c)

## Determining X and Y values
```
x = data1.iloc[:,:-1]
print("Data-status")
print(x)

y = data1["Disease"]
print("data-status")
y
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/84af8d1e-ccb5-44bc-a5e0-49d6aa001fc2)
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/43e7dd96-3081-4e67-9c9f-0323eec34aa0)
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/5f4da218-d102-415f-bc3c-279c969097a1)
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/8efc4e6e-a784-4cd3-8dde-f2fb73224cb9)

## Conversion of categorical variables into numerical format
```
le = LabelEncoder()
data1["Disease"] = le.fit_transform(data1["Disease"])
data1["Symptom_1"] = le.fit_transform(data1["Symptom_1"])
data1["Symptom_2"] = le.fit_transform(data1["Symptom_2"])
data1["Symptom_3"] = le.fit_transform(data1["Symptom_3"])
data1["Symptom_4"] = le.fit_transform(data1["Symptom_4"])
data1["Symptom_5"] = le.fit_transform(data1["Symptom_5"])
data1["Symptom_6"] = le.fit_transform(data1["Symptom_6"])
data1["Symptom_7"] = le.fit_transform(data1["Symptom_7"])
data1["Symptom_8"] = le.fit_transform(data1["Symptom_8"])
data1["Symptom_9"] = le.fit_transform(data1["Symptom_9"])
data1["Symptom_10"] = le.fit_transform(data1["Symptom_10"])
data1["Symptom_11"] = le.fit_transform(data1["Symptom_11"])
data1["Symptom_12"] = le.fit_transform(data1["Symptom_12"])
data1["Symptom_13"] = le.fit_transform(data1["Symptom_13"])
data1["Symptom_14"] = le.fit_transform(data1["Symptom_14"])
data1["Symptom_15"] = le.fit_transform(data1["Symptom_15"])
data1["Symptom_16"] = le.fit_transform(data1["Symptom_16"])
data1["Symptom_17"] = le.fit_transform(data1["Symptom_17"])
print("Data")
data1
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/a07e40a4-0d84-4256-a659-e5e0dda036de)

```
x = data1.iloc[:,:-1]
print("Data-status")
x

y = data1["Disease"]
print("data-status")
y
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/8f744c36-c50d-4e36-8341-ef2b6d8426be)

## Training & Testing of data using Logistic Regression
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(" y_prediction array")
y_pred
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/795e44bf-13bc-4cae-8729-446c77618697)

## Calculating Accuracy
```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy value")
accuracy
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/a7e0eb51-84b2-4052-a350-df9aa5834478)

## Determining Confusion matrix
```
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion array")
confusion
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/fc393c0c-8d94-44ee-888c-bc1df8c145b4)

## Generating Classification report
```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification report")
print(classification_report1)
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/8aeb0fdb-32d7-4ed1-82e6-ed8ba619ca82)
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/2a86b558-f07a-4eab-9173-c23691d6c577)

# Linear Regression Model for Patient health camp record analysis 

## Importing necessary libraries 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
```

## Reading the dataset
```
data = pd.read_csv("Patient_Profile.csv")
print("PATIENT PROFILE")
data
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/a321e4d3-4858-4b2f-b773-37a58f205ed4)

## Conversion of data types
```
data['Patient_ID'] = data['Patient_ID'].astype('int64')
data['First_Interaction'] = pd.to_datetime(data['First_Interaction'])

data['date'] = data['First_Interaction'].apply(lambda x: x.date())
data
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/16b81619-3a37-491b-9565-6e6191289962)

## Determining X and Y values
```
x=data.iloc[:,:-1].values
print("X =",x)
y=data.iloc[:,1].values
print("Y =",y)
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/4cad2f6c-e0bd-42a1-a6eb-afd17acee14a)

## Reading Train.csv dataset
```
data1 = pd.read_csv("PROJECT/Train.csv")
data1
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/f6459a0a-acc3-446c-b5f2-41f5c9107c2c)

## Training & Testing using Linear Regression
```
a = data1['Var1'].values.reshape(-1, 1)
b = data1['Patient_ID']
X = data1['Var2'].values.reshape(-1,1)
y = data1['Patient_ID']
u = data1['Var3'].values.reshape(-1,1)
v = data1['Patient_ID']
c = data1['Var4'].values.reshape(-1,1)
d = data1['Patient_ID']

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=1/3, random_state=0)
model = LinearRegression()
model.fit(a_train, b_train)
b_pred = model.predict(a_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

u_train, u_test, v_train, v_test = train_test_split(u, v, test_size=1/3, random_state=0)
model = LinearRegression()
model.fit(u_train, v_train)
v_pred = model.predict(u_test)

c_train, c_test, d_train, d_test = train_test_split(u, v, test_size=1/3, random_state=0)
model = LinearRegression()
model.fit(c_train, d_train)
d_pred = model.predict(c_test)
```
## Plotting Training set graph
```
plt.scatter(X_train, y_train, color='purple', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.scatter(a_train, b_train, color='orange', label='Actual')
plt.plot(a_test, b_pred, color='blue', linewidth=3, label='Predicted')
plt.scatter(u_train, v_train, color='pink', label='Actual')
plt.plot(u_test, v_pred, color='blue', linewidth=3, label='Predicted')
plt.scatter(c_train, d_train, color='brown', label='Actual')
plt.plot(c_test, d_pred, color='blue', linewidth=3, label='Predicted')
#Anonymous Variables
plt.xlabel('Var1 or Var2 or Var3 or Var4')
plt.ylabel('Patient_ID')
plt.title('TRAINING SET GRAPH')
plt.show()
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/b6ecdef6-c1be-44d6-994b-a279545c60f4)

## Plotting Testing set graph
```
plt.scatter(X_test, y_test, color='yellow', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.scatter(a_test, b_test, color='red', label='Actual')
plt.plot(a_test, b_pred, color='blue', linewidth=3, label='Predicted')
plt.scatter(u_test, v_test, color='pink', label='Actual')
plt.plot(u_test, v_pred, color='blue', linewidth=3, label='Predicted')
plt.scatter(c_test, d_test, color='green', label='Actual')
plt.plot(c_test, d_pred, color='blue', linewidth=3, label='Predicted')
plt.xlabel('Var1 or Var2 or Var3 or Var4')
plt.ylabel('Patient_ID')
plt.title('TESTING SET GRAPH')
plt.show()
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/07e67c80-c2b4-4e86-a0b3-e050166d5cac)

# Conclusion:
Thus, There are 2 different features developed with bascic machine learning algorithms to classify, analyse and predict the data. This project focuses on healthcare and remote health system for monitoring and helping the users to self-diagonse upto certain accuracy level.
