# Datathon-Project

# Logistic Regression Model to classify various kinds of diseases along with their symptoms in a given dataset

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("PROJECT/dataset.csv")
print("Disease Set")
data.head()
```
![op-1](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/2c477797-8ccf-4605-8b0d-a340786148fc)


```
print("\nChecking the null")
data.isnull().sum()
print("Data Duplicate")
data.duplicated().sum()
```

![op-2](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/bbac868b-08ba-44e9-82a9-a11805d2944e)

```
data1=data.copy()
data1
```

![op-3](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/00f49ed1-4a86-41ca-9858-3dc533cb80a2)

```
sev_df = pd.read_csv('PROJECT/Symptom-severity.csv')
sev_df
```
![image](https://github.com/AnnBlessy/Datathon-Project/assets/119477835/e19eb352-271d-417b-9cc8-d41d6e88805c)

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

```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(" y_prediction array")
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy value")
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion array")
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification report")
print(classification_report1)
```
