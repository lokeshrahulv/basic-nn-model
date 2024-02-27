# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network regression model is a type of artificial neural network designed for predicting continuous numerical values. It consists of an input layer, hidden layers with activation functions, and an output layer without activation for regression tasks.

Training involves minimizing a loss function, typically mean squared error, using optimization algorithms like stochastic gradient descent. The model learns patterns in input data to make accurate predictions. 

Hyperparameters like the number of layers and neurons influence the model's complexity. Regularization techniques, such as dropout, can prevent overfitting. Neural network regression is widely used in diverse fields, including finance, science, and engineering.

## Neural Network Model
![Screenshot 2024-02-27 223843](https://github.com/lokeshrahulv/basic-nn-model/assets/118423842/3f6bdeea-c8ce-44cf-ab2c-3b7964682e34)
## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Program developed by: LOKESH RAHUL V V 
Register Number: 212222100024
```
## Importing Modules:
```python
from google.colab import auth
import gspread
from google.auth import default

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den

from tensorflow.keras.metrics import RootMeanSquaredError as rmse

import pandas as pd
import matplotlib.pyplot as plt
```
## Authenticate & Create Dataframe using Data in Sheets:
```python
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('SomDocs DL-01').sheet1 
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
```
## Assign X and Y values:
```python
x = df[["Input"]] .values
y = df[["Output"]].values
```
## Normalize the values & Split the data:
```python
scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)
```
## Create a Neural Network & Train it:
```python
ai_brain = Seq([
    Den(9,activation = 'relu',input_shape=[1]),
    Den(16,activation = 'relu'),
    Den(1),
])

ai_brain.compile(optimizer = 'rmsprop',loss = 'mse')

ai_brain.fit(x_train,y_train,epochs=1000)
ai_brain.fit(x_train,y_train,epochs=1000)
```
## Plot the Loss:
```python
loss_plot = pd.DataFrame(ai_brain.history.history)
loss_plot.plot()
```
## Evaluate the model:
```python
err = rmse()
preds = ai_brain.predict(x_test)
err(y_test,preds)
```
## Predict for some value:
```python
x_n1 = [[9]]
x_n_n = scaler.transform(x_n1)
ai_brain.predict(x_n_n)
```
![Screenshot 2024-02-27 231403](https://github.com/lokeshrahulv/basic-nn-model/assets/118423842/c6bd25c4-7069-4047-9449-3e4c64b8f1e7)
## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot 2024-02-27 233336](https://github.com/lokeshrahulv/basic-nn-model/assets/118423842/07407d74-91af-4630-b23d-6b91ce7127fb)
### Test Data Root Mean Squared Error
![Screenshot 2024-02-27 234131](https://github.com/lokeshrahulv/basic-nn-model/assets/118423842/eea78319-974e-4e51-8c2a-76ce7390e1bb)
### New Sample Data Prediction
![image](https://github.com/lokeshrahulv/basic-nn-model/assets/118423842/abc3795f-d1e7-434f-a1d9-00636d7f4843)
## RESULT
Thus to develop a neural network regression model for the dataset created is successfully executed.
