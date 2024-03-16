# -*- coding: utf-8 -*-
"""EXP -1 (DEEP LEARNING)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ANn0zQgjB3PgP9Bmt68n9KCNovCuAkt3
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential as seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('dataset').sheet1
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})

x = df[["Input"]] .values
y = df[["Output"]].values

scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)

ai_brain = seq([
    Den(9,activation = 'relu',input_shape=[1]),
    Den(16,activation = 'relu'),
    Den(1),
])

ai_brain.compile(optimizer = 'rmsprop',loss = 'mse')

ai_brain.fit(x_train,y_train,epochs=1000)
ai_brain.fit(x_train,y_train,epochs=1000)

loss_plot = pd.DataFrame(ai_brain.history.history)
loss_plot.plot()

x_n1 = [[9]]
x_n_n = scaler.transform(x_n1)
ai_brain.predict(x_n_n)

err = rmse()
preds = ai_brain.predict(x_test)
err(y_test,preds)