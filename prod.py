import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1,  input_shape=(1,)))
model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())

houseSalesDataset = pd.read_csv("https://raw.githubusercontent.com/assiri/simpleAnnRegrission/main/house_data.csv")

pointsDataset = houseSalesDataset[["sqft_living","price"]]
X_train, X_test, y_train, y_test = train_test_split(pointsDataset["sqft_living"], pointsDataset["price"], test_size=0.33, random_state=42)

scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train.values.reshape(-1, 1))
y_train=scaler.transform(y_train.values.reshape(-1, 1))

model.fit(X_train,y_train, epochs= 50)

res=model.predict(scaler.transform(np.array([[80000]])))
l=scaler.inverse_transform(res)
print(l) #9698469