import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1,  input_shape=(1,)))
model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError(),metrics=['mse'])

houseSalesDataset = pd.read_csv("https://raw.githubusercontent.com/assiri/simpleAnnRegrission/main/house_data.csv")

pointsDataset = houseSalesDataset[["sqft_living","price"]]
X_train, X_test, y_train, y_test = train_test_split(pointsDataset["sqft_living"], pointsDataset["price"], test_size=0.33, random_state=42)

scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train.values.reshape(-1, 1))
y_train=scaler.transform(y_train.values.reshape(-1, 1))

results=model.fit(X_train,y_train, epochs= 50,batch_size=32,validation_split=0.2)

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

predictions = model.predict(X_test)

plt.scatter(y_test,predictions)

# Perfect predictions
plt.plot(y_test,y_test,'r')
plt.show()


trainingLoss = results.history["loss"][-1]  # print(result.history.keys())
print(f"Training set loss: {trainingLoss}")


validationLoss =results.history["val_loss"][-1]
print(f"Validation set loss: {validationLoss}")

results= model.evaluate(X_test, y_test)
print("test loss, test acc:", results)


res=model.predict(scaler.transform(np.array([[2000]])))
l=scaler.inverse_transform(res)
print(l) #528000
