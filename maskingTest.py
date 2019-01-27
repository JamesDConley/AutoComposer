from keras.models import Sequential
from keras.layers import LSTM,  Dense,  Dropout,  Masking

from keras.optimizers import RMSprop
import numpy as np

inputData = [[[1]]*10 + [[0]]*90] * 10
print(inputData)
outputData = [[[.01]]*10 + [[1]]*90]*10

t1 =  [[[1]]*10 ] * 10
o1 = [[[.01]]*10 ]*10

t2 = [[[0]]*100]*10
inputData = t1
outputData = o1

inputData = np.array(inputData)
outputData = np.array(outputData)

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(len(inputData[0]), len(inputData[0][0])) ))
model.add(LSTM(1, return_sequences=True))

model.add(Dense(len(outputData[0][0]),  activation='sigmoid'))
rms = RMSprop()
model.compile(loss='mean_squared_error', optimizer=rms,  metrics=['accuracy'])
print("Fitting")
model.fit(inputData, outputData, epochs=100,  verbose=1,  shuffle=False)
print(model.predict(inputData))
