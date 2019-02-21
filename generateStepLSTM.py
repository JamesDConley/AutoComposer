from keras.models import Sequential
from keras.layers import LSTM,  Dense,  Dropout,  Masking
from keras.activations import softmax
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
import keras
import numpy as np
import ezPickle as p
import math

keras.callbacks.TensorBoard(histogram_freq=0)

w2v = p.load('w2v')
def convertW2V(items,  padLength):
    l = []
    for item in items:
        l.append(w2v[item])
    for i in range(padLength - len(items)):
        l.append(list(np.zeros(20)))
    return l
def makeOneHotEncodeDict(corpus):
    dict = {}
    count = 0
    for item in corpus:
        if not str(item) in dict.keys():
            dict[str(item)] = count
            count+=1
    for key in dict.keys():
        dict[key] = [0]*dict[key] + [1] + ([0]*(count - dict[key]))
    return dict

def dataGenerator(n):
    for i in range(math.floor(n/10)):
        yield (np.array(p.load('inputData'+str(i))),np.array(p.load('outputData'+str(i)))) 

inp, out = [item for item in dataGenerator(24)][0]
print(inp.shape)
print(out.shape)
import sys
print(sys.getsizeof(out))
inputData = p.load('inputData1')
outputData = p.load('outputData1')

batch_size = len(inputData)
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(len(inputData[0]), len(inputData[0][0])) ))
model.add(LSTM(256,  return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(.2))
#model.add(LSTM(16,  return_sequences=True))
#model.add(Dropout(.1))
model.add(Dense(len(outputData[0][0]),  activation='softmax'))
rms = RMSprop()
#model = multi_gpu_model(model, gpus=1)
model.compile(loss='categorical_crossentropy', optimizer=rms,  metrics=['categorical_accuracy'])
print("Fitting")
#model.fit(inputData, outputData, epochs=250,  verbose=1,  shuffle=False)
#model.fit_generator(dataGenerator(150),  epochs=10,  verbose = 1,  steps_per_epoch=10)
#p.save('kerasTrained', model)
print("Saved")

