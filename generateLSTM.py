from keras.models import Sequential
from keras.layers import LSTM,  Dense,  Dropout,  Masking
from keras.optimizers import RMSprop
import numpy as np
import pickle 
def pickleSave(filename,  object):
    pickleOut = open('pickledObjects/' + filename,  'wb')
    pickle.dump(object,  pickleOut)
    pickleOut.close()

def pickleLoad(filename):
    pickleIn = open('pickledObjects/' + filename,  'rb')
    temp = pickle.load(pickleIn)
    pickleIn.close()
    return temp
w2v = pickleLoad('w2v.pickle')
def convertW2V(items,  padLength):
    l = []
    for item in items:
        l.append(w2v[item])
    for i in range(padLength - len(items)):
        l.append(list(np.zeros(20)))
    return l
def oneHotEncodeOutput(original_list):
    dict={}
    for item in original_list:
        



songList = pickleLoad('songList.pickle')
print("here")
inputData = []
outputData = []
maxLen = 0
"""
for song in songList[0:3]:
    if len(song) > maxLen:
        maxLen = len(song)
    inputData.append([])
    outputData.append([])
    for i in range(len(song)-1):
        inputData[-1].append(list(map(convertW2V, list(song[0:i+1]))))
        outputData[-1].append(w2v[song[i+1]])
"""
for song in songList:
    if len(song) > maxLen:
        maxLen = len(song)
for song in songList:
    inputData.append(convertW2V(song[0:len(song)-1],  maxLen))
    outputData.append(convertW2V(  song[1:],  maxLen))


print("Creating Model")

inputData = np.array(inputData).reshape(len(inputData), maxLen, 20)
print(np.array(inputData).shape)
features = 20
timesteps = maxLen
batch_size = len(inputData)
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(16,  return_sequences=True))
model.add(Dropout(.1))
model.add(Dense(20,  activation='tanh'))
rms = RMSprop()
model.compile(loss='mean_squared_error', optimizer=rms,  metrics=['accuracy'])
print("Fitting")
model.fit(np.array(inputData), np.array(outputData), epochs=10,  verbose=1)
pickleSave('kerasTrained.pickle', model)
print("Saved")

