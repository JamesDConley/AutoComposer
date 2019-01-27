from keras.models import Sequential
from keras.layers import LSTM,  Dense,  Dropout,  Masking
from keras.activations import softmax
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


inputData = pickleLoad('inputData.pickle')
outputData = pickleLoad('outputData.pickle')
print(len(inputData[0]))
print(len(inputData[0][0]))
print(len(outputData[0][0]))

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
model.compile(loss='categorical_crossentropy', optimizer=rms,  metrics=['categorical_accuracy'])
print("Fitting")
model.fit(inputData, outputData, epochs=250,  verbose=1,  shuffle=False)
pickleSave('kerasTrained.pickle', model)
print("Saved")

