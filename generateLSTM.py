from keras.models import Sequential
from keras.layers import LSTM,  Dense,  Dropout,  Masking
from keras.utils import multi_gpu_model
from keras.optimizers import RMSprop
from keras.activations import softmax
import tensorflow as tf
import keras
import math
import numpy as np
import pickle 
import os, sys
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
keras.callbacks.TensorBoard(histogram_freq=0)
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
        #print(w2v[item])
    for i in range(padLength - len(items)):
        l.append(list(np.zeros(20)))
    return l
def convertOneHot(items,  dict,  padLength):
    l = []
    for item in items:
        l.append(dict[str(item)])
    for i in range(padLength - len(items)):
        l.append(np.zeros(len(l[0])))
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

def getDict(songList):
	allNotes = [item for song in songList for item in song]
	print(sys.getsizeof(allNotes))
	encoderDict = makeOneHotEncodeDict(allNotes)
	pickleSave('oneHotDict.pickle', encoderDict)
	del allNotes
	return encoderDict
songList = pickleLoad('songList.pickle')[0:90]
inputData = []
outputData = []
maxLen = 0

encoderDict = getDict(songList)

for song in songList:
    if len(song) > maxLen:
        maxLen = len(song)
print(maxLen)
print(len(songList))
@profile
def songBatchGenerator(songList,batchSize):
	start = 0
	count = 0
	while True:
		end = start+batchSize
		i_d = []
		o_d = []
		if end > len(songList)-1:
			end = len(songList)-1
		tempList = songList[start:end]
		for song in tempList:
		    i_d.append(convertW2V(song[0:len(song)-1],  maxLen))
		    o_d.append(convertOneHot(song[1:], encoderDict,  maxLen))
		print("*************************")
		print(count," ", start, " ", end)
		print("*************************")
		yield np.array(i_d),np.array(o_d)
		start = end		
		if start >= len(songList)-1:
			start = 0		
			
		count+=1
		
		

inputData = []
outputData = []
for song in songList[0:3]:
	inputData.append(convertW2V(song[0:len(song)-1],  maxLen))
	outputData.append(convertOneHot(song[1:], encoderDict,  maxLen))
inputData = np.array(inputData).reshape(len(inputData), maxLen, 20)
batch_size = 8

def create_model():
	model = Sequential()
	model.add(Masking(mask_value=0., input_shape=(len(inputData[0]), len(inputData[0][0])) ))
	model.add(LSTM(256,  return_sequences=True))
	model.add(Dropout(.2))
	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(.2))
	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(.2))
	model.add(Dense(len(outputData[0][0]),  activation='softmax'))
	return model
#print(len(inputData[0]))
#print(len(inputData[0][0]))
#print(len(outputData[0][0]))
with tf.device("/cpu:0"):
     model = create_model()

# make the model parallel
p_model = multi_gpu_model(model, gpus=4)
rms = RMSprop()
#p_model = multi_gpu_model(model, gpus=4)

p_model.compile(loss='categorical_crossentropy',optimizer=rms, metrics=['categorical_accuracy'])

print("Fitting")

p_model.fit_generator(songBatchGenerator(songList,batch_size), epochs=100,  verbose=1,  shuffle=False, steps_per_epoch=math.ceil(len(songList)/batch_size))
pickleSave('kerasTrained.pickle', p_model)
print("Saved")

