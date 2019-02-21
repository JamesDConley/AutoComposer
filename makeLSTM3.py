import ezPickle as p
from keras.models import Sequential
from keras.layers import LSTM,  Dense,  Dropout,  Masking
from keras.utils import multi_gpu_model
from keras.optimizers import RMSprop
from keras.activations import softmax
import tensorflow as tf
import numpy as np
import math, keras
import os
import time
keras.callbacks.TensorBoard(histogram_freq=0)
w2v =p.load('w2v')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#"1,2,3"
def convertW2V(items,  padLength):
    l = []
    for item in items:
        l.append(w2v[item])
        #print(w2v[item])
    for i in range(padLength - len(items)):
        l.append(list(np.zeros(20)))
    return l
def convertOneHot(items,  mydict,  padLength):
    l = []
    for item in items:
        l.append(mydict[str(item)])
    for i in range(padLength - len(items)):
        l.append(np.zeros(len(l[0])))
    return l
encoderDict = p.load('oneHotDict')
def songBatchGenerator(songList,batchSize, stop):
	start = 0
	count = 0
	maxLen = p.load('maxLen')
	lastTime = 0
	totalTime = 0
	thisTime = 0
	while True:
		currentTime = time.time()
		if lastTime != 0:
			thisTime = currentTime - lastTime			
			totalTime += thisTime
		lastTime = currentTime
			
		end = start+batchSize
		i_d = []
		o_d = []
		if end > len(songList)-1:
			end = len(songList)-1
		tempList = songList[start:end]
		for song in tempList:
		    i_d.append(convertW2V(song[0:len(song)-1],  maxLen))
		    o_d.append(convertOneHot(song[1:], encoderDict,  maxLen))
		
		yield np.array(i_d),np.array(o_d)
		
		print("********************************************************************************")
		print(count," ", start, " ", end)
		print(count/stop * 100,"% Done")
		if count > 0:
			print("Estimated Time Remaining: ", totalTime/count*(stop-count)/60,"minutes")
			print("Average Time: ", totalTime/count)
			print("Remaining Iterations: ", (stop-count))
		print("********************************************************************************")
		start = end		
		if start >= len(songList)-1:
			start = 0		
			
		count+=1
		if count==stop:
			break
		



def create_model():
	inputShape = p.load('inputShape')
	outputSize = p.load('outputSize')
	model = Sequential()
	model.add(Masking(mask_value=0., input_shape=inputShape))
	model.add(LSTM(256,  return_sequences=True))
	
	model.add(LSTM(128, return_sequences=True))
	
	model.add(Dense(outputSize,  activation='softmax'))
	return model

#with tf.device("/cpu:0"):
model = create_model()

# make the model parallel
#p_model = multi_gpu_model(model, gpus=3)
rms = RMSprop()
try:
	model.compile(loss='categorical_crossentropy',optimizer=rms, metrics=['categorical_accuracy'])

	print("Fitting")
	batch_size = 12
	epochs = 100
	songList = p.load('songList')
	#model.fit_generator(songBatchGenerator(songList,batch_size), epochs=10,  verbose=1,  shuffle=False, steps_per_epoch=math.ceil(len(songList)/batch_size),max_queue_size=2)
	for inp, out in songBatchGenerator(songList,batch_size,epochs*math.ceil(len(songList)/batch_size)):		
		model.train_on_batch(inp,out)
	p.save(model, 'kerasTrainedNoDropout')
	print("Saved")
except MemoryError:
	print("Memory whyyyyyy")

