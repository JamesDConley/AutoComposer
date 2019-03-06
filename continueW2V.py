from keras.models import Sequential
from keras.layers import LSTM,  Dense,  Dropout,  Masking
from keras.utils import multi_gpu_model
from keras.optimizers import RMSprop
from keras.activations import softmax
import tensorflow as tf
import keras, math
import numpy as np
import pickle
import os, sys, csv
import ezPickle as p
import time
window_size = p.load('window_size')
os.environ["CUDA_VISIBLE_DEVICES"]="2"
keras.callbacks.TensorBoard(histogram_freq=0)

w2v = p.load('w2v')
def convertW2V(items,  padLength):
    l = []
    for item in items:
        l.append(w2v[item])
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

encoderDict = p.load("oneHotDict")
batch_size = 300
output_size = p.load('outputSize')
def songBatchGenerator(batch_size):
	num_samples= p.load('numSamples')
	start = 0
	current_line = 0
	myfile = open('training_data.csv')
	reader = csv.reader(myfile, delimiter=',')
	while True:
		start = 0		
		end = start+batch_size
		i_d = []
		o_d = []
		
		myfile.seek(0)
		current_line = 0 
		for row in reader:
			if current_line >= end:
				#print(np.array(o_d)[0])
				yield (np.array(i_d), np.array(o_d))
				start = end
				end = start+batch_size
				i_d = []
				o_d = []
				if end > num_samples:
					end = num_samples-1
					
			i_d.append(convertW2V(row[1:],window_size))
			o_d.append(convertW2V([row[0]],1)[0])
			current_line+=1

p_model = p.load('vecTrained')
rms = RMSprop()

#p_model.compile(loss='categorical_crossentropy',optimizer=rms, metrics=['categorical_accuracy'])

print("Fitting")
num_samples = p.load('numSamples')
p_model.fit_generator(songBatchGenerator(batch_size), epochs=200,  verbose=1,  shuffle=False, steps_per_epoch=math.ceil(num_samples/batch_size))
p.save( p_model,'vecTrained')
print("Saved")

