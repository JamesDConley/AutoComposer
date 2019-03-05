import math
import numpy as np
import ezPickle as p
w2v = p.load('w2v')
size = p.load('size')
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
def makeOneHotEncodeDict(corpus):
    dict = {}
    count = 0
    for item in corpus:
        if not str(item) in dict.keys():
            dict[str(item)] = count
            count+=1
    for key in dict.keys():
        my_list = [0]*count
        my_list[dict[key]] = 1
        dict[key] = my_list
    return dict

def getDict(songList):
	allNotes = [item for song in songList for item in song]
	encoderDict = makeOneHotEncodeDict(allNotes)
	return encoderDict
print("loading Songs")
songList = p.load('songList')
inputData = []
outputData = []
print("Buidling Dictionary")
encoderDict = getDict(songList)
print("Finding Maximum SongLength")
maxLen = 0
for song in songList:
    if len(song) > maxLen:
        maxLen = len(song)
print(maxLen)
print(len(songList))
		
inputData = []
outputData = []
window_size = 100
print("Writing Data")
import csv
data_file = open('training_data.csv',mode='w')
file_writer = csv.writer(data_file, delimiter=',')
num_samples = 0
for song in [item for item in songList if len(item) > window_size]:
	for i in range(0,len(song)-window_size):
		file_writer.writerow([song[i+window_size]] + song[i:i+window_size])
		num_samples+=1

print("done")
inputData = np.array(inputData).reshape(len(inputData), maxLen, 20)
inputShape = (window_size, size)
outputSize = len(encoderDict.keys())
print("Saving")
p.save(encoderDict,'oneHotDict')
p.save(inputShape, 'inputShape')
p.save(outputSize,'outputSize')
p.save(num_samples,'numSamples')
p.save(window_size,'window_size')


