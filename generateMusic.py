import pickle,  music21
import numpy as np
import random, os
import ezPickle as p
maxLen = p.load('maxLen')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"#"1,2,3"
import math, random, csv
window_size = p.load('window_size')
w2v = p.load("w2v")
def closestVec(vec):
    newVec = w2v.wv.most_similar( [vec], [], 1)
    return w2v[newVec[0][0]]
def randCloseVec(vec):
    newVecs = w2v.wv.most_similar( [vec], [], 10)
    
    newVec = []
    for i in range(10):
        newVec = newVecs[i][0]
        if random.random() < .5:
            
            break
    
    return w2v[newVec]
def convertW2V(items,  padLength):
    l = []
    for item in items:
        l.append(w2v[item])
    for i in range(padLength - len(items)):
        l.append(list(np.zeros(20)))
    return l
def padZeros(original_list, maxLen):
    l = original_list.copy()
    for i in range(maxLen - len(l)):
        l.append(np.zeros(20))
    return l
def getReverseOneHotDict(dict):
    inverted_dict = {str(value): key for key, value in dict.items()}
    return inverted_dict

print("started")
model = p.load("kerasTrained")
encoderDict = p.load("oneHotDict")
reverseEncoderDict = getReverseOneHotDict(encoderDict)

num_samples= p.load('numSamples')
targetLine = math.floor(random.random() * num_samples)

seed_line = 0
myfile = open('training_data.csv', 'r')
reader = csv.reader(myfile, delimiter=',')
row = []
for i in range(targetLine):
    row = next(reader)
    #print(i)
song = convertW2V(row[1:],window_size)
for i in range(100):
    tempSong = song[-100:]
    model.reset_states()
    outputHot = list(model.predict(np.array(tempSong).reshape(1,100,20)))
    #print(max(outputHot))
    song.append(w2v[reverseEncoderDict[str(outputHot)]])
p.save( song, 'song' )

