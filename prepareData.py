import math
import numpy as np
import ezPickle as p
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
	encoderDict = makeOneHotEncodeDict(allNotes)
	return encoderDict

songList = p.load('songList')
inputData = []
outputData = []

encoderDict = getDict(songList)

maxLen = 0
for song in songList:
    if len(song) > maxLen:
        maxLen = len(song)
print(maxLen)
print(len(songList))
		
inputData = []
outputData = []
for song in songList:
	inputData.append(convertW2V(song[0:len(song)-1],  maxLen))
	outputData.append(convertOneHot(song[1:], encoderDict,  maxLen))
otherLen = len(inputData[0][0])
otherOLen = len(outputData[0])
outputSize = len(outputData[0][0])
print(outputSize)
for item in inputData:
	if len(item) == 4006:
		for other in item:
			if len(other) != otherLen:
				print("IT DONE BROOPKE")
	else:
		print("OH BOI IT REAL BROKE")
for item in outputData:
	if len(item) != otherOLen:
		print("OH MAN EY BROKE EY DID BAD")
		print(len(item))
	else:
		for other in item:
			if len(other) != outputSize:
				print("IT BE BROKE IN HERE MAI DUDE")
print("done")
inputData = np.array(inputData).reshape(len(inputData), maxLen, 20)
inputShape = (len(inputData[0]), len(inputData[0][0]))
outputSize = len(outputData[0][0])

p.save(encoderDict,'oneHotDict')
p.save(inputShape, 'inputShape')
p.save(outputSize,'outputSize')
p.save(maxLen,'maxLen')


