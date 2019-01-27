import pickle
import numpy as np
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
    
songList = pickleLoad('songList.pickle')
inputData = []
outputData = []
maxLen = 0

allNotes = [item for song in songList for item in song]
#print(allNotes)
encoderDict = makeOneHotEncodeDict(allNotes)
pickleSave('oneHotDict.pickle', encoderDict)
for song in songList:
    if len(song) > maxLen:
        maxLen = len(song)
print(maxLen)
for song in songList:
    inputData.append(convertW2V(song[0:len(song)-1],  maxLen))
    outputData.append(convertOneHot(song[1:], encoderDict,  maxLen))
inputData = np.array(inputData).reshape(len(inputData), maxLen, 20)

print(inputData.shape)
print(len(outputData))
outputData = np.array(outputData).reshape(len(outputData),  len(outputData[0]),  len(outputData[0][0]))
print(outputData.shape)
pickleSave('inputData.pickle', inputData)
pickleSave('outputData.pickle', outputData)
