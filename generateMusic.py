import pickle,  music21
import numpy as np
import random
maxLen = 4006
def pickleSave(filename,  object):
    pickleOut = open('pickledObjects/' + filename,  'wb')
    pickle.dump(object,  pickleOut)
    pickleOut.close()

def pickleLoad(filename):
    pickleIn = open('pickledObjects/' + filename,  'rb')
    temp = pickle.load(pickleIn)
    pickleIn.close()
    return temp
w2v = pickleLoad("w2v.pickle")
def closestVec(vec):
    newVec = w2v.wv.most_similar( [vec], [], 1)
    print(newVec)
    
    return w2v[newVec[0][0]]
def randCloseVec(vec):
    newVecs = w2v.wv.most_similar( [vec], [], 10)
    #print(newVecs)
    newVec = []
    for i in range(10):
        newVec = newVecs[i][0]
        if random.random() < .5:
            #print(i)
            break
    print(newVec)
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
model = pickleLoad("kerasTrained.pickle")
encoderDict = pickleLoad("oneHotDict.pickle")
reverseEncoderDict = getReverseOneHotDict(encoderDict)
inputData = pickleLoad('inputData.pickle')
testInput = inputData[0][0:50]
print(testInput)
song = list(testInput)
#for i in range(100):
#    seed = 2* np.random.random(20) - 1
#    song.append(closestVec(seed))

for i in range(1000):
    tempSong = np.array(padZeros(song,  maxLen)).reshape(1, maxLen, 20)
    output = model.predict(tempSong)
    model.reset_states()
    #for j in range(i+2):
    #    print(j)
    #    print(closestVec(output[0][j]))
    maxItem = 0
    maxVal = 0
    count = 0
    for item in output[0][i+1]:
        if item > maxVal:
            maxItem = count
            maxVal = item
        count+=1
    print(maxItem)
    cleanedOutput = [0]*len(output[0][i])
    cleanedOutput[maxItem] = 1
    newItem = w2v[reverseEncoderDict[str(cleanedOutput)]]
    song.append(newItem)
    #print(newItem)
    #print(closestVec(output[0][i]))
closestSong = []

pickleSave('song.pickle', song )

