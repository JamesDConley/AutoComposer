import pickle,  music21
import numpy as np
import random, os
maxLen = 4006
os.environ["CUDA_VISIBLE_DEVICES"] = "1"#"1,2,3"
import math
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
testInput = inputData[math.floor(random.uniform(0,1)*100)][0:50]
print(testInput)
song = list(testInput)
#for i in range(100):
#    seeccccccd = 2* np.random.random(20) - 1
#    song.append(closestVec(seed))

for i in range(1):
    tempSong = np.array(padZeros(song,  maxLen)).reshape(1, maxLen, 20)
    outputSequence = model.predict(tempSong)
    #model.reset_states()
    #for j in range(i+2):
    #    print(j)
    #    print(closestVec(output[0][j]))
    
    print(len(outputSequence))
    for output in outputSequence[0]:
        maxItem = 0
        maxVal = 0
        count = 0
        cumulative = 0
        randVal = random.uniform(0,1)
        for item in output:
            cumulative+=item
            if len(output) < 163:
                break
            if cumulative > randVal:
                maxItem = count
                maxVal = item
                print(maxItem, " with value ", maxVal)
                break
            count+=1
        #print(maxItem)
        cleanedOutput = [0]*len(output)
        cleanedOutput[maxItem] = 1
        if str(cleanedOutput) in reverseEncoderDict:
            newItem = w2v[reverseEncoderDict[str(cleanedOutput)]]
            if newItem != '0':
            	song.append(newItem)
        
        #print(newItem)
        #print(closestVec(output[0][i]))
pickleSave('song.pickle', song )

