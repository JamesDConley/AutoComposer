import pickle,  music21
import numpy as np
import math
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
    newVec = []
    for i in range(10):
        newVec = newVecs[0][i]
        if math.random() > .3:
            return newVec
    return newVec
    return w2v[newVec[0][0]]
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
print("started")
model = pickleLoad("kerasTrained.pickle")
song = []
for i in range(10):
    seed = 2* np.random.random(20) - 1
    song.append(closestVec(seed))

for i in range(200):
    tempSong = np.array(padZeros(song,  maxLen)).reshape(1, maxLen, 20)
    output = model.predict(tempSong)
    #for j in range(i+2):
    #    print(j)
    #    print(closestVec(output[0][j]))
    song.append(randCloseVec(output[0][i]))
    
    #print(closestVec(output[0][i]))
pickleSave('song.pickle', song )

