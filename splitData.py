#split the training data into batches
import ezPickle as p
import math
print("loading data")
inputData = p.load('inputData')
outputData = p.load('outputData')
print('Size is : ',  len(inputData))
#Split into Sets of 12 songs
numSongs = 12
sets = math.floor(len(inputData)/numSongs)
tempInput = []
tempOutput = []
#making output
numSongs = 12
for i in range(sets):
    start = i*numSongs
    end = (i+1)*numSongs
    tempInput = inputData[start:end]
    tempOutput = outputData[start:end]
    p.save(tempInput, 'inputData'+str(i))
    p.save(tempOutput, 'outputData'+str(i))
    if end >= len(inputData):
        tempInput = inputData[start:]
        tempOutput = outputData[start:]
        p.save(tempInput, 'inputData'+str(i))
        p.save(tempOutput, 'outputData'+str(i))
print(inputData[0])
