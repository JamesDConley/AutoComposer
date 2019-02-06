#split the training data into batches
import ezPickle as p
import math
print("loading data")
inputData = p.load('inputData')
outputData = p.load('outputData')
print('Size is : ',  len(inputData))
#Split into Sets of 10 songs
sets = math.floor(len(inputData)/10)
tempInput = []
tempOutput = []
#making output
numSongs = 10
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
