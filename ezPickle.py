import pickle
def save(object,  filename):
    pickleOut = open('pickledObjects/' + filename + '.pickle',  'wb')
    pickle.dump(object,  pickleOut)
    pickleOut.close()

def load(filename):
    pickleIn = open('pickledObjects/' + filename + '.pickle',  'rb')
    temp = pickle.load(pickleIn)
    pickleIn.close()
    return temp
