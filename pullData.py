import pickle
def pickleSave(filename,  object):
    pickleOut = open('pickledObjects/' + filename,  'wb')
    pickle.dump(object,  pickleOut)
    pickleOut.close()

def pickleLoad(filename):
    pickleIn = open('pickledObjects/' + filename,  'rb')
    temp = pickle.load(pickleIn)
    pickleIn.close()
    return temp
from music21 import corpus,  note,  chord, instrument
bachSongs = corpus.getComposer('bach')
print(len(bachSongs))
songs = []
for song in bachSongs:
    song = corpus.parse(song)
    parts = instrument.partitionByInstrument(song)
    if parts:
        notes = parts.parts[0].recurse()
    else:
        notes = song.flat.notes
    currentSong = []
    #This snippet for flattening based on a web tutorial
    for item in notes:
        if isinstance(item, note.Note):
            currentSong.append(str(item.pitch)+":"+str(item.quarterLength))   #For melody I am recording octave information
        if isinstance(item,  chord.Chord):
            currentSong.append('.'.join(str(n) for n in item.normalOrder)+":"+str(item.quarterLength))  #For harmony I am only recording the 12 tone values
    songs.append(currentSong)
pickleSave('songList.pickle',  songs)
#print(songs[0:3])
            
