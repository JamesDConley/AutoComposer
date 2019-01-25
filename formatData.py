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
from music21 import corpus,  note,  chord
bachSongs = corpus.getComposer('bach')
print(len(bachSongs))
songs = []
for song in bachSongs:
    song = corpus.parse(song)
    notes = song.flat.notes
    currentSong = []
    #This snippet for flattening based on a web tutorial
    for item in notes:
        if isinstance(item, note.Note):
            currentSong.append(str(item.pitch))    #For melody I am recording octave information
        if isinstance(item,  chord.Chord):
            currentSong.append('.'.join(str(n) for n in item.normalOrder))  #For harmony I am only recording the 12 tone values
    songs.append(currentSong)
pickleSave('songList.pickle',  songs)
print(songs[0:3])
            
