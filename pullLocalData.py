from music21 import corpus, stream, instrument, converter, note,  chord
import glob, music21
import ezPickle as p
print("Number of Songs is: ", len(glob.glob("songs/*.mid")))
count = 0
songs = []
#This section is largely based on a web tutorial 
for file in glob.glob("songs/*.mid"):
        song = converter.parse(file)    
        parts = instrument.partitionByInstrument(song)
        if parts:
                notes = parts.parts[0].recurse()
        else:
                notes = song.flat.notes
        currentSong = []
        
        for item in notes:
                if isinstance(item, note.Note):
                        if item.isNote:
                                currentSong.append(str(item.pitch)+":"+str(item.quarterLength))   #For melody I am recording octave information
                        else:
                                currentSong.append('rest:'+str(item.quarterLength))
                if isinstance(item,  chord.Chord):
                        currentSong.append('.'.join(str(n) for n in item.normalOrder)+":"+str(item.quarterLength))  #For harmony I am only recording the 12 tone values
        songs.append(currentSong)
p.save(songs,'songList')
#print(songs[0:3])   
                
