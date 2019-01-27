import pickle,  music21
from music21 import stream,  note,  chord
from gensim.models import Word2Vec


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
def closestWord(vec):
    newVec = w2v.wv.most_similar( [vec], [], 1)
    print(newVec)
    return newVec[0][0]
    
song = pickleLoad('song.pickle')
music21.environment.set('musicxmlPath', '/usr/bin/musescore')
s = stream.Stream()
for vec in song[100:]:
    
    item = closestWord(vec)
    print(item)
    if '.' in item:
        print("Chord")
        current_chord = map(int, item.split('.'))
        s.append(chord.Chord(current_chord,  type='eighth'))
    else:
        print("Note")
        s.append(note.Note(item,  type='eighth'))
docvariant = s.activateVariants('docvariants')
s.show()
