from gensim.models import Word2Vec
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
songs = pickleLoad('songList.pickle')


w2v = Word2Vec(sg=1,  seed = 1,  size=20, window=8, min_count=1, workers=7)
w2v.build_vocab(songs)
w2v.train(songs,  total_examples = w2v.corpus_count,  epochs = 100)
pickleSave('w2v.pickle',w2v )
print(len(w2v.wv.vocab))
