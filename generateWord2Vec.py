from gensim.models import Word2Vec
import ezPickle as p
size = 20

songs = p.load('songList')


w2v = Word2Vec(sg=1,  seed = 1,  size=size, window=8, min_count=0, workers=2)
w2v.build_vocab(songs)
w2v.train(songs,  total_examples = w2v.corpus_count,  epochs = 100)
p.save(w2v, 'w2v')
p.save(size,"size")
print(len(w2v.wv.vocab))
