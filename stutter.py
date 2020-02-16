from random import gauss
import numpy as np
import json
import spacy

path = 'models/en_vectors_web_lg-2.1.0/en_vectors_web_lg/en_vectors_web_lg-2.1.0/'
# nlp = spacy.load('models/en_core_web_sm-2.2.5/en_core_web_sm/en_core_web_sm-2.2.5/')
nlp = spacy.load(path)
with open(path+'vocab/strings.json') as f:
    vocab = spacy.vocab.Vocab(json.loads(f.read()))
vectors = spacy.vectors.Vectors(data=np.load(path+'vocab/vectors'))

VOWELS = 'aeiouy'
def hyphen8(word):
   first_vowels = list(filter(lambda x: x!=-1, [word.find(v) for v in VOWELS]))
   return '-'.join((int(abs(gauss(0, 2)))+1) * [word[:min(first_vowels)]]) + word[min(first_vowels):] if (len(word)>2 and len(first_vowels)>0 and word[0].isalpha()) and word[0].lower() not in VOWELS else word

with open("h-aha.txt", 'r') as f:
    text = f.read()
    parsed = nlp(text)
    for token in parsed[:20]:
        print(token.vector.shape)
        print(vectors.shape)
        print(vectors.most_similar(token.vector))
    # words = text.split(' ')
    # words = map(hyphen8, words)
    # print(' '.join(words))
