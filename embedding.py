#!/usr/bin/env python3
import logging, time
from gensim.models import KeyedVectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

sTime = time.time()
# Access vectors for specific words with a keyed lookup:
vector = model['easy']
print(vector)
# see the shape of the vector (300,)
print(vector.shape)
# Processing sentences is not as simple as with Spacy:
vectors = [model[x] for x in "This is some text I am processing with Spacy".split(' ')]
print(vectors)

eTime = time.time()
print("Cost time %s." %(sTime-eTime))
